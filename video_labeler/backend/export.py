"""Export labeled segments into SlowFast training clips + annotation lists."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from . import config


def load_annotation(video_id: str) -> Optional[Dict[str, Any]]:
    path = config.ANNOTATIONS_DIR / f"{video_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_annotations() -> Dict[str, Any]:
    """Count saved posture/activity tags across all annotation files."""
    counts = {name: 0 for name in config.ACTION_LABELS}
    other = 0
    videos_labeled = 0
    total = 0
    for path in config.ANNOTATIONS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        segs = ann.get("segments") or []
        if segs:
            videos_labeled += 1
        for seg in segs:
            names = config.segment_class_names(seg)
            if not names:
                continue
            total += 1
            for label in names:
                if label in counts:
                    counts[label] += 1
                else:
                    other += 1
                    counts.setdefault(label, 0)
                    counts[label] += 1
    return {
        "counts": counts,
        "total": total,
        "videos_labeled": videos_labeled,
        "other": other,
    }


def save_annotation(video_id: str, data: Dict[str, Any]) -> Path:
    path = config.ANNOTATIONS_DIR / f"{video_id}.json"
    data["video_id"] = video_id
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def _open_video(video_path: Path) -> Tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total


def _crop_frame(frame, bbox: Optional[List[float]], frame_w: int, frame_h: int):
    if not bbox or len(bbox) != 4:
        return frame
    x1, y1, x2, y2 = bbox
    # Support normalized [0,1] or pixel coords
    if max(x1, y1, x2, y2) <= 1.5:
        x1, x2 = x1 * frame_w, x2 * frame_w
        y1, y2 = y1 * frame_h, y2 * frame_h
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def export_segment_clip(
    video_path: Path,
    segment: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """Write one labeled segment to an mp4 clip."""
    cap, fps, width, height, total = _open_video(video_path)
    start = max(0, int(segment["start_frame"]))
    end = min(total - 1, int(segment["end_frame"]))
    if end - start + 1 < config.MIN_CLIP_FRAMES:
        cap.release()
        raise ValueError(
            f"Segment too short ({end - start + 1} frames), need >= {config.MIN_CLIP_FRAMES}"
        )

    bbox = segment.get("bbox")
    # Probe first cropped frame for writer size
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read start frame")
    sample = _crop_frame(frame, bbox, width, height)
    out_h, out_w = sample.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    writer.write(sample)

    written = 1
    for fidx in range(start + 1, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        cropped = _crop_frame(frame, bbox, width, height)
        if cropped.shape[0] != out_h or cropped.shape[1] != out_w:
            cropped = cv2.resize(cropped, (out_w, out_h))
        writer.write(cropped)
        written += 1

    writer.release()
    cap.release()
    names = config.segment_class_names(segment)
    return {
        "path": str(output_path),
        "frames": written,
        "label": names[0] if names else "",
        "labels": names,
        "start_frame": start,
        "end_frame": end,
    }


def _collect_all_segments() -> List[Dict[str, Any]]:
    items = []
    for ann_path in sorted(config.ANNOTATIONS_DIR.glob("*.json")):
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        video_id = ann.get("video_id") or ann_path.stem
        video_path = config.VIDEOS_DIR / ann.get("filename", f"{video_id}.mp4")
        if not video_path.exists():
            # try any extension
            matches = list(config.VIDEOS_DIR.glob(f"{video_id}.*"))
            if not matches:
                continue
            video_path = matches[0]
        for i, seg in enumerate(ann.get("segments", [])):
            names = config.segment_class_names(seg)
            if not names:
                continue
            items.append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "segment": seg,
                    "seg_index": i,
                    "labels": names,
                }
            )
    return items


def export_dataset(
    out_root: Optional[Path] = None,
    val_ratio: float = None,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """
    Export all annotations into train/val folders and list files
    compatible with create_clean_dataset / SlowFast VideoDataset.
    """
    out_root = Path(out_root) if out_root else config.TRAINING_VIDEOS_DIR
    # Safety: only allow clearing inside the ActionMark export root
    out_root = out_root.resolve()
    allowed_root = config.TRAINING_VIDEOS_DIR.resolve()
    if allowed_root not in out_root.parents and out_root != allowed_root:
        raise RuntimeError(f"Refusing to export outside {allowed_root}")

    val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio
    items = _collect_all_segments()
    if not items:
        return {"ok": False, "error": "No labeled segments found", "clips": 0}

    if clear_existing and out_root.exists():
        # Wipe only the ActionMark export tree (not other data folders)
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    random.Random(config.RANDOM_SEED).shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) > 4 else 0
    val_set = set(id(x) for x in items[:n_val])

    exported = []
    label_counts = {l: {"train": 0, "val": 0} for l in config.ACTION_LABELS}
    clip_records = []

    for item in items:
        names = item.get("labels") or config.segment_class_names(item["segment"])
        folder_label = names[-1] if names else "unknown"
        split = "val" if id(item) in val_set else "train"
        dest_dir = out_root / split / folder_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        clip_name = f"{item['video_id']}_seg{item['seg_index']:04d}.mp4"
        dest = dest_dir / clip_name
        try:
            info = export_segment_clip(item["video_path"], item["segment"], dest)
            info["split"] = split
            info["labels"] = names
            exported.append(info)
            clip_records.append(
                {
                    "split": split,
                    "folder": folder_label,
                    "filename": clip_name,
                    "labels": names,
                }
            )
            for name in names:
                if name not in label_counts:
                    label_counts[name] = {"train": 0, "val": 0}
                label_counts[name][split] += 1
        except Exception as exc:
            exported.append(
                {
                    "error": str(exc),
                    "video_id": item["video_id"],
                    "seg_index": item["seg_index"],
                    "labels": names,
                }
            )

    # Write labels.txt and list files (single-label index format)
    out_root.mkdir(parents=True, exist_ok=True)
    labels_path = out_root / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in config.ACTION_LABELS:
            f.write(name + "\n")

    # Also write clean-style lists for SlowFast (multi-label indices)
    clean_root = config.CLEAN_VIDEOS_DIR
    _rebuild_clean_lists(out_root, clean_root, clip_records)

    ok_clips = [e for e in exported if "error" not in e]
    summary = {
        "ok": True,
        "clips": len(ok_clips),
        "errors": len(exported) - len(ok_clips),
        "label_counts": label_counts,
        "out_root": str(out_root),
        "labels": config.ACTION_LABELS,
        "exported_at": datetime.utcnow().isoformat() + "Z",
    }
    summary_path = config.EXPORTS_DIR / f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": exported}, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def _rebuild_clean_lists(
    source_root: Path,
    clean_root: Path,
    clip_records: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Symlink/copy clips into clean layout and write train_list.txt / val_list.txt."""
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True, exist_ok=True)

    label_to_idx = {name: i for i, name in enumerate(config.ACTION_LABELS)}

    records = clip_records or []
    if not records:
        # Fallback: infer a single label from folder name
        for split in ("train", "val"):
            for label in config.ACTION_LABELS:
                src_dir = source_root / split / label
                if not src_dir.exists():
                    continue
                for video_file in sorted(src_dir.glob("*.mp4")):
                    records.append(
                        {
                            "split": split,
                            "folder": label,
                            "filename": video_file.name,
                            "labels": [label],
                        }
                    )

    counters = {("train", l): 0 for l in config.ACTION_LABELS}
    counters.update({("val", l): 0 for l in config.ACTION_LABELS})
    lines_by_split = {"train": [], "val": []}

    for rec in records:
        split = rec["split"]
        folder = rec.get("folder") or (rec["labels"][0] if rec.get("labels") else "unknown")
        src = source_root / split / folder / rec["filename"]
        if not src.exists():
            continue
        dest_dir = clean_root / split / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        idx = counters.get((split, folder), 0)
        counters[(split, folder)] = idx + 1
        clean_name = f"video_{idx:05d}{src.suffix}"
        target = dest_dir / clean_name
        try:
            target.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, target)
        names = [n for n in rec.get("labels") or [] if n in label_to_idx]
        if not names:
            continue
        indices = " ".join(str(label_to_idx[n]) for n in names)
        lines_by_split[split].append(f"{split}/{folder}/{clean_name} {indices}")

    for split in ("train", "val"):
        list_path = clean_root / f"{split}_list.txt"
        lines = lines_by_split[split]
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    with open(clean_root / "labels.txt", "w", encoding="utf-8") as f:
        for name in config.ACTION_LABELS:
            f.write(name + "\n")


def sync_train_config_labels() -> None:
    """Update SlowFast multilabel config ACTION_LABELS / NUM_CLASSES to match config."""
    cfg_path = config.TRAIN_CONFIG
    if not cfg_path.exists():
        return
    text = cfg_path.read_text(encoding="utf-8")
    labels_repr = json.dumps(config.ACTION_LABELS)
    # Replace ACTION_LABELS = [...] and NUM_CLASSES = N
    import re

    text2 = re.sub(
        r"ACTION_LABELS\s*=\s*\[.*?\]",
        f"ACTION_LABELS = {labels_repr}",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text2 = re.sub(
        r"NUM_CLASSES\s*=\s*\d+",
        f"NUM_CLASSES = {len(config.ACTION_LABELS)}",
        text2,
        count=1,
    )
    if text2 != text:
        cfg_path.write_text(text2, encoding="utf-8")
