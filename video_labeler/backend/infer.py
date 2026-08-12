"""SlowFast video inference with person boxes + action labels."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from . import config

TEST_DIR = config.DATA_DIR / "tests"
TEST_INPUT_DIR = TEST_DIR / "inputs"
TEST_OUTPUT_DIR = TEST_DIR / "outputs"
TEST_JOBS_DIR = TEST_DIR / "jobs"

DET_CONFIG = config.PROJECT_ROOT / "mmdetection" / "configs" / "faster_rcnn" / "faster-rcnn_r50_fpn_1x_coco.py"
DET_CHECKPOINT = config.PROJECT_ROOT / "checkpoints" / "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
ACTION_CONFIG = config.TRAIN_CONFIG

CLIP_LEN = 16
MIN_FRAMES = 5
DETECT_EVERY = 2  # run detector every N frames (speed)
WINDOW_STRIDE = 8
DET_THRESHOLD = 0.6
TARGET_SIZE = (160, 160)
# Only accept action predictions at or above this confidence
CONFIDENCE_THRESHOLD = 0.70

for d in (TEST_INPUT_DIR, TEST_OUTPUT_DIR, TEST_JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def list_checkpoints() -> List[Dict[str, Any]]:
    """List available SlowFast checkpoints under work_dirs."""
    work = config.WORK_DIR
    items = []
    if not work.exists():
        return items
    for path in sorted(work.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
        items.append(
            {
                "id": path.name,
                "path": str(path),
                "name": path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "recommended": path.name.startswith("best_acc") or path.name == "epoch_50.pth",
            }
        )
    return items


def _job_path(job_id: str) -> Path:
    return TEST_JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, data: Dict[str, Any]) -> None:
    data["job_id"] = job_id
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(_job_path(job_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_test_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_test_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    jobs = []
    for p in sorted(TEST_JOBS_DIR.glob("*.json"), reverse=True)[:limit]:
        with open(p, "r", encoding="utf-8") as f:
            jobs.append(json.load(f))
    return jobs


class _PersonTracker:
    def __init__(self, iou_threshold: float = 0.2, max_missing: int = 15):
        self.next_id = 0
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    @staticmethod
    def iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, frame_idx: int, detections: List[Tuple[int, int, int, int]]):
        assigned = set()
        for box in detections:
            best_id, best_iou = None, self.iou_threshold
            for tid, td in self.tracks.items():
                if tid in assigned:
                    continue
                if frame_idx - td["last_frame"] > self.max_missing:
                    continue
                iou = self.iou(box, td["last_bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                assigned.add(best_id)
                self.tracks[best_id]["last_bbox"] = box
                self.tracks[best_id]["last_frame"] = frame_idx
                self.tracks[best_id]["frames"].append((frame_idx, box))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "last_bbox": box,
                    "last_frame": frame_idx,
                    "frames": [(frame_idx, box)],
                }


def _load_labels_from_config() -> List[str]:
    labels_file = config.DATA_DIR / "labels.json"
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("labels"):
            return list(data["labels"])
    return list(config.ACTION_LABELS)


def _detect_persons(det_model, frame) -> List[Tuple[int, int, int, int]]:
    from mmdet.apis import inference_detector
    from mmengine import init_default_scope

    init_default_scope("mmdet")
    result = inference_detector(det_model, frame)
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    persons = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if int(label) == 0 and float(score) >= DET_THRESHOLD:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 - x1 > 10 and y2 - y1 > 10:
                persons.append((x1, y1, x2, y2))
    return persons


def _classify_crops(action_model, crops: List[np.ndarray], labels: List[str]) -> Tuple[str, float]:
    from mmaction.apis import inference_recognizer

    valid = [c for c in crops if c is not None and c.size > 0 and c.shape[0] > 1 and c.shape[1] > 1]
    if len(valid) < MIN_FRAMES:
        return "unknown", 0.0

    resized = [cv2.resize(c, TARGET_SIZE) for c in valid[:CLIP_LEN]]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, TARGET_SIZE)
    for frame in resized:
        writer.write(frame)
    writer.release()

    try:
        result = inference_recognizer(action_model, tmp_path)
        scores = result.pred_score.detach().float().cpu().numpy().reshape(-1)
        # Softmax if not already probabilities
        if scores.min() < 0 or scores.max() > 1.5:
            exp = np.exp(scores - scores.max())
            probs = exp / exp.sum()
        else:
            probs = scores / (scores.sum() + 1e-8)
        idx = int(np.argmax(probs[: len(labels)]))
        conf = float(probs[idx])
        # Reject low-confidence predictions — not a reliable pose/action
        if conf < CONFIDENCE_THRESHOLD:
            return "unknown", conf
        return labels[idx], conf
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        torch.cuda.empty_cache()
        gc.collect()


def _to_h264(src: Path, dest: Path) -> Path:
    if not shutil.which("ffmpeg"):
        shutil.copy2(src, dest)
        return dest
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists():
        shutil.copy2(src, dest)
    return dest


def run_inference(
    video_path: Path,
    checkpoint_path: Path,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run detect → SlowFast → annotated video. Updates job file if job_id given."""
    from mmengine.config import Config
    from mmaction.apis import init_recognizer
    from mmdet.apis import init_detector
    from mmengine import init_default_scope

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")

    labels = _load_labels_from_config()
    job_id = job_id or datetime.utcnow().strftime("infer_%Y%m%d_%H%M%S")
    log_lines: List[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)
        if job_id:
            job = get_test_job(job_id) or {"job_id": job_id, "status": "running"}
            job["status"] = "running"
            job["log"] = log_lines[-80:]
            _write_job(job_id, job)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not DET_CHECKPOINT.exists():
        raise FileNotFoundError(f"Detector checkpoint missing: {DET_CHECKPOINT}")

    log(f"[infer] video={video_path.name}")
    log(f"[infer] checkpoint={checkpoint_path.name}")
    log(f"[infer] labels={labels}")
    log(f"[infer] confidence_threshold={CONFIDENCE_THRESHOLD:.0%}")

    # Load models
    device_action = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = Config.fromfile(str(ACTION_CONFIG))
    # Keep head size consistent with checkpoint / labels
    if hasattr(cfg, "model") and "cls_head" in cfg.model:
        cfg.model["cls_head"]["num_classes"] = len(labels)
    action_model = init_recognizer(cfg, str(checkpoint_path), device=device_action)
    action_model.eval()
    log(f"[infer] SlowFast on {device_action}")

    init_default_scope("mmdet")
    det_model = init_detector(str(DET_CONFIG), str(DET_CHECKPOINT), device="cpu")
    log("[infer] Detector on CPU")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    log(f"[infer] loaded {len(frames)} frames @ {fps:.1f}fps")

    tracker = _PersonTracker()
    detections_per_frame: List[List[Tuple[int, int, int, int]]] = []
    last_boxes: List[Tuple[int, int, int, int]] = []

    for i, frame in enumerate(frames):
        if i % DETECT_EVERY == 0:
            last_boxes = _detect_persons(det_model, frame)
        detections_per_frame.append(list(last_boxes))
        tracker.update(i, last_boxes)
        if (i + 1) % 50 == 0:
            log(f"[infer] detect {i + 1}/{len(frames)}")

    # Build per-track crop sequences and classify with sliding windows
    track_frame_labels: Dict[int, Dict[int, Tuple[str, float]]] = {}
    summary: Dict[int, Dict[str, Any]] = {}

    for tid, td in tracker.tracks.items():
        seq = td["frames"]
        if len(seq) < MIN_FRAMES:
            continue
        frame_labels: Dict[int, Tuple[str, float]] = {}
        # Sliding windows along the track
        starts = list(range(0, max(1, len(seq) - CLIP_LEN + 1), WINDOW_STRIDE))
        if not starts:
            starts = [0]
        votes: Dict[str, float] = {}
        for s in starts:
            window = seq[s : s + CLIP_LEN]
            crops = []
            for fidx, box in window:
                x1, y1, x2, y2 = box
                crop = frames[fidx][max(0, y1) : y2, max(0, x1) : x2]
                crops.append(crop)
            label, conf = _classify_crops(action_model, crops, labels)
            for fidx, _ in window:
                frame_labels[fidx] = (label, conf)
            # Only count confident action labels toward the person summary
            if label != "unknown" and conf >= CONFIDENCE_THRESHOLD:
                votes[label] = votes.get(label, 0.0) + conf
        track_frame_labels[tid] = frame_labels
        if votes:
            best = max(votes.items(), key=lambda x: x[1])
            avg_score = float(best[1] / max(1, len(starts)))
            # Final person label must still clear the threshold on average
            if avg_score >= CONFIDENCE_THRESHOLD:
                summary_label, summary_score = best[0], avg_score
            else:
                summary_label, summary_score = "unknown", avg_score
        else:
            summary_label, summary_score = "unknown", 0.0
        summary[tid] = {
            "label": summary_label,
            "score": round(summary_score, 4),
            "frames": len(seq),
        }
        log(f"[infer] person {tid}: {summary_label} ({summary_score:.2f})")

    # Annotate
    raw_out = TEST_OUTPUT_DIR / f"{job_id}_raw.mp4"
    final_out = TEST_OUTPUT_DIR / f"{job_id}.mp4"
    writer = cv2.VideoWriter(
        str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 1 else 25, (width, height)
    )
    colors = [
        (45, 180, 120),
        (40, 90, 220),
        (0, 200, 255),
        (200, 80, 200),
        (0, 165, 255),
        (80, 200, 80),
    ]

    # Map each detection box to nearest track for drawing
    for i, frame in enumerate(frames):
        vis = frame.copy()
        boxes = detections_per_frame[i]
        for bi, box in enumerate(boxes):
            # Find track owning this box at this frame
            tid = None
            for t, td in tracker.tracks.items():
                for fidx, tbox in td["frames"]:
                    if fidx == i and _PersonTracker.iou(box, tbox) > 0.5:
                        tid = t
                        break
                if tid is not None:
                    break
            color = colors[(tid if tid is not None else bi) % len(colors)]
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            text = "person"
            if tid is not None and tid in track_frame_labels:
                fl = track_frame_labels[tid]
                # nearest labeled frame
                if i in fl:
                    lab, conf = fl[i]
                else:
                    nearby = [k for k in fl if abs(k - i) <= WINDOW_STRIDE]
                    if nearby:
                        k = min(nearby, key=lambda z: abs(z - i))
                        lab, conf = fl[k]
                    elif tid in summary:
                        lab, conf = summary[tid]["label"], summary[tid]["score"]
                    else:
                        lab, conf = "unknown", 0.0
                if lab == "unknown" or conf < CONFIDENCE_THRESHOLD:
                    text = f"P{tid}: unknown"
                else:
                    text = f"P{tid}: {lab} {conf * 100:.0f}%"
            elif tid is not None and tid in summary:
                lab, conf = summary[tid]["label"], summary[tid]["score"]
                if lab == "unknown" or conf < CONFIDENCE_THRESHOLD:
                    text = f"P{tid}: unknown"
                else:
                    text = f"P{tid}: {lab} {conf * 100:.0f}%"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(0, y1 - 8)
            cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(
                vis,
                text,
                (x1 + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        writer.write(vis)
    writer.release()

    log("[infer] encoding H.264 for browser…")
    _to_h264(raw_out, final_out)
    raw_out.unlink(missing_ok=True)

    # Free models
    del action_model
    del det_model
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "ok": True,
        "job_id": job_id,
        "status": "completed",
        "output_video": str(final_out),
        "output_url": f"/api/test/result/{job_id}/video",
        "checkpoint": checkpoint_path.name,
        "labels": labels,
        "persons": [
            {"id": tid, "label": info["label"], "score": info["score"], "frames": info["frames"]}
            for tid, info in summary.items()
        ],
        "num_frames": len(frames),
        "fps": fps,
        "log": log_lines[-80:],
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_job(job_id, result)
    log(f"[infer] done → {final_out.name}")
    return result


def start_inference_job(video_path: Path, checkpoint_path: Path) -> Dict[str, Any]:
    """Spawn inference in a background process."""
    import subprocess

    job_id = datetime.utcnow().strftime("infer_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "video": str(video_path),
        "checkpoint": str(checkpoint_path),
        "log": [],
    }
    _write_job(job_id, job)

    log_path = TEST_JOBS_DIR / f"{job_id}.log"
    cmd = [
        str(config.VENV_PYTHON),
        "-m",
        "video_labeler.backend.infer",
        "--job-id",
        job_id,
        "--video",
        str(video_path),
        "--checkpoint",
        str(checkpoint_path),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(config.PROJECT_ROOT),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={
            **os.environ,
            "PYTHONPATH": str(config.PROJECT_ROOT),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64,expandable_segments:True",
        },
    )
    job["status"] = "running"
    job["pid"] = proc.pid
    job["log_path"] = str(log_path)
    _write_job(job_id, job)
    return {"ok": True, "job": job}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    try:
        run_inference(Path(args.video), Path(args.checkpoint), job_id=args.job_id)
    except Exception as exc:
        job = get_test_job(args.job_id) or {"job_id": args.job_id}
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        _write_job(args.job_id, job)
        raise


if __name__ == "__main__":
    main()
