"""SlowFast video inference with person boxes + action labels."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import tempfile
import threading
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
DET_THRESHOLD = 0.7
TARGET_SIZE = (160, 160)
# Activity needs high confidence; posture can be a bit lower
CONFIDENCE_THRESHOLD = 0.70
POSTURE_THRESHOLD = 0.50
ACTIVITY_THRESHOLD = 0.60

for d in (TEST_INPUT_DIR, TEST_OUTPUT_DIR, TEST_JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def list_checkpoints() -> List[Dict[str, Any]]:
    """List available SlowFast .pth and .onnx files under work_dirs."""
    work = config.WORK_DIR
    items = []
    if not work.exists():
        return items
    files = list(work.glob("*.pth")) + list(work.glob("*.onnx"))
    for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        fmt = "onnx" if path.suffix.lower() == ".onnx" else "pth"
        items.append(
            {
                "id": path.name,
                "path": str(path),
                "name": path.name,
                "format": fmt,
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


def _job_source_name(job: Dict[str, Any]) -> str:
    video = job.get("video")
    if video:
        return Path(video).name
    for line in job.get("log") or []:
        if isinstance(line, str) and line.startswith("[infer] video="):
            return line.split("=", 1)[1].strip()
    return job.get("job_id") or "Unknown"


def _job_has_output(job: Dict[str, Any]) -> bool:
    job_id = job.get("job_id") or ""
    out_path = Path(job.get("output_video") or TEST_OUTPUT_DIR / f"{job_id}.mp4")
    return bool(job_id and out_path.exists())


def _job_library_entry(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job.get("job_id") or ""
    out_path = Path(job.get("output_video") or TEST_OUTPUT_DIR / f"{job_id}.mp4")
    checkpoint = job.get("checkpoint") or ""
    ckpt_name = Path(checkpoint).name if checkpoint else "—"
    persons = job.get("persons") or []
    status = job.get("status") or "unknown"
    if out_path.exists() and status not in ("failed",):
        status = "completed"
    labels = []
    for p in persons:
        parts = [p.get("posture"), p.get("activity")]
        label = " + ".join(x for x in parts if x) or p.get("label") or ""
        if label and label not in labels:
            labels.append(label)
    summary = ", ".join(labels[:3])
    if len(labels) > 3:
        summary += f" +{len(labels) - 3}"
    return {
        "job_id": job_id,
        "status": status,
        "source_name": _job_source_name(job),
        "checkpoint": ckpt_name,
        "finished_at": job.get("finished_at") or job.get("updated_at") or job.get("created_at"),
        "created_at": job.get("created_at"),
        "num_frames": job.get("num_frames"),
        "fps": job.get("fps"),
        "person_count": len(persons),
        "summary": summary or "No detections",
        "output_url": f"/api/test/result/{job_id}/video",
        "has_output": out_path.exists(),
    }


def delete_test_input(filename: str) -> Dict[str, Any]:
    """Delete a previously uploaded test video from TEST_INPUT_DIR."""
    name = Path(filename).name
    if name != filename or ".." in filename or "/" in filename or "\\" in filename:
        return {"ok": False, "error": "Invalid filename"}
    path = TEST_INPUT_DIR / name
    if not path.exists() or not path.is_file():
        return {"ok": False, "error": "Test upload not found"}
    try:
        path.unlink()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "filename": name}


def delete_test_result(job_id: str) -> Dict[str, Any]:
    """Delete a test job record and its output video."""
    job_id = (job_id or "").strip()
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        return {"ok": False, "error": "Invalid job id"}
    job = get_test_job(job_id)
    if not job:
        return {"ok": False, "error": "Result not found"}

    removed: List[str] = []
    out_path = Path(job.get("output_video") or TEST_OUTPUT_DIR / f"{job_id}.mp4")
    # Only delete outputs that live under our test dirs
    for candidate in (
        out_path,
        TEST_OUTPUT_DIR / f"{job_id}.mp4",
        TEST_OUTPUT_DIR / f"{job_id}_raw.mp4",
        TEST_OUTPUT_DIR / f"{job_id}.json",
    ):
        try:
            resolved = candidate.resolve()
            if (
                resolved.exists()
                and resolved.is_file()
                and (
                    str(resolved).startswith(str(TEST_OUTPUT_DIR.resolve()))
                    or str(resolved).startswith(str(TEST_JOBS_DIR.resolve()))
                )
            ):
                resolved.unlink()
                removed.append(resolved.name)
        except OSError:
            continue

    job_path = _job_path(job_id)
    try:
        if job_path.exists():
            job_path.unlink()
            removed.append(job_path.name)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "removed": removed}

    return {"ok": True, "job_id": job_id, "removed": removed}


def list_test_inputs(
    page: int = 1,
    per_page: int = 50,
    q: str = "",
) -> Dict[str, Any]:
    """List videos previously uploaded for Test (not Label library)."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    items: List[Dict[str, Any]] = []
    if TEST_INPUT_DIR.exists():
        for path in TEST_INPUT_DIR.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in video_exts:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            items.append(
                {
                    "filename": path.name,
                    "id": path.stem,
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                }
            )
    items.sort(key=lambda x: x["mtime"], reverse=True)

    query = (q or "").strip().lower()
    if query:
        items = [it for it in items if query in it["filename"].lower() or query in it["id"].lower()]

    total = len(items)
    per_page = max(1, min(per_page, 200))
    page = max(1, page)
    start = (page - 1) * per_page
    page_items = items[start : start + per_page]
    pages = max(1, (total + per_page - 1) // per_page) if total else 1
    return {
        "videos": page_items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }


def list_test_library(
    page: int = 1,
    per_page: int = 50,
    q: str = "",
) -> Dict[str, Any]:
    """List completed inference outputs for the test-page library."""
    items: List[Dict[str, Any]] = []
    for p in sorted(TEST_JOBS_DIR.glob("*.json"), reverse=True):
        with open(p, "r", encoding="utf-8") as f:
            job = json.load(f)
        if not _job_has_output(job):
            continue
        items.append(_job_library_entry(job))

    query = (q or "").strip().lower()
    if query:
        items = [
            item
            for item in items
            if query in item["source_name"].lower()
            or query in item["job_id"].lower()
            or query in item["checkpoint"].lower()
            or query in item["summary"].lower()
        ]

    total = len(items)
    per_page = max(1, min(per_page, 200))
    page = max(1, page)
    start = (page - 1) * per_page
    page_items = items[start : start + per_page]
    pages = max(1, (total + per_page - 1) // per_page)
    return {
        "results": page_items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }


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


def _load_label_taxonomy() -> Tuple[List[str], List[str], List[str]]:
    labels_file = config.DATA_DIR / "labels.json"
    labels = list(config.ACTION_LABELS)
    postures = list(config.POSTURE_LABELS)
    activities = list(config.ACTIVITY_LABELS)
    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("labels"):
                labels = list(data["labels"])
            if data.get("postures"):
                postures = list(data["postures"])
            if data.get("activities"):
                activities = list(data["activities"])
        except (OSError, json.JSONDecodeError):
            pass
    return labels, postures, activities


def _cfg_is_multilabel(cfg) -> bool:
    try:
        head = cfg.model.get("cls_head", {}) if hasattr(cfg, "model") else {}
        loss = head.get("loss_cls") or {}
        if isinstance(loss, dict) and "BCE" in str(loss.get("type", "")):
            return True
        dataset = cfg.train_dataloader.get("dataset", {})
        return bool(dataset.get("multi_class"))
    except Exception:
        return False


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (e.sum() + 1e-8)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def _best_in_group(
    labels: List[str],
    probs: np.ndarray,
    group: List[str],
    threshold: float,
) -> Tuple[str, float]:
    idxs = [i for i, name in enumerate(labels) if name in group and i < len(probs)]
    if not idxs:
        return "", 0.0
    best = max(idxs, key=lambda i: float(probs[i]))
    score = float(probs[best])
    if score < threshold:
        return "", score
    return labels[best], score


def _compose_prediction(posture: str, p_score: float, activity: str, a_score: float) -> Dict[str, Any]:
    # Walking already implies locomotion; don't also print standing.
    show_posture = posture if activity != "walking" else ""
    parts = [p for p in (show_posture, activity) if p]
    display = " + ".join(parts) if parts else "unknown"
    if activity:
        score = a_score if a_score > 0 else p_score
    else:
        score = p_score
    return {
        "label": display,
        "posture": show_posture,
        "posture_score": round(p_score, 4),
        "activity": activity,
        "activity_score": round(a_score, 4),
        "score": round(float(score), 4),
    }


def _decode_posture_activity(
    scores: np.ndarray,
    labels: List[str],
    postures: List[str],
    activities: List[str],
    multi_label: bool,
) -> Dict[str, Any]:
    n = min(len(scores), len(labels))
    raw = np.asarray(scores[:n], dtype=np.float32)
    looks_softmax = raw.min() >= 0 and abs(float(raw.sum()) - 1.0) < 0.2
    looks_logits = raw.min() < 0 or raw.max() > 1.5
    if looks_softmax:
        probs = raw / (raw.sum() + 1e-8)
        p_thr, a_thr = 0.15, ACTIVITY_THRESHOLD
    elif multi_label or not looks_logits:
        probs = _sigmoid(raw) if looks_logits else raw
        p_thr, a_thr = POSTURE_THRESHOLD, ACTIVITY_THRESHOLD
    else:
        probs = _softmax(raw)
        p_thr, a_thr = 0.15, ACTIVITY_THRESHOLD
    posture, p_score = _best_in_group(labels[:n], probs, postures, p_thr)
    activity, a_score = _best_in_group(labels[:n], probs, activities, a_thr)
    return _compose_prediction(posture, p_score, activity, a_score)


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


IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1, 1)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1, 1)


def _crops_to_ncthw(crops: List[np.ndarray]) -> Optional[np.ndarray]:
    valid = [c for c in crops if c is not None and c.size > 0 and c.shape[0] > 1 and c.shape[1] > 1]
    if len(valid) < MIN_FRAMES:
        return None
    frames = [cv2.resize(c, TARGET_SIZE) for c in valid[:CLIP_LEN]]
    while len(frames) < CLIP_LEN:
        frames.append(frames[-1])
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    arr = np.stack(rgb, axis=0).astype(np.float32)
    arr = np.transpose(arr, (3, 0, 1, 2))[None, ...]
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


class PthActionBackend:
    def __init__(self, model):
        self.model = model
        self.kind = "pth"

    def scores_from_crops(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        from mmaction.apis import inference_recognizer

        valid = [c for c in crops if c is not None and c.size > 0 and c.shape[0] > 1 and c.shape[1] > 1]
        if len(valid) < MIN_FRAMES:
            return None
        resized = [cv2.resize(c, TARGET_SIZE) for c in valid[:CLIP_LEN]]
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, TARGET_SIZE)
        for frame in resized:
            writer.write(frame)
        writer.release()
        try:
            result = inference_recognizer(self.model, tmp_path)
            return result.pred_score.detach().float().cpu().numpy().reshape(-1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            torch.cuda.empty_cache()
            gc.collect()


class OnnxActionBackend:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.kind = "onnx"
        self.providers = providers

    def scores_from_crops(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        tensor = _crops_to_ncthw(crops)
        if tensor is None:
            return None
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)


def load_action_backend(model_path: Path, labels: List[str], log=None):
    """Load SlowFast from .pth (MMAction2) or .onnx (ONNX Runtime)."""
    from mmengine.config import Config

    cfg = Config.fromfile(str(ACTION_CONFIG))
    if hasattr(cfg, "model") and "cls_head" in cfg.model:
        cfg.model["cls_head"]["num_classes"] = len(labels)
    multi_label = _cfg_is_multilabel(cfg)
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        backend = OnnxActionBackend(model_path)
        if log:
            log(f"[infer] SlowFast ONNX ({', '.join(backend.providers)})")
        return backend, multi_label
    from mmaction.apis import init_recognizer

    device_action = "cuda:0" if torch.cuda.is_available() else "cpu"
    action_model = init_recognizer(cfg, str(model_path), device=device_action)
    action_model.eval()
    if log:
        log(f"[infer] SlowFast .pth on {device_action} (multi_label={multi_label})")
    return PthActionBackend(action_model), multi_label


def _classify_crops(
    backend,
    crops: List[np.ndarray],
    labels: List[str],
    postures: List[str],
    activities: List[str],
    multi_label: bool,
) -> Dict[str, Any]:
    empty = _compose_prediction("", 0.0, "", 0.0)
    scores = backend.scores_from_crops(crops)
    if scores is None:
        return empty
    return _decode_posture_activity(scores, labels, postures, activities, multi_label)


_live_lock = threading.Lock()
_live_bundle: Dict[str, Any] = {}


def _get_live_models(checkpoint_path: Path) -> Dict[str, Any]:
    """Load detector + SlowFast once and reuse for camera clips."""
    global _live_bundle
    ckpt = str(checkpoint_path.resolve())
    if _live_bundle.get("checkpoint") == ckpt and _live_bundle.get("action_backend") is not None:
        return _live_bundle

    from mmdet.apis import init_detector
    from mmengine import init_default_scope

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")
    labels, postures, activities = _load_label_taxonomy()
    backend, multi_label = load_action_backend(checkpoint_path, labels)
    init_default_scope("mmdet")
    det_model = init_detector(str(DET_CONFIG), str(DET_CHECKPOINT), device="cpu")
    _live_bundle = {
        "checkpoint": ckpt,
        "action_backend": backend,
        "action_model": backend,
        "det_model": det_model,
        "labels": labels,
        "postures": postures,
        "activities": activities,
        "multi_label": multi_label,
        "device": getattr(backend, "kind", "pth"),
    }
    return _live_bundle


def run_live_clip(frames: List[np.ndarray], checkpoint_path: Path) -> Dict[str, Any]:
    """Detect people on the last frame and classify a SlowFast clip per person."""
    if not frames:
        return {"ok": False, "error": "No frames", "persons": []}
    if not checkpoint_path.exists():
        return {"ok": False, "error": f"Checkpoint not found: {checkpoint_path.name}", "persons": []}
    if not DET_CHECKPOINT.exists():
        return {"ok": False, "error": f"Detector missing: {DET_CHECKPOINT}", "persons": []}

    with _live_lock:
        bundle = _get_live_models(checkpoint_path)
        last = frames[-1]
        boxes = _detect_persons(bundle["det_model"], last)
        persons = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crops = []
            for frame in frames:
                h, w = frame.shape[:2]
                xa, ya = max(0, x1), max(0, y1)
                xb, yb = min(w, x2), min(h, y2)
                if xb <= xa or yb <= ya:
                    continue
                crops.append(frame[ya:yb, xa:xb])
            pred = _classify_crops(
                bundle["action_backend"],
                crops,
                bundle["labels"],
                bundle["postures"],
                bundle["activities"],
                bundle["multi_label"],
            )
            persons.append(
                {
                    "id": i,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "label": pred.get("label") or "unknown",
                    "posture": pred.get("posture") or "",
                    "activity": pred.get("activity") or "",
                    "score": pred.get("score", 0),
                    "posture_score": pred.get("posture_score", 0),
                    "activity_score": pred.get("activity_score", 0),
                    "frames": len(crops),
                }
            )
    h, w = frames[-1].shape[:2]
    return {
        "ok": True,
        "width": int(w),
        "height": int(h),
        "persons": persons,
        "checkpoint": checkpoint_path.name,
    }


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
    from mmdet.apis import init_detector
    from mmengine import init_default_scope

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")

    labels, postures, activities = _load_label_taxonomy()
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
    log(f"[infer] postures={postures}")
    log(f"[infer] activities={activities}")
    log(f"[infer] activity_threshold={ACTIVITY_THRESHOLD:.0%} posture_threshold={POSTURE_THRESHOLD:.0%}")

    action_backend, multi_label = load_action_backend(checkpoint_path, labels, log=log)

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
    track_frame_labels: Dict[int, Dict[int, Dict[str, Any]]] = {}
    summary: Dict[int, Dict[str, Any]] = {}

    for tid, td in tracker.tracks.items():
        seq = td["frames"]
        if len(seq) < MIN_FRAMES:
            continue
        frame_labels: Dict[int, Dict[str, Any]] = {}
        starts = list(range(0, max(1, len(seq) - CLIP_LEN + 1), WINDOW_STRIDE))
        if not starts:
            starts = [0]
        posture_votes: Dict[str, float] = {}
        activity_votes: Dict[str, float] = {}
        for s in starts:
            window = seq[s : s + CLIP_LEN]
            crops = []
            for fidx, box in window:
                x1, y1, x2, y2 = box
                crop = frames[fidx][max(0, y1) : y2, max(0, x1) : x2]
                crops.append(crop)
            pred = _classify_crops(
                action_backend, crops, labels, postures, activities, multi_label
            )
            for fidx, _ in window:
                frame_labels[fidx] = pred
            if pred.get("posture"):
                posture_votes[pred["posture"]] = posture_votes.get(pred["posture"], 0.0) + pred.get(
                    "posture_score", 0.0
                )
            if pred.get("activity"):
                activity_votes[pred["activity"]] = activity_votes.get(pred["activity"], 0.0) + pred.get(
                    "activity_score", 0.0
                )
        track_frame_labels[tid] = frame_labels
        best_posture, p_score = ("", 0.0)
        best_activity, a_score = ("", 0.0)
        if posture_votes:
            best_posture = max(posture_votes.items(), key=lambda x: x[1])[0]
            p_score = posture_votes[best_posture] / max(1, len(starts))
        if activity_votes:
            best_activity = max(activity_votes.items(), key=lambda x: x[1])[0]
            a_score = activity_votes[best_activity] / max(1, len(starts))
        composed = _compose_prediction(best_posture, p_score, best_activity, a_score)
        summary[tid] = {
            **composed,
            "frames": len(seq),
        }
        log(f"[infer] person {tid}: {composed['label']} ({composed['score']:.2f})")

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
            pred = None
            if tid is not None and tid in track_frame_labels:
                fl = track_frame_labels[tid]
                if i in fl:
                    pred = fl[i]
                else:
                    nearby = [k for k in fl if abs(k - i) <= WINDOW_STRIDE]
                    if nearby:
                        k = min(nearby, key=lambda z: abs(z - i))
                        pred = fl[k]
                    elif tid in summary:
                        pred = summary[tid]
            elif tid is not None and tid in summary:
                pred = summary[tid]
            if pred:
                lab = pred.get("label") or "unknown"
                conf = float(pred.get("score") or 0.0)
                if lab == "unknown":
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
    del action_backend
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
            {
                "id": tid,
                "label": info.get("label"),
                "posture": info.get("posture") or "",
                "activity": info.get("activity") or "",
                "score": info.get("score", 0),
                "posture_score": info.get("posture_score", 0),
                "activity_score": info.get("activity_score", 0),
                "frames": info.get("frames"),
            }
            for tid, info in summary.items()
        ],
        "num_frames": len(frames),
        "fps": fps,
        "log": log_lines[-80:],
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_job(job_id, result)
    print(f"[infer] done → {final_out.name}")
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
