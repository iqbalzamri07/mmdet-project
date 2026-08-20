"""FastAPI backend for video action labeling + SlowFast training."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .export import (
    count_annotations,
    export_dataset,
    load_annotation,
    save_annotation,
    sync_train_config_labels,
)
from .media import is_browser_playable, probe_codecs
from .infer import (
    TEST_INPUT_DIR,
    TEST_OUTPUT_DIR,
    get_test_job,
    list_checkpoints,
    list_test_jobs,
    run_live_clip,
    start_inference_job,
)
from .train_runner import get_active_job, get_job, list_jobs, start_training, stop_training

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="ActionMark", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Segment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: Optional[str] = ""
    posture: Optional[str] = ""
    activity: Optional[str] = ""
    start_frame: int
    end_frame: int
    bbox: Optional[List[float]] = None  # [x1,y1,x2,y2] pixels or normalized
    note: Optional[str] = ""


class AnnotationPayload(BaseModel):
    filename: str
    fps: float = 30.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    duration: float = 0.0
    segments: List[Segment] = []


class ExportRequest(BaseModel):
    clear_existing: bool = True
    sync_labels: bool = True


class TrainRequest(BaseModel):
    export_first: bool = True
    sync_labels: bool = True


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return stem or f"video_{uuid.uuid4().hex[:8]}"


def _video_meta(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": 30, "width": 0, "height": 0, "total_frames": 0, "duration": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total / fps if fps > 0 else 0
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total,
        "duration": duration,
    }


@app.get("/api/health")
def health():
    return {"ok": True, "service": "ActionMark"}


@app.get("/api/labels")
def get_labels():
    stats = count_annotations()
    return {
        "labels": config.ACTION_LABELS,
        "postures": config.POSTURE_LABELS,
        "activities": config.ACTIVITY_LABELS,
        "counts": stats["counts"],
        "total": stats["total"],
        "videos_labeled": stats["videos_labeled"],
    }


@app.put("/api/labels")
def set_labels(payload: Dict[str, Any]):
    labels = payload.get("labels") or []
    labels = [l.strip().replace(" ", "_") for l in labels if l.strip()]
    postures = payload.get("postures")
    activities = payload.get("activities")
    if postures is not None:
        postures = [l.strip().replace(" ", "_") for l in postures if str(l).strip()]
    if activities is not None:
        activities = [l.strip().replace(" ", "_") for l in activities if str(l).strip()]

    if not labels and not activities and not postures:
        raise HTTPException(400, "labels cannot be empty")

    if postures is None and activities is None:
        # Flat list from older clients: keep known postures, rest are activities.
        postures = [l for l in labels if l in config.POSTURE_LABELS] or list(config.POSTURE_LABELS)
        activities = [l for l in labels if l not in postures]
    elif activities is None:
        activities = [l for l in labels if l not in (postures or [])]
    elif postures is None:
        postures = list(config.POSTURE_LABELS)

    extras = [l for l in labels if l not in postures and l not in activities]
    if not postures:
        raise HTTPException(400, "Keep at least one posture")
    config.POSTURE_LABELS = postures
    config.ACTIVITY_LABELS = list(dict.fromkeys(activities + extras))
    config.persist_label_taxonomy()
    return {
        "labels": config.ACTION_LABELS,
        "postures": config.POSTURE_LABELS,
        "activities": config.ACTIVITY_LABELS,
    }


@app.delete("/api/labels/{name}")
def delete_label(name: str):
    name = name.strip().replace(" ", "_")
    if not name:
        raise HTTPException(400, "Name is empty")
    postures = [p for p in config.POSTURE_LABELS if p != name]
    activities = [a for a in config.ACTIVITY_LABELS if a != name]
    if name not in config.POSTURE_LABELS and name not in config.ACTIVITY_LABELS:
        raise HTTPException(404, f"Unknown class: {name}")
    if not postures:
        raise HTTPException(400, "Keep at least one posture")
    config.POSTURE_LABELS = postures
    config.ACTIVITY_LABELS = activities
    config.persist_label_taxonomy()
    return {
        "ok": True,
        "labels": config.ACTION_LABELS,
        "postures": config.POSTURE_LABELS,
        "activities": config.ACTIVITY_LABELS,
    }


def _load_persisted_labels():
    config.load_label_taxonomy()


_load_persisted_labels()


_video_index_cache: List[Dict[str, Any]] = []
_video_index_mtime: float = 0.0


def _rebuild_video_index() -> List[Dict[str, Any]]:
    """Build a fast index from annotation JSONs + video directory."""
    global _video_index_cache, _video_index_mtime
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos_on_disk = {}
    for path in sorted(config.VIDEOS_DIR.iterdir()):
        if path.suffix.lower() in video_exts:
            videos_on_disk[path.stem] = path

    index = []
    for stem, path in videos_on_disk.items():
        ann = load_annotation(stem)
        if ann and ann.get("fps"):
            meta = {
                "fps": ann.get("fps", 30),
                "width": ann.get("width", 0),
                "height": ann.get("height", 0),
                "total_frames": ann.get("total_frames", 0),
                "duration": ann.get("duration", 0),
            }
        else:
            meta = _video_meta(path)
            if ann is None:
                ann = {
                    "video_id": stem,
                    "filename": path.name,
                    **meta,
                    "segments": [],
                }
                save_annotation(stem, ann)
        index.append({
            "id": stem,
            "filename": path.name,
            "segments": len(ann.get("segments", [])) if ann else 0,
            **meta,
        })
    _video_index_cache = index
    _video_index_mtime = max(
        (p.stat().st_mtime for p in config.ANNOTATIONS_DIR.glob("*.json")),
        default=0.0,
    )
    return index


def _get_video_index(force: bool = False) -> List[Dict[str, Any]]:
    """Return cached video index, rebuilding if annotations changed."""
    global _video_index_cache, _video_index_mtime
    if not force and _video_index_cache:
        latest = max(
            (p.stat().st_mtime for p in config.ANNOTATIONS_DIR.glob("*.json")),
            default=0.0,
        )
        dir_count = sum(
            1 for p in config.VIDEOS_DIR.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        )
        if latest <= _video_index_mtime and dir_count == len(_video_index_cache):
            return _video_index_cache
    return _rebuild_video_index()


@app.get("/api/videos")
def list_videos(
    q: Optional[str] = None,
    page: int = 1,
    per_page: int = 100,
    labeled: Optional[bool] = None,
):
    index = _get_video_index()
    filtered = index
    if q:
        ql = q.strip().lower()
        filtered = [v for v in filtered if ql in v["filename"].lower() or ql in v["id"].lower()]
    if labeled is True:
        filtered = [v for v in filtered if v["segments"] > 0]
    elif labeled is False:
        filtered = [v for v in filtered if v["segments"] == 0]
    total = len(filtered)
    total_all = len(index)
    total_labeled = sum(1 for v in index if v["segments"] > 0)
    start = (page - 1) * per_page
    end = start + per_page
    page_videos = filtered[start:end]
    return {
        "videos": page_videos,
        "total": total,
        "total_all": total_all,
        "total_labeled": total_labeled,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page else 1,
    }


def _find_library_video(stem: str) -> Optional[Path]:
    """Return an existing library video path for this id, if any."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for path in sorted(config.VIDEOS_DIR.glob(f"{stem}.*")):
        if path.suffix.lower() in video_exts and not path.name.startswith("."):
            return path
    return None


def _ingest_uploaded_video(file: UploadFile) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(400, "No filename")
    ext = Path(file.filename).suffix.lower() or ".mp4"
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported format: {ext}")

    stem = _safe_stem(file.filename)
    existing = _find_library_video(stem)
    if existing:
        ann = load_annotation(stem) or {}
        meta = _video_meta(existing)
        return {
            "ok": True,
            "skipped": True,
            "video": {
                "id": stem,
                "filename": existing.name,
                **meta,
                "segments": len(ann.get("segments", [])),
            },
            "message": f"Skipped {file.filename} — already in library",
        }

    raw_dest = config.VIDEOS_DIR / f"{stem}_upload{ext}"
    while raw_dest.exists():
        stem = f"{stem}_{uuid.uuid4().hex[:6]}"
        raw_dest = config.VIDEOS_DIR / f"{stem}_upload{ext}"

    with open(raw_dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    from .media import transcode_to_h264

    vcodec, _ = probe_codecs(raw_dest)
    final_path = config.VIDEOS_DIR / f"{stem}.mp4"
    converted = False
    try:
        if is_browser_playable(raw_dest) and ext == ".mp4":
            raw_dest.replace(final_path)
        else:
            # HEVC / exotic codecs → H.264 for Chrome/Firefox
            transcode_to_h264(raw_dest, final_path)
            converted = True
            raw_dest.unlink(missing_ok=True)
    except Exception as exc:
        # Fall back to original bytes if conversion fails
        if final_path.exists():
            final_path.unlink(missing_ok=True)
        fallback = config.VIDEOS_DIR / f"{stem}{ext}"
        raw_dest.replace(fallback)
        final_path = fallback
        print(f"[upload] transcode failed, keeping original: {exc}")

    meta = _video_meta(final_path)
    meta["codec"] = probe_codecs(final_path)[0]
    meta["converted_to_h264"] = converted
    if vcodec:
        meta["source_codec"] = vcodec
    ann = {
        "video_id": stem,
        "filename": final_path.name,
        **meta,
        "segments": [],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_annotation(stem, ann)
    return {
        "ok": True,
        "video": {"id": stem, "filename": final_path.name, **meta, "segments": 0},
        "converted_to_h264": converted,
        "message": (
            f"Converted from {vcodec} to H.264 for browser playback"
            if converted and vcodec
            else None
        ),
    }


@app.post("/api/videos/upload")
async def upload_video(file: List[UploadFile] = File(...)):
    uploads = [f for f in file if f and f.filename]
    if not uploads:
        raise HTTPException(400, "No filename")

    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for item in uploads:
        try:
            result = _ingest_uploaded_video(item)
            if result.get("skipped"):
                skipped.append(result)
            else:
                results.append(result)
        except HTTPException as exc:
            errors.append({"filename": item.filename or "", "error": str(exc.detail)})
        except Exception as exc:
            errors.append({"filename": item.filename or "", "error": str(exc)})

    if not results and not skipped:
        raise HTTPException(400, errors[0]["error"] if errors else "Upload failed")
    if len(uploads) == 1 and not errors:
        out = results[0] if results else skipped[0]
        return out
    last_video = (results or skipped)[-1].get("video")
    return {
        "ok": True,
        "video": last_video,
        "videos": [r["video"] for r in results],
        "uploaded": len(results),
        "skipped": len(skipped),
        "skipped_videos": [r["video"] for r in skipped],
        "failed": len(errors),
        "errors": errors,
        "converted_to_h264": any(r.get("converted_to_h264") for r in results),
        "message": (
            f"Uploaded {len(results)} video{'s' if len(results) != 1 else ''}"
            + (f", skipped {len(skipped)}" if skipped else "")
            + (f", {len(errors)} failed" if errors else "")
        ),
    }


@app.post("/api/videos/{video_id}/transcode")
def transcode_video(video_id: str):
    """Convert an existing library video to H.264 for browser playback."""
    matches = sorted(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(404, "Video not found")
    src = matches[0]
    if is_browser_playable(src):
        return {"ok": True, "already_playable": True, "filename": src.name}
    try:
        from .media import transcode_to_h264

        out = config.VIDEOS_DIR / f"{video_id}.mp4"
        tmp = config.VIDEOS_DIR / f".{video_id}_tmp.mp4"
        transcode_to_h264(src, tmp)
        if src.resolve() != out.resolve():
            src.unlink(missing_ok=True)
        tmp.replace(out)
        meta = _video_meta(out)
        ann = load_annotation(video_id) or {}
        ann.update(
            {
                "filename": out.name,
                **meta,
                "codec": probe_codecs(out)[0],
                "converted_to_h264": True,
            }
        )
        save_annotation(video_id, ann)
        return {"ok": True, "video": {"id": video_id, "filename": out.name, **meta}}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    matches = list(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(404, "Video not found")
    for m in matches:
        m.unlink()
    ann = config.ANNOTATIONS_DIR / f"{video_id}.json"
    if ann.exists():
        ann.unlink()
    _get_video_index(force=True)
    return {"ok": True}


@app.get("/api/videos/{video_id}/file")
def get_video_file(video_id: str):
    matches = list(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(404, "Video not found")
    path = matches[0]
    media = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/videos/{video_id}/meta")
def get_video_meta(video_id: str):
    matches = list(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(404, "Video not found")
    meta = _video_meta(matches[0])
    ann = load_annotation(video_id) or {}
    return {
        "id": video_id,
        "filename": matches[0].name,
        **meta,
        "segments": ann.get("segments", []),
    }


@app.get("/api/annotations/{video_id}")
def get_annotation(video_id: str):
    ann = load_annotation(video_id)
    if not ann:
        raise HTTPException(404, "Annotation not found")
    return ann


@app.put("/api/annotations/{video_id}")
def put_annotation(video_id: str, payload: AnnotationPayload):
    matches = list(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(404, "Video not found")
    for seg in payload.segments:
        names = config.segment_class_names(seg.model_dump())
        if not names:
            raise HTTPException(400, "Each segment needs a posture and/or activity")
        if seg.end_frame < seg.start_frame:
            raise HTTPException(400, "end_frame must be >= start_frame")
        # Keep a display label for older tools
        if not seg.label:
            seg.label = " + ".join(names)
    data = payload.model_dump()
    data["video_id"] = video_id
    path = save_annotation(video_id, data)
    _get_video_index(force=True)
    return {"ok": True, "path": str(path), "segments": len(payload.segments)}


@app.post("/api/export")
def api_export(req: ExportRequest):
    if req.sync_labels:
        sync_train_config_labels()
    summary = export_dataset(clear_existing=req.clear_existing)
    if not summary.get("ok"):
        raise HTTPException(400, summary.get("error", "Export failed"))
    return summary


@app.post("/api/train")
def api_train(req: TrainRequest):
    result = start_training(export_first=req.export_first, sync_labels=req.sync_labels)
    if not result.get("ok"):
        raise HTTPException(409, result.get("error", "Cannot start training"))
    return result


@app.get("/api/train/status")
def train_status(job_id: Optional[str] = None):
    if job_id:
        job = get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return {"job": job}
    active = get_active_job()
    return {"job": active, "jobs": list_jobs()[:10]}


@app.post("/api/train/stop/{job_id}")
def train_stop(job_id: str):
    result = stop_training(job_id)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Stop failed"))
    return result


@app.get("/api/train/log/{job_id}")
def train_log(job_id: str, tail: int = 200):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    log_path = Path(job.get("log_path", ""))
    if not log_path.exists():
        return {"lines": []}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-tail:]}


# ---------- Test / inference ----------
class TestRequest(BaseModel):
    checkpoint: str  # filename under work_dirs/slowfast_multilabel
    video_id: Optional[str] = None  # use library video
    # or upload via multipart separately


@app.get("/api/models")
def api_models():
    return {"models": list_checkpoints()}


@app.post("/api/test/upload")
async def test_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    ext = Path(file.filename).suffix.lower() or ".mp4"
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported format: {ext}")
    stem = _safe_stem(file.filename)
    dest = TEST_INPUT_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    # Convert to H.264 if needed for consistency
    try:
        from .media import is_browser_playable, transcode_to_h264

        if not is_browser_playable(dest):
            h264 = dest.with_suffix(".mp4")
            tmp = dest.with_name(f".{dest.stem}_tmp.mp4")
            transcode_to_h264(dest, tmp)
            if dest.resolve() != h264.resolve():
                dest.unlink(missing_ok=True)
            tmp.replace(h264)
            dest = h264
    except Exception as exc:
        print(f"[test upload] transcode skipped: {exc}")
    return {"ok": True, "path": str(dest), "filename": dest.name, "id": dest.stem}


@app.post("/api/test/run")
async def test_run(
    checkpoint: str = Form(...),
    video_id: Optional[str] = Form(None),
    test_video: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Start inference.
    Provide one of: library video_id, previously uploaded test_video filename, or file upload.
    checkpoint: e.g. best_acc_top1_epoch_5.pth
    """
    ckpt = config.WORK_DIR / checkpoint
    if not ckpt.exists():
        raise HTTPException(404, f"Checkpoint not found: {checkpoint}")

    video_path: Optional[Path] = None
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower() or ".mp4"
        stem = _safe_stem(file.filename)
        video_path = TEST_INPUT_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
        with open(video_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    elif test_video:
        candidate = TEST_INPUT_DIR / test_video
        if not candidate.exists():
            raise HTTPException(404, "Test video not found")
        video_path = candidate
    elif video_id:
        matches = list(config.VIDEOS_DIR.glob(f"{video_id}.*"))
        if not matches:
            raise HTTPException(404, "Library video not found")
        video_path = matches[0]
    else:
        raise HTTPException(400, "Provide file, test_video, or video_id")

    result = start_inference_job(video_path, ckpt)
    return result


@app.post("/api/test/live")
async def test_live(
    checkpoint: str = Form(...),
    frames: List[UploadFile] = File(...),
):
    """Classify a short webcam clip (JPEG frames) with person boxes + posture/activity."""
    ckpt = config.WORK_DIR / checkpoint
    if not ckpt.exists():
        raise HTTPException(404, f"Checkpoint not found: {checkpoint}")
    decoded: List[Any] = []
    for item in frames:
        raw = await item.read()
        if not raw:
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            decoded.append(img)
    if len(decoded) < 5:
        raise HTTPException(400, "Need at least 5 camera frames")
    try:
        return run_live_clip(decoded, ckpt)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/test/status")
def test_status(job_id: Optional[str] = None):
    if job_id:
        job = get_test_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return {"job": job}
    return {"jobs": list_test_jobs()}


@app.get("/api/test/result/{job_id}/video")
def test_result_video(job_id: str):
    job = get_test_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = Path(job.get("output_video") or TEST_OUTPUT_DIR / f"{job_id}.mp4")
    if not path.exists():
        raise HTTPException(404, "Result video not ready")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    path = FRONTEND_DIR / "favicon.svg"
    if not path.exists():
        raise HTTPException(404, "favicon not found")
    return FileResponse(path, media_type="image/svg+xml")


# Serve frontend last
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
