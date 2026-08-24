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
from .transcode_queue import enqueue_video_transcode, is_transcode_pending
from .infer import (
    TEST_INPUT_DIR,
    TEST_OUTPUT_DIR,
    delete_test_input,
    delete_test_result,
    get_test_job,
    list_checkpoints,
    list_test_inputs,
    list_test_jobs,
    list_test_library,
    run_live_clip,
    start_inference_job,
)
from .onnx_export import (
    get_active_onnx_export,
    get_onnx_export_job,
    resolve_work_file,
    start_onnx_export,
)
from .collab import (
    acquire_lock,
    bump_library_revision,
    collab_status,
    get_library_revision,
    heartbeat,
    new_client_id,
    release_all_for_client,
    release_lock,
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
    annotator: Optional[str] = ""


class ExportRequest(BaseModel):
    clear_existing: bool = True
    sync_labels: bool = True


class TrainRequest(BaseModel):
    export_first: bool = True
    sync_labels: bool = True
    epochs: int = Field(default=100, ge=1, le=300)


class BulkDeleteRequest(BaseModel):
    ids: List[str] = Field(default_factory=list, min_length=1)


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


def _parse_iso_ts(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _video_sort_ts(ann: Optional[Dict[str, Any]], path: Path) -> float:
    ts = _parse_iso_ts((ann or {}).get("updated_at") or "")
    if not ts:
        ts = _parse_iso_ts((ann or {}).get("created_at") or "")
    if not ts:
        ts = path.stat().st_mtime
    return ts


def _video_sort_key(video: Dict[str, Any]) -> tuple:
    return (video.get("sort_ts") or 0.0, video.get("filename") or video.get("id") or "")


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
            "last_annotator": (ann or {}).get("last_annotator") or "",
            "updated_at": (ann or {}).get("updated_at") or "",
            "sort_ts": _video_sort_ts(ann, path),
            "processing_status": (ann or {}).get("processing_status") or "ready",
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
    filtered.sort(key=_video_sort_key, reverse=True)
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

    vcodec, _ = probe_codecs(raw_dest)
    final_path = config.VIDEOS_DIR / f"{stem}.mp4"
    converted = False
    processing = False
    try:
        if is_browser_playable(raw_dest) and ext == ".mp4":
            raw_dest.replace(final_path)
        elif is_browser_playable(raw_dest):
            fallback = config.VIDEOS_DIR / f"{stem}{ext}"
            raw_dest.replace(fallback)
            final_path = fallback
        else:
            # Defer heavy ffmpeg work — upload returns immediately
            staging = config.VIDEOS_DIR / f"{stem}{ext}"
            raw_dest.replace(staging)
            final_path = staging
            processing = True
    except Exception as exc:
        if final_path.exists():
            final_path.unlink(missing_ok=True)
        fallback = config.VIDEOS_DIR / f"{stem}{ext}"
        raw_dest.replace(fallback)
        final_path = fallback
        print(f"[upload] ingest failed, keeping original: {exc}")

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
    if processing:
        ann["processing_status"] = "transcoding"
        ann["processing_started_at"] = datetime.utcnow().isoformat() + "Z"
    save_annotation(stem, ann)
    if processing:
        enqueue_video_transcode(stem, final_path, config.VIDEOS_DIR / f"{stem}.mp4", vcodec or "")
    return {
        "ok": True,
        "video": {
            "id": stem,
            "filename": final_path.name,
            **meta,
            "segments": 0,
            "processing_status": ann.get("processing_status", "ready"),
        },
        "converted_to_h264": converted,
        "processing": processing,
        "message": (
            "Uploaded — converting to H.264 in background (you can keep labeling other videos)"
            if processing
            else (
                f"Converted from {vcodec} to H.264 for browser playback"
                if converted and vcodec
                else None
            )
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
    if results:
        _get_video_index(force=True)
        bump_library_revision()
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
    if is_transcode_pending(video_id):
        raise HTTPException(409, "Video is already converting in the background")
    matches = sorted(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(404, "Video not found")
    src = matches[0]
    ann = load_annotation(video_id) or {}
    if ann.get("processing_status") == "transcoding":
        raise HTTPException(409, "Video is already converting in the background")
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
                "processing_status": "ready",
            }
        )
        ann.pop("processing_error", None)
        save_annotation(video_id, ann)
        bump_library_revision()
        return {"ok": True, "video": {"id": video_id, "filename": out.name, **meta}}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


def _delete_video_files(video_id: str) -> bool:
    """Delete video file(s), annotation, and lock. Returns True if a video existed."""
    matches = list(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        return False
    for m in matches:
        m.unlink(missing_ok=True)
    ann = config.ANNOTATIONS_DIR / f"{video_id}.json"
    if ann.exists():
        ann.unlink(missing_ok=True)
    for lock in list(collab_status()["locks"]):
        if lock["video_id"] == video_id:
            release_lock(video_id, lock["client_id"])
            break
    return True


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    if not _delete_video_files(video_id):
        raise HTTPException(404, "Video not found")
    _get_video_index(force=True)
    bump_library_revision()
    return {"ok": True}


@app.post("/api/videos/bulk-delete")
def bulk_delete_videos(payload: BulkDeleteRequest):
    deleted: List[str] = []
    missing: List[str] = []
    seen = set()
    for video_id in payload.ids:
        vid = (video_id or "").strip()
        if not vid or vid in seen:
            continue
        seen.add(vid)
        if _delete_video_files(vid):
            deleted.append(vid)
        else:
            missing.append(vid)
    if deleted:
        _get_video_index(force=True)
        bump_library_revision()
    return {
        "ok": True,
        "deleted": deleted,
        "missing": missing,
        "count": len(deleted),
    }


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
        "last_annotator": ann.get("last_annotator") or "",
        "updated_at": ann.get("updated_at") or "",
        "annotation_log": ann.get("annotation_log") or [],
        "processing_status": ann.get("processing_status") or "ready",
        "processing_error": ann.get("processing_error") or "",
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
    data = payload.model_dump(exclude={"annotator"})
    annotator = (payload.annotator or "").strip() or "Annotator"
    existing = load_annotation(video_id) or {}
    now = datetime.utcnow().isoformat() + "Z"
    log = list(existing.get("annotation_log") or [])
    log.append(
        {
            "at": now,
            "annotator": annotator,
            "segments": len(payload.segments),
        }
    )
    data["last_annotator"] = annotator
    data["annotation_log"] = log[-50:]
    if existing.get("created_at"):
        data["created_at"] = existing["created_at"]
    elif not data.get("created_at"):
        data["created_at"] = now
    path = save_annotation(video_id, data)
    _get_video_index(force=True)
    rev = bump_library_revision()
    return {
        "ok": True,
        "path": str(path),
        "segments": len(payload.segments),
        "revision": rev,
    }


# ---------- Collaboration (locks + library refresh) ----------
class CollabHelloRequest(BaseModel):
    name: Optional[str] = ""


class CollabLockRequest(BaseModel):
    client_id: str
    name: Optional[str] = ""


class CollabHeartbeatRequest(BaseModel):
    client_id: str
    video_id: Optional[str] = None
    name: Optional[str] = ""


@app.post("/api/collab/hello")
def collab_hello(req: CollabHelloRequest = CollabHelloRequest()):
    return {
        "ok": True,
        "client_id": new_client_id(),
        "name": (req.name or "").strip() or "Annotator",
        "revision": get_library_revision(),
        "ttl_sec": 45,
    }


def _enrich_locks(locks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach filename to lock entries for the Currently editing panel."""
    out = []
    for lock in locks or []:
        item = dict(lock)
        vid = item.get("video_id") or ""
        matches = list(config.VIDEOS_DIR.glob(f"{vid}.*")) if vid else []
        item["filename"] = matches[0].name if matches else vid
        out.append(item)
    return out


@app.get("/api/collab/status")
def api_collab_status(since: int = 0):
    data = collab_status(since=since)
    data["locks"] = _enrich_locks(data.get("locks") or [])
    return data


@app.post("/api/collab/lock/{video_id}")
def api_collab_lock(video_id: str, req: CollabLockRequest):
    result = acquire_lock(video_id, req.client_id, req.name or "")
    if not result.get("ok"):
        raise HTTPException(409, result.get("error", "Locked"))
    if result.get("lock"):
        result["lock"]["filename"] = _enrich_locks([result["lock"]])[0].get("filename")
    return result


@app.delete("/api/collab/lock/{video_id}")
def api_collab_unlock(video_id: str, client_id: str):
    result = release_lock(video_id, client_id)
    if not result.get("ok"):
        raise HTTPException(403, result.get("error", "Cannot release lock"))
    return result


@app.post("/api/collab/heartbeat")
def api_collab_heartbeat(req: CollabHeartbeatRequest):
    data = heartbeat(req.client_id, req.video_id, req.name or "")
    data["locks"] = _enrich_locks(data.get("locks") or [])
    return data


@app.post("/api/collab/bye")
def api_collab_bye(req: CollabLockRequest):
    n = release_all_for_client(req.client_id)
    return {"ok": True, "released": n}


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
    result = start_training(
        export_first=req.export_first,
        sync_labels=req.sync_labels,
        epochs=req.epochs,
    )
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
    test_video: Optional[str] = None  # filename under data/tests/inputs


@app.get("/api/models")
def api_models():
    return {"models": list_checkpoints()}


class OnnxExportRequest(BaseModel):
    checkpoint: str  # .pth filename under work_dirs


@app.post("/api/models/export-onnx")
def api_export_onnx(req: OnnxExportRequest):
    try:
        result = start_onnx_export(req.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not result.get("ok"):
        raise HTTPException(409, result.get("error", "Cannot start ONNX export"))
    return result


@app.get("/api/models/export-onnx/status")
def api_export_onnx_status(job_id: Optional[str] = None):
    if job_id:
        job = get_onnx_export_job(job_id)
        if not job:
            raise HTTPException(404, "Export job not found")
    else:
        job = get_active_onnx_export()
    if not job:
        return {"job": None}
    log_path = Path(job.get("log_path") or "")
    lines = []
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    out = dict(job)
    out["log"] = lines
    return {"job": out}


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
    test_video: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Start inference on a Test video only (not Label library).
    Provide one of: previously uploaded test_video filename, or file upload.
    checkpoint: e.g. best_acc_top1_epoch_5.pth or epoch_100.onnx
    """
    try:
        ckpt = resolve_work_file(checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc

    video_path: Optional[Path] = None
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower() or ".mp4"
        stem = _safe_stem(file.filename)
        video_path = TEST_INPUT_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
        with open(video_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    elif test_video:
        # Only allow files under TEST_INPUT_DIR (never Label VIDEOS_DIR)
        name = Path(test_video).name
        if name != test_video or ".." in test_video or "/" in test_video or "\\" in test_video:
            raise HTTPException(400, "Invalid test video name")
        candidate = TEST_INPUT_DIR / name
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(404, "Test video not found")
        video_path = candidate
    else:
        raise HTTPException(400, "Provide file or test_video")

    result = start_inference_job(video_path, ckpt)
    return result


@app.get("/api/test/inputs")
def test_inputs(
    page: int = 1,
    per_page: int = 50,
    q: Optional[str] = None,
):
    """List videos uploaded for Test (separate from Label library)."""
    return list_test_inputs(page=page, per_page=per_page, q=q or "")


@app.delete("/api/test/inputs/{filename}")
def test_input_delete(filename: str):
    result = delete_test_input(filename)
    if not result.get("ok"):
        raise HTTPException(404 if "not found" in (result.get("error") or "").lower() else 400, result.get("error"))
    return result


@app.delete("/api/test/result/{job_id}")
def test_result_delete(job_id: str):
    result = delete_test_result(job_id)
    if not result.get("ok"):
        raise HTTPException(404 if "not found" in (result.get("error") or "").lower() else 400, result.get("error"))
    return result


@app.post("/api/test/live")
async def test_live(
    checkpoint: str = Form(...),
    frames: List[UploadFile] = File(...),
):
    """Classify a short webcam clip (JPEG frames) with person boxes + posture/activity."""
    try:
        ckpt = resolve_work_file(checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
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


@app.get("/api/test/library")
def test_library(
    page: int = 1,
    per_page: int = 50,
    q: Optional[str] = None,
):
    return list_test_library(page=page, per_page=per_page, q=q or "")


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
