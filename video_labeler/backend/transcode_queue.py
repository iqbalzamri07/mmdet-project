"""Background H.264 transcode queue — keeps uploads fast during bulk ingest."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from . import config
from .collab import bump_library_revision
from .export import load_annotation, save_annotation
from .media import is_browser_playable, probe_codecs, probe_video_meta, transcode_to_h264

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="am-transcode")
_state_lock = threading.Lock()
_pending: Set[str] = set()


def is_transcode_pending(video_id: str) -> bool:
    with _state_lock:
        return video_id in _pending


def pending_count() -> int:
    with _state_lock:
        return len(_pending)


def _video_path(video_id: str) -> Optional[Path]:
    matches = sorted(config.VIDEOS_DIR.glob(f"{video_id}.*"))
    if not matches:
        return None
    return matches[0]


def get_transcode_status() -> Dict[str, Any]:
    """Summarize stuck/failed/pending conversion work (fast — uses annotation status)."""
    stuck = 0
    failed = 0
    transcoding = 0
    ready = 0
    for path in config.ANNOTATIONS_DIR.glob("*.json"):
        if path.name.startswith("."):
            continue
        video_id = path.stem
        if not _video_path(video_id):
            continue
        ann = load_annotation(video_id) or {}
        status = ann.get("processing_status") or "ready"
        if status == "failed":
            failed += 1
        elif status == "transcoding":
            transcoding += 1
            if not is_transcode_pending(video_id):
                stuck += 1
        else:
            ready += 1
    needs_convert = failed + transcoding
    return {
        "pending": pending_count(),
        "stuck": stuck,
        "failed": failed,
        "transcoding": transcoding,
        "needs_convert": needs_convert,
        "ready_playable": ready,
    }


def enqueue_video_transcode(
    video_id: str,
    src: Path,
    dest: Path,
    source_codec: Optional[str] = None,
) -> bool:
    """Queue a transcode job. Returns False if this video is already queued."""
    with _state_lock:
        if video_id in _pending:
            return False
        _pending.add(video_id)

    def _run() -> None:
        try:
            transcode_to_h264(src, dest)
            ann = load_annotation(video_id) or {}
            meta_path = dest if dest.exists() else src
            meta = probe_video_meta(meta_path)
            vcodec, _ = probe_codecs(meta_path)
            ann.update(
                {
                    "filename": meta_path.name,
                    "fps": meta.get("fps", 30),
                    "width": meta.get("width", 0),
                    "height": meta.get("height", 0),
                    "total_frames": meta.get("total_frames", 0),
                    "duration": meta.get("duration", 0),
                    "codec": vcodec,
                    "converted_to_h264": True,
                    "processing_status": "ready",
                    "processing_finished_at": datetime.utcnow().isoformat() + "Z",
                }
            )
            ann.pop("processing_error", None)
            if source_codec:
                ann["source_codec"] = source_codec
            save_annotation(video_id, ann)
            if src.resolve() != dest.resolve() and src.exists():
                src.unlink(missing_ok=True)
            bump_library_revision()
        except Exception as exc:
            ann = load_annotation(video_id) or {}
            ann["processing_status"] = "failed"
            ann["processing_error"] = str(exc)
            ann["processing_finished_at"] = datetime.utcnow().isoformat() + "Z"
            save_annotation(video_id, ann)
            bump_library_revision()
            print(f"[transcode queue] failed {video_id}: {exc}")
        finally:
            with _state_lock:
                _pending.discard(video_id)

    _executor.submit(_run)
    return True


def requeue_stuck_transcodes(
    include_failed: bool = True,
    include_stuck: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Re-queue HEVC / non-playable videos stuck in transcoding or marked failed.
    Returns counts and sample ids.
    """
    queued: List[str] = []
    skipped: List[str] = []
    fixed_ready: List[str] = []

    for path in sorted(config.ANNOTATIONS_DIR.glob("*.json")):
        if path.name.startswith("."):
            continue
        video_id = path.stem
        src = _video_path(video_id)
        if not src:
            continue

        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if '"failed"' not in raw and '"transcoding"' not in raw:
            continue

        ann = load_annotation(video_id) or {}
        status = ann.get("processing_status") or "ready"

        if status == "ready":
            continue

        should_queue = False
        if include_stuck and status == "transcoding" and not is_transcode_pending(video_id):
            should_queue = True
        elif include_failed and status == "failed":
            should_queue = True

        if not should_queue:
            skipped.append(video_id)
            continue
        if is_transcode_pending(video_id):
            skipped.append(video_id)
            continue

        if is_browser_playable(src):
            ann["processing_status"] = "ready"
            ann.pop("processing_error", None)
            ann["filename"] = src.name
            meta = probe_video_meta(src)
            ann.update(
                {
                    "fps": meta.get("fps", 30),
                    "width": meta.get("width", 0),
                    "height": meta.get("height", 0),
                    "total_frames": meta.get("total_frames", 0),
                    "duration": meta.get("duration", 0),
                }
            )
            save_annotation(video_id, ann)
            fixed_ready.append(video_id)
            continue

        dest = config.VIDEOS_DIR / f"{video_id}.mp4"
        vcodec, _ = probe_codecs(src)
        ann["processing_status"] = "transcoding"
        ann["processing_started_at"] = datetime.utcnow().isoformat() + "Z"
        ann.pop("processing_error", None)
        save_annotation(video_id, ann)
        if enqueue_video_transcode(video_id, src, dest, vcodec or ""):
            queued.append(video_id)
        else:
            skipped.append(video_id)

        if limit and len(queued) >= limit:
            break

    if queued or fixed_ready:
        bump_library_revision()

    return {
        "ok": True,
        "queued": len(queued),
        "skipped": len(skipped),
        "fixed_ready": len(fixed_ready),
        "pending": pending_count(),
        "queued_ids": queued[:30],
        "message": (
            f"Queued {len(queued)} video(s) for H.264 conversion"
            if queued
            else "No videos needed re-queuing"
        ),
    }


def resume_stuck_on_startup() -> None:
    """Re-queue transcoding jobs that were interrupted (e.g. disk full / server restart)."""
    result = requeue_stuck_transcodes(include_failed=False, include_stuck=True, limit=50)
    if result.get("queued"):
        print(
            f"[transcode queue] resumed {result['queued']} stuck conversion(s) on startup "
            f"({result.get('pending', 0)} in queue; use Retry button for more)"
        )
