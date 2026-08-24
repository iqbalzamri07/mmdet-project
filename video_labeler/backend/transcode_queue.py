"""Background H.264 transcode queue — keeps uploads fast during bulk ingest."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

from .collab import bump_library_revision
from .export import load_annotation, save_annotation
from .media import probe_codecs, transcode_to_h264

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="am-transcode")
_state_lock = threading.Lock()
_pending: Set[str] = set()


def is_transcode_pending(video_id: str) -> bool:
    with _state_lock:
        return video_id in _pending


def pending_count() -> int:
    with _state_lock:
        return len(_pending)


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
            meta_path = dest
            import cv2

            cap = cv2.VideoCapture(str(meta_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = total / fps if fps > 0 else 0

            vcodec, _ = probe_codecs(dest)
            ann.update(
                {
                    "filename": dest.name,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "total_frames": total,
                    "duration": duration,
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
