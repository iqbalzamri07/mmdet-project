"""Browser-friendly video helpers (HEVC/etc → H.264)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


BROWSER_SAFE_VIDEO = {"h264", "avc1", "vp8", "vp9", "av1"}
BROWSER_SAFE_AUDIO = {"aac", "mp3", "opus", "vorbis", "none", ""}


def _ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            return {"_error": (proc.stderr or proc.stdout or "ffprobe failed").strip()}
        data = json.loads(proc.stdout)
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return {"_error": str(exc)}


def probe_codecs(path: Path) -> Tuple[Optional[str], Optional[str]]:
    data = _ffprobe(path)
    if data.get("_error"):
        return None, None
    vcodec = None
    acodec = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not vcodec:
            vcodec = (stream.get("codec_name") or "").lower()
        if stream.get("codec_type") == "audio" and not acodec:
            acodec = (stream.get("codec_name") or "").lower()
    return vcodec, acodec


def validate_video_file(path: Path) -> Dict[str, Any]:
    """
    Check whether a video file is readable (has streams / moov atom).
    Returns {ok, error?, vcodec?, duration?, width?, height?}.
    """
    if not path.exists():
        return {"ok": False, "error": "File not found"}
    if path.stat().st_size < 64:
        return {"ok": False, "error": "File is empty or truncated"}

    data = _ffprobe(path)
    err = (data.get("_error") or "").strip()
    if err:
        low = err.lower()
        if "moov atom not found" in low:
            return {
                "ok": False,
                "error": (
                    "Incomplete or corrupt MP4 (moov atom missing). "
                    "Re-export/re-download the video and upload again."
                ),
                "code": "moov_missing",
            }
        return {"ok": False, "error": f"Unreadable video: {err[:300]}", "code": "unreadable"}

    streams = data.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        return {"ok": False, "error": "No video stream found in file", "code": "no_video"}

    vs = video_streams[0]
    fmt = data.get("format") or {}
    try:
        duration = float(fmt.get("duration") or vs.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        width = int(vs.get("width") or 0)
        height = int(vs.get("height") or 0)
    except (TypeError, ValueError):
        width, height = 0, 0

    return {
        "ok": True,
        "vcodec": (vs.get("codec_name") or "").lower() or None,
        "duration": duration,
        "width": width,
        "height": height,
    }


def try_repair_mp4(src: Path) -> Optional[Path]:
    """
    Attempt a remux repair (copy streams + faststart).
    Helps some truncated/odd MP4s; cannot invent a missing moov from nothing.
    Returns path to repaired file, or None.
    """
    if not shutil.which("ffmpeg"):
        return None
    check = validate_video_file(src)
    if check.get("ok"):
        return src

    tmp = src.with_name(f".{src.stem}_repair.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-err_detect",
        "ignore_err",
        "-i",
        str(src),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return None
    if not validate_video_file(tmp).get("ok"):
        tmp.unlink(missing_ok=True)
        return None
    fixed = src.with_name(f"{src.stem}_fixed.mp4")
    tmp.replace(fixed)
    return fixed


def is_browser_playable(path: Path) -> bool:
    check = validate_video_file(path)
    if not check.get("ok"):
        return False
    vcodec, acodec = probe_codecs(path)
    if not vcodec:
        return False
    if vcodec not in BROWSER_SAFE_VIDEO:
        return False
    if acodec and acodec not in BROWSER_SAFE_AUDIO:
        # Still playable often if video is h264; keep strict-ish for a/v sync
        return vcodec in BROWSER_SAFE_VIDEO
    return True


def probe_video_meta(path: Path) -> Dict[str, Any]:
    """Read fps/size/duration via ffprobe (no OpenCV — avoids moov spam)."""
    data = _ffprobe(path)
    if data.get("_error"):
        return {"fps": 30, "width": 0, "height": 0, "total_frames": 0, "duration": 0, "readable": False}
    streams = data.get("streams") or []
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not vs:
        return {"fps": 30, "width": 0, "height": 0, "total_frames": 0, "duration": 0, "readable": False}
    fmt = data.get("format") or {}
    fps = 30.0
    rate = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "30/1"
    try:
        if isinstance(rate, str) and "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den) or 1.0
            fps = float(num) / den_f
        else:
            fps = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 30.0
    if fps <= 0 or fps > 240:
        fps = 30.0
    try:
        duration = float(fmt.get("duration") or vs.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        width = int(vs.get("width") or 0)
        height = int(vs.get("height") or 0)
    except (TypeError, ValueError):
        width, height = 0, 0
    nb = vs.get("nb_frames")
    try:
        total = int(nb) if nb not in (None, "N/A", "") else int(round(duration * fps))
    except (TypeError, ValueError):
        total = int(round(duration * fps)) if duration > 0 else 0
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total,
        "duration": duration,
        "readable": True,
    }


def transcode_to_h264(src: Path, dest: Optional[Path] = None) -> Path:
    """
    Transcode to H.264 + AAC MP4 with faststart for HTML5 playback.
    Returns path to the playable file.
    """
    check = validate_video_file(src)
    if not check.get("ok"):
        raise RuntimeError(check.get("error") or "Source video is unreadable")

    if dest is None:
        dest = src.with_name(f"{src.stem}_h264.mp4")

    if dest.resolve() == src.resolve():
        tmp = src.with_name(f".{src.stem}_transcoding.mp4")
        _run_ffmpeg(src, tmp)
        tmp.replace(src)
        return src

    _run_ffmpeg(src, dest)
    return dest


def _run_ffmpeg(src: Path, dest: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found — install ffmpeg to convert videos for the browser")

    dest.parent.mkdir(parents=True, exist_ok=True)
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
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        err = proc.stderr or ""
        if "moov atom not found" in err.lower():
            raise RuntimeError(
                "Incomplete or corrupt MP4 (moov atom missing). "
                "Re-export/re-download the video and upload again."
            )
        raise RuntimeError(f"ffmpeg failed: {err[-800:] if err else 'unknown error'}")


def ensure_browser_playable(path: Path) -> Path:
    """If needed, replace/create an H.264 sibling and return the playable path."""
    if is_browser_playable(path):
        return path

    playable = path.with_suffix(".mp4")
    if playable == path:
        # overwrite via temp
        return transcode_to_h264(path, path)

    out = path.with_name(f"{path.stem}_browser.mp4")
    return transcode_to_h264(path, out)
