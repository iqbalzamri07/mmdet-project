"""Browser-friendly video helpers (HEVC/etc → H.264)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


BROWSER_SAFE_VIDEO = {"h264", "avc1", "vp8", "vp9", "av1"}
BROWSER_SAFE_AUDIO = {"aac", "mp3", "opus", "vorbis", "none", ""}


def _ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return json.loads(out)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return {}


def probe_codecs(path: Path) -> Tuple[Optional[str], Optional[str]]:
    data = _ffprobe(path)
    vcodec = None
    acodec = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not vcodec:
            vcodec = (stream.get("codec_name") or "").lower()
        if stream.get("codec_type") == "audio" and not acodec:
            acodec = (stream.get("codec_name") or "").lower()
    return vcodec, acodec


def is_browser_playable(path: Path) -> bool:
    vcodec, acodec = probe_codecs(path)
    if not vcodec:
        return False
    if vcodec not in BROWSER_SAFE_VIDEO:
        return False
    if acodec and acodec not in BROWSER_SAFE_AUDIO:
        # Still playable often if video is h264; keep strict-ish for a/v sync
        return vcodec in BROWSER_SAFE_VIDEO
    return True


def transcode_to_h264(src: Path, dest: Optional[Path] = None) -> Path:
    """
    Transcode to H.264 + AAC MP4 with faststart for HTML5 playback.
    Returns path to the playable file.
    """
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
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:] if proc.stderr else 'unknown error'}")


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
