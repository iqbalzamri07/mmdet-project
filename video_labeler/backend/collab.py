"""In-memory collaboration: video edit locks + library revision for refresh."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional

_lock = threading.Lock()

# video_id -> lock info
_locks: Dict[str, Dict[str, Any]] = {}

# Monotonic revision; bumps when library/annotations change
_library_revision: int = 1

LOCK_TTL_SEC = 45  # must heartbeat before this expires


def bump_library_revision() -> int:
    global _library_revision
    with _lock:
        _library_revision += 1
        return _library_revision


def get_library_revision() -> int:
    with _lock:
        return _library_revision


def _purge_expired(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    expired = [vid for vid, info in _locks.items() if float(info.get("expires_at", 0)) <= now]
    for vid in expired:
        del _locks[vid]


def list_locks() -> List[Dict[str, Any]]:
    with _lock:
        _purge_expired()
        return [
            {
                "video_id": vid,
                "client_id": info["client_id"],
                "name": info.get("name") or "Someone",
                "expires_at": info["expires_at"],
            }
            for vid, info in _locks.items()
        ]


def get_lock(video_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        _purge_expired()
        info = _locks.get(video_id)
        if not info:
            return None
        return {
            "video_id": video_id,
            "client_id": info["client_id"],
            "name": info.get("name") or "Someone",
            "expires_at": info["expires_at"],
        }


def acquire_lock(video_id: str, client_id: str, name: str = "") -> Dict[str, Any]:
    """Acquire or renew a lock. Fails if held by another client."""
    if not video_id or not client_id:
        return {"ok": False, "error": "video_id and client_id required"}
    now = time.time()
    with _lock:
        _purge_expired(now)
        current = _locks.get(video_id)
        if current and current["client_id"] != client_id:
            return {
                "ok": False,
                "error": f"Locked by {current.get('name') or 'another user'}",
                "lock": {
                    "video_id": video_id,
                    "client_id": current["client_id"],
                    "name": current.get("name") or "Someone",
                    "expires_at": current["expires_at"],
                },
            }
        expires = now + LOCK_TTL_SEC
        _locks[video_id] = {
            "client_id": client_id,
            "name": (name or "").strip() or "Annotator",
            "expires_at": expires,
            "updated_at": now,
        }
        return {
            "ok": True,
            "lock": {
                "video_id": video_id,
                "client_id": client_id,
                "name": _locks[video_id]["name"],
                "expires_at": expires,
            },
        }


def release_lock(video_id: str, client_id: str) -> Dict[str, Any]:
    with _lock:
        _purge_expired()
        current = _locks.get(video_id)
        if not current:
            return {"ok": True, "released": False}
        if current["client_id"] != client_id:
            return {"ok": False, "error": "Not your lock"}
        del _locks[video_id]
        return {"ok": True, "released": True}


def release_all_for_client(client_id: str) -> int:
    with _lock:
        vids = [vid for vid, info in _locks.items() if info.get("client_id") == client_id]
        for vid in vids:
            del _locks[vid]
        return len(vids)


def heartbeat(client_id: str, video_id: Optional[str] = None, name: str = "") -> Dict[str, Any]:
    """Renew lock for the client's current video (if any)."""
    now = time.time()
    with _lock:
        _purge_expired(now)
        renewed = []
        targets = [video_id] if video_id else list(_locks.keys())
        for vid in targets:
            info = _locks.get(vid)
            if not info or info.get("client_id") != client_id:
                continue
            if name.strip():
                info["name"] = name.strip()
            info["expires_at"] = now + LOCK_TTL_SEC
            info["updated_at"] = now
            renewed.append(vid)
        return {
            "ok": True,
            "renewed": renewed,
            "revision": _library_revision,
            "locks": [
                {
                    "video_id": vid,
                    "client_id": info["client_id"],
                    "name": info.get("name") or "Someone",
                    "expires_at": info["expires_at"],
                }
                for vid, info in _locks.items()
            ],
        }


def collab_status(since: int = 0) -> Dict[str, Any]:
    with _lock:
        _purge_expired()
        rev = _library_revision
        return {
            "revision": rev,
            "changed": rev > since,
            "locks": [
                {
                    "video_id": vid,
                    "client_id": info["client_id"],
                    "name": info.get("name") or "Someone",
                    "expires_at": info["expires_at"],
                }
                for vid, info in _locks.items()
            ],
            "ttl_sec": LOCK_TTL_SEC,
        }


def new_client_id() -> str:
    return uuid.uuid4().hex[:12]
