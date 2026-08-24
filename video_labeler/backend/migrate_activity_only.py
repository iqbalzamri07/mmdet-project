"""
Physically migrate the Label library from posture+activity -> activity-only.

This script updates:
1) `video_labeler/data/labels.json`:
   - sets `postures: []`
   - sets `labels: activities`
2) `video_labeler/data/annotations/*.json`:
   - removes `segment.posture`
   - keeps only segments that have an activity label
   - normalizes `segment.activity` and `segment.label` to the activity label

It makes a timestamped backup folder under `video_labeler/data/` before edits.

Usage:
  python3 -m video_labeler.backend.migrate_activity_only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _backup_dir(ts: str) -> Path:
    # Back up both labels + annotations so you can revert safely.
    return config.DATA_DIR / f"backup_activity_only_{ts}"


def _load_labels_json(labels_path: Path) -> Dict[str, Any]:
    if not labels_path.exists():
        return {"labels": [], "postures": [], "activities": list(config.ACTIVITY_LABELS)}
    return json.loads(labels_path.read_text(encoding="utf-8"))


def _extract_activity_from_label(label: str, activity_set: set[str]) -> str:
    # Legacy labels may look like "standing + walking" or "sitting, smoking".
    parts = re.split(r"[+,]", label or "")
    for p in parts:
        p = p.strip()
        if p in activity_set:
            return p
    return ""


def migrate_once(
    labels_path: Path,
    annotations_dir: Path,
) -> Tuple[int, int]:
    labels_data = _load_labels_json(labels_path)
    activities = labels_data.get("activities") or labels_data.get("labels") or list(config.ACTIVITY_LABELS)
    activities = [a.strip() for a in activities if a and str(a).strip()]
    activity_set = set(activities)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_root = _backup_dir(ts)
    backup_root.mkdir(parents=True, exist_ok=True)

    # Copy labels.json
    if labels_path.exists():
        shutil.copy2(labels_path, backup_root / "labels.json.bak")

    # Copy annotations json files
    ann_backup = backup_root / "annotations"
    ann_backup.mkdir(parents=True, exist_ok=True)
    for p in sorted(annotations_dir.glob("*.json")):
        shutil.copy2(p, ann_backup / p.name)

    # Update labels.json to activity-only
    labels_data["postures"] = []
    labels_data["activities"] = activities
    labels_data["labels"] = activities
    labels_path.write_text(json.dumps(labels_data, indent=2), encoding="utf-8")

    # Update each annotation file
    updated_files = 0
    kept_segments = 0
    for ann_path in sorted(annotations_dir.glob("*.json")):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        segs = data.get("segments") or []
        new_segs: List[Dict[str, Any]] = []
        changed = False

        for seg in segs:
            if not isinstance(seg, dict):
                continue
            seg = dict(seg)

            activity = (seg.get("activity") or "").strip()
            if not activity:
                activity = _extract_activity_from_label(seg.get("label") or "", activity_set)

            # Remove posture regardless.
            seg.pop("posture", None)

            if not activity:
                changed = True
                continue

            # Normalize label fields for older tools.
            seg["activity"] = activity
            seg["label"] = activity
            new_segs.append(seg)
            kept_segments += 1

        if new_segs != segs:
            changed = True

        if changed:
            data["segments"] = new_segs
            data["updated_at"] = _now_iso()
            ann_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            updated_files += 1

    return updated_files, kept_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate posture annotations -> activity-only")
    parser.add_argument(
        "--labels-json",
        default=str(config.DATA_DIR / "labels.json"),
        help="Path to labels.json",
    )
    parser.add_argument(
        "--annotations-dir",
        default=str(config.ANNOTATIONS_DIR),
        help="Path to annotations directory",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels_json)
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.exists():
        raise SystemExit(f"annotations dir not found: {annotations_dir}")

    updated_files, kept_segments = migrate_once(labels_path, annotations_dir)
    print(f"Migration done. Updated files: {updated_files}. Kept segments: {kept_segments}.")


if __name__ == "__main__":
    main()

