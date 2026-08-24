"""Video labeler configuration."""

import json
from pathlib import Path

# Project roots
LABELER_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LABELER_ROOT.parent

DATA_DIR = LABELER_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
EXPORTS_DIR = DATA_DIR / "exports"
JOBS_DIR = DATA_DIR / "jobs"

# Isolated export root (never wipe hand-curated libraries outside this path)
TRAINING_VIDEOS_DIR = PROJECT_ROOT / "data" / "actionmark_dataset"
CLEAN_VIDEOS_DIR = PROJECT_ROOT / "data" / "custom_actions_videos_clean"

# Activity-only taxonomy.
# (Historically this project also supported posture + activity, but we now train on activity only.)
POSTURE_LABELS: list[str] = []
ACTIVITY_LABELS = [
    "walking",
    "calling",
    "playing_phone",
    "smoking",
    "eating",
]
ACTION_LABELS = list(POSTURE_LABELS) + ACTIVITY_LABELS

# Train / val split for export
VAL_RATIO = 0.2
RANDOM_SEED = 42

# SlowFast training
TRAIN_CONFIG = PROJECT_ROOT / "configs" / "slowfast_multilabel.py"
TRAIN_SCRIPT = PROJECT_ROOT / "mmaction2" / "tools" / "train.py"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
WORK_DIR = PROJECT_ROOT / "work_dirs" / "slowfast_multilabel"

# Clip export
MIN_CLIP_FRAMES = 8
DEFAULT_FPS = 30


def persist_label_taxonomy() -> None:
    """Write current posture/activity lists to labels.json."""
    global ACTION_LABELS
    ACTION_LABELS = list(dict.fromkeys(POSTURE_LABELS + ACTIVITY_LABELS))
    labels_file = DATA_DIR / "labels.json"
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": ACTION_LABELS,
                "postures": POSTURE_LABELS,
                "activities": ACTIVITY_LABELS,
            },
            f,
            indent=2,
        )


def load_label_taxonomy() -> None:
    """Load posture/activity lists from labels.json if present."""
    global ACTION_LABELS, POSTURE_LABELS, ACTIVITY_LABELS
    labels_file = DATA_DIR / "labels.json"
    if not labels_file.exists():
        return
    try:
        with open(labels_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    postures = data.get("postures")
    activities = data.get("activities")
    labels = data.get("labels")
    if postures:
        POSTURE_LABELS = [x for x in postures if x]
    if activities:
        ACTIVITY_LABELS = [x for x in activities if x]
    ACTION_LABELS = list(dict.fromkeys(POSTURE_LABELS + ACTIVITY_LABELS))
    if labels:
        for x in labels:
            if x and x not in ACTION_LABELS:
                ACTION_LABELS.append(x)


def is_posture(name: str) -> bool:
    return (name or "") in POSTURE_LABELS


def is_activity(name: str) -> bool:
    return (name or "") in ACTIVITY_LABELS


def segment_class_names(seg: dict) -> list:
    """Return unique class names for a segment (activity-only)."""
    names = []
    activity = (seg.get("activity") or "").strip()
    legacy = (seg.get("label") or "").strip()
    if activity and activity in ACTION_LABELS:
        names.append(activity)
    if not names and legacy:
        parts = [p.strip() for p in legacy.replace(",", "+").split("+") if p.strip()]
        for part in parts:
            if part in ACTION_LABELS:
                names.append(part)
    # de-dupe, keep order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


for d in (VIDEOS_DIR, ANNOTATIONS_DIR, EXPORTS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)
