"""Video labeler configuration."""

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

# Action classes (extend as needed)
ACTION_LABELS = [
    "sitting",
    "standing",
    "walking",
    "calling",
    "playing_phone",
    "smoking",
    "eating",
]

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

for d in (VIDEOS_DIR, ANNOTATIONS_DIR, EXPORTS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)
