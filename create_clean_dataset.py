"""
Create annotation files for SlowFast training
Creates symlinks to avoid spaces in filenames
"""

import os
import shutil
from pathlib import Path

# Configuration
DATA_ROOT = "data/custom_actions_videos"
SYMLINK_ROOT = "data/custom_actions_videos_clean"
ACTION_LABELS = ["smoking", "sitting", "standing", "walking", "calling", "playing_phone"]

def create_clean_dataset():
    """Create dataset with no spaces in filenames using symlinks"""
    print("Creating clean dataset with symlinks...")

    if os.path.exists(SYMLINK_ROOT):
        shutil.rmtree(SYMLINK_ROOT)
    os.makedirs(SYMLINK_ROOT, exist_ok=True)

    for split in ['train', 'val']:
        for label in ACTION_LABELS:
            # Create target directory
            target_dir = os.path.join(SYMLINK_ROOT, split, label)
            os.makedirs(target_dir, exist_ok=True)

            # Source directory
            source_dir = os.path.join(DATA_ROOT, split, label)

            if not os.path.exists(source_dir):
                print(f"  ⚠️  Skipping {split}/{label} - directory not found")
                continue

            # Find all video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
                video_files.extend(Path(source_dir).glob(ext))

            print(f"  {split}/{label}: {len(video_files)} videos")

            # Create symlinks with clean names
            for i, video_file in enumerate(video_files):
                # Remove spaces and special characters
                clean_name = f"video_{i:05d}{video_file.suffix}"
                target_path = os.path.join(target_dir, clean_name)

                # Create symlink
                try:
                    os.symlink(os.path.abspath(video_file), target_path)
                except OSError as e:
                    # If symlinks don't work, copy instead
                    shutil.copy2(video_file, target_path)
                    print(f"    Copied (symlink failed): {clean_name}")

    print(f"✅ Clean dataset created: {SYMLINK_ROOT}")

def create_annotation_file(split):
    """Create annotation file for train or val split"""
    split_dir = os.path.join(SYMLINK_ROOT, split)
    output_file = os.path.join(SYMLINK_ROOT, f"{split}_list.txt")

    if not os.path.exists(split_dir):
        print(f"❌ Directory not found: {split_dir}")
        return False

    video_list = []

    for label in ACTION_LABELS:
        label_dir = os.path.join(split_dir, label)

        if not os.path.exists(label_dir):
            continue

        # Find all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
            video_files.extend(Path(label_dir).glob(ext))

        # Add to list with class index
        for video_file in video_files:
            rel_path = os.path.relpath(video_file, SYMLINK_ROOT)
            video_list.append(f"{rel_path} {ACTION_LABELS.index(label)}")

    # Write to file
    with open(output_file, 'w') as f:
        for item in video_list:
            f.write(item + '\n')

    print(f"✅ Created {output_file} with {len(video_list)} videos")
    return True

def main():
    print("="*60)
    print("Creating Clean Dataset for SlowFast")
    print("="*60)

    # Check if original data directory exists
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Data directory not found: {DATA_ROOT}")
        print("\nPlease run: python prepare_slowfast_dataset.py")
        return

    # Create clean dataset with symlinks
    create_clean_dataset()

    # Create train annotation
    print("\nCreating train annotation...")
    train_success = create_annotation_file('train')

    # Create val annotation
    print("\nCreating val annotation...")
    val_success = create_annotation_file('val')

    if train_success and val_success:
        print("\n" + "="*60)
        print("✅ Ready for training!")
        print("="*60)
        print(f"\nClean dataset: {SYMLINK_ROOT}")
        print(f"Annotation files:")
        print(f"  - {SYMLINK_ROOT}/train_list.txt")
        print(f"  - {SYMLINK_ROOT}/val_list.txt")
        print("\nNow update config to use:")
        print(f"  ann_file: '{SYMLINK_ROOT}/train_list.txt'")
        print(f"  data_prefix: {{video: '{SYMLINK_ROOT}'}}")
        print("\nThen run:")
        print("  ./train_slowfast_fixed.sh")
        print("="*60)
    else:
        print("\n❌ Failed to create annotation files")

if __name__ == "__main__":
    main()
