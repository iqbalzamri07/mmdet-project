"""
Create annotation files for SlowFast training
FIXED: Strict single/multi-label mapping (No incorrect combinations!)
"""

import os
import shutil
from pathlib import Path

# Configuration - STICK TO 5 CLASSES SINCE YOU TRAINED ON 5
DATA_ROOT = "data/custom_actions_videos"
SYMLINK_ROOT = "data/custom_actions_videos_clean"

ACTION_LABELS = ["sitting", "standing", "walking", "calling", "playing_phone"]
NUM_CLASSES = len(ACTION_LABELS)

def create_clean_dataset():
    """Create dataset with no spaces in filenames using symlinks"""
    print("Creating clean dataset with symlinks...")

    if os.path.exists(SYMLINK_ROOT):
        shutil.rmtree(SYMLINK_ROOT)
    os.makedirs(SYMLINK_ROOT, exist_ok=True)

    for split in ['train', 'val']:
        for label in ACTION_LABELS:
            target_dir = os.path.join(SYMLINK_ROOT, split, label)
            os.makedirs(target_dir, exist_ok=True)

            source_dir = os.path.join(DATA_ROOT, split, label)

            if not os.path.exists(source_dir):
                print(f"  ⚠️  Skipping {split}/{label} - directory not found")
                continue

            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
                video_files.extend(Path(source_dir).glob(ext))

            print(f"  {split}/{label}: {len(video_files)} videos")

            for i, video_file in enumerate(video_files):
                clean_name = f"video_{i:05d}{video_file.suffix}"
                target_path = os.path.join(target_dir, clean_name)

                try:
                    os.symlink(os.path.abspath(video_file), target_path)
                except OSError:
                    shutil.copy2(video_file, target_path)

    print(f"✅ Clean dataset created: {SYMLINK_ROOT}")


def create_annotation_file(split):
    """Create annotation file - STRICT LABELING"""
    split_dir = os.path.join(SYMLINK_ROOT, split)
    output_file = os.path.join(SYMLINK_ROOT, f"{split}_list.txt")

    if not os.path.exists(split_dir):
        print(f"❌ Directory not found: {split_dir}")
        return False

    video_list = []

    for label_idx, label in enumerate(ACTION_LABELS):
        label_dir = os.path.join(split_dir, label)

        if not os.path.exists(label_dir):
            continue

        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
            video_files.extend(Path(label_dir).glob(ext))

        for video_file in video_files:
            rel_path = os.path.relpath(video_file, SYMLINK_ROOT)
            
            # STRICT LABELING: Only assign the exact folder it is in.
            # If it is in the "calling" folder, it gets a 3. Nothing else.
            label_str = str(label_idx)
            
            video_list.append(f"{rel_path} {label_str}")

    # Write to file
    with open(output_file, 'w') as f:
        for item in video_list:
            f.write(item + '\n')

    print(f"✅ Created {output_file} with {len(video_list)} videos")
    
    # Print a sample to verify
    if video_list:
        print(f"   Sample line: {video_list[0]}")
    
    return True


def main():
    print("="*60)
    print("Creating Clean Dataset for SlowFast")
    print("="*60)
    print(f"Classes ({NUM_CLASSES}): {ACTION_LABELS}\n")

    if not os.path.exists(DATA_ROOT):
        print(f"❌ Data directory not found: {DATA_ROOT}")
        return

    create_clean_dataset()

    print("\nCreating train annotation...")
    train_success = create_annotation_file('train')

    print("\nCreating val annotation...")
    val_success = create_annotation_file('val')

    if train_success and val_success:
        print("\n" + "="*60)
        print("✅ Ready for training!")
        print("="*60)


if __name__ == "__main__":
    main()