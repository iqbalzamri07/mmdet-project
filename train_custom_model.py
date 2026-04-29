"""
Training Script for Custom Action Recognition Model
Trains ST-GCN on your 6 custom actions

Usage:
    python train_custom_model.py
"""

import os
import sys
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmaction.utils import register_all_modules

# Register mmaction modules
register_all_modules()


def main():
    print("=" * 60)
    print("Custom Action Recognition - Training")
    print("=" * 60)

    # Load configuration
    config_file = "configs/custom_action_recognition.py"

    if not os.path.exists(config_file):
        print(f"\n❌ Error: Config file not found: {config_file}")
        print("Please create the config file first.")
        sys.exit(1)

    print(f"\nLoading config: {config_file}")
    cfg = Config.fromfile(config_file)

    # Verify dataset exists
    train_data_file = "data/custom_actions_processed/train_data.pkl"
    val_data_file = "data/custom_actions_processed/val_data.pkl"

    if not os.path.exists(train_data_file):
        print(f"\n❌ Error: Training data not found: {train_data_file}")
        print("Please run: python prepare_dataset.py")
        sys.exit(1)

    if not os.path.exists(val_data_file):
        print(f"\n❌ Error: Validation data not found: {val_data_file}")
        print("Please run: python prepare_dataset.py")
        sys.exit(1)

    print(f"✓ Training data: {train_data_file}")
    print(f"✓ Validation data: {val_data_file}")

    # Load dataset info
    import pickle

    with open(train_data_file, "rb") as f:
        train_data = pickle.load(f)
    with open(val_data_file, "rb") as f:
        val_data = pickle.load(f)

    # Load labels
    if os.path.exists("data/custom_actions_processed/labels.txt"):
        with open("data/custom_actions_processed/labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        labels = [
            "smoking",
            "sitting",
            "standing",
            "walking",
            "calling",
            "playing_phone",
        ]

    print(f"\nDataset info:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")

    if len(train_data) > 0:
        first_anno = train_data[0]
        print(f"  Num frames per sample: {first_anno['total_frames']}")
        print(f"  Num keypoints: {first_anno['keypoint'].shape[2]}")

    print(f"\nLabel distribution (train):")
    label_counts = {}
    for anno in train_data:
        label_name = labels[anno["label"]]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    for label, count in label_counts.items():
        print(f"  {label}: {count} videos")

    # Check GPU
    num_gpus = torch.cuda.device_count()
    print(f"\nGPU available: {num_gpus}")

    if num_gpus == 0:
        print("⚠️  Warning: No GPU detected. Training will be very slow!")
        # Reduce batch size for CPU
        cfg.train_dataloader.batch_size = 2
        cfg.val_dataloader.batch_size = 1

    # Create work directory
    work_dir = cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)
    print(f"\nWork directory: {work_dir}")

    # Build and run trainer
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    runner = Runner.from_cfg(cfg)
    runner.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {work_dir}")
    print("Best model: best_acc_top1_epoch_*.pth")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Checkpoints are saved in the work directory.")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
