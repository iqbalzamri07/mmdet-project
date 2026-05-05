#!/bin/bash

# Train SlowFast on Custom Action Recognition Dataset

set -e

echo "=========================================="
echo "SlowFast Custom Training Script"
echo "=========================================="

# Check if config exists
if [ ! -f "configs/slowfast_custom.py" ]; then
    echo "❌ Config file not found: configs/slowfast_custom.py"
    exit 1
fi

# Check if dataset exists
if [ ! -d "data/custom_actions_videos" ]; then
    echo "❌ Dataset not found: data/custom_actions_videos"
    echo "Please run: python prepare_slowfast_dataset.py"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "checkpoints/slowfast_r50_kinetics400_rgb.pth" ]; then
    echo "⚠️  Pretrained checkpoint not found"
    echo "Downloading..."
    mkdir -p checkpoints
    wget -O checkpoints/slowfast_r50_kinetics400_rgb.pth \
        https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth
fi

echo ""
echo "Configuration:"
echo "  Config: configs/slowfast_custom.py"
echo "  Dataset: data/custom_actions_videos"
echo "  Work Dir: work_dirs/slowfast_custom"
echo ""

# Start training using the correct MMAction2 training command
echo "Starting training..."
python mmaction2/tools/train.py configs/slowfast_custom.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Checkpoints saved to: work_dirs/slowfast_custom/"
echo "Best checkpoint: best_acc_top1_epoch_*.pth"
echo "=========================================="