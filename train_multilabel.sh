#!/bin/bash

set -e

echo "=========================================="
echo "Multi-Label SlowFast Training"
echo "=========================================="

# Step 1: Prepare dataset
echo "Step 1: Creating clean dataset..."
python create_clean_dataset.py

# Step 2: Download checkpoint if needed
if [ ! -f "checkpoints/slowfast_r50_kinetics400_rgb.pth" ]; then
    echo "Step 2: Downloading pretrained weights..."
    mkdir -p checkpoints
    wget -O checkpoints/slowfast_r50_kinetics400_rgb.pth \
        https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth
else
    echo "Step 2: Pretrained weights exist"
fi

# Step 3: Train with multi-label config
echo "Step 3: Starting multi-label training..."
echo "  Config: configs/slowfast_multilabel.py"
echo "  Work dir: work_dirs/slowfast_multilabel"

python mmaction2/tools/train.py configs/slowfast_multilabel.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Checkpoint: work_dirs/slowfast_multilabel/best_mAP_epoch_*.pth"
echo ""
echo "To run inference:"
echo "  python slowfast_memory_optimized.py"
echo "=========================================="