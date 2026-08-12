# Quick Start Guide - Pose Classification Training

This guide will help you quickly get started with training and using the pose classification system.

## Installation

```bash
# Install additional dependencies
pip install -r requirements_pose_classifier.txt
```

## Quick Test (5 minutes)

Test the system with dummy data to verify everything works:

```bash
# Train on dummy data
python run_pose_training.py --mode dummy --config pose_classifier_config.json

# This will:
# 1. Generate random pose sequences
# 2. Train the Transformer model
# 3. Save checkpoints and plots
# 4. Test inference
```

Expected output:
- Training progress with accuracy metrics
- Best model saved to `checkpoints/pose_classifier/best_model.pth`
- Training plots saved to `outputs/pose_classifier/`
- Confusion matrix and classification report

## Training on Real Data

### Step 1: Organize Your Videos

```
data/custom_actions_videos/
├── smoking/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── sitting/
├── standing/
├── walking/
├── calling/
└── playing_phone/
```

### Step 2: Configure the System

Edit `pose_classifier_config.json` to match your setup:

```json
{
  "data": {
    "num_classes": 6,
    "class_names": ["smoking", "sitting", "standing", "walking", "calling", "playing_phone"],
    "raw_video_dir": "data/custom_actions_videos"
  },
  "training": {
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 0.0001
  }
}
```

### Step 3: Process Videos and Train

```bash
# Process videos and train in one go
python run_pose_training.py --mode process --config pose_classifier_config.json
```

Or do it step by step:

```bash
# Step 1: Process videos into pose sequences
python run_pose_training.py --mode process --config pose_classifier_config.json

# Step 2: Train on processed data
python run_pose_training.py --mode existing --config pose_classifier_config.json
```

## Running Inference

### Basic Usage

```bash
# Classify actions in a video
python run_pose_inference.py \
    --video path/to/your/video.mp4 \
    --config pose_classifier_config.json \
    --model checkpoints/pose_classifier/best_model.pth \
    --output results.json
```

### With Visualization

```bash
# Create visualization with action labels
python run_pose_inference.py \
    --video path/to/your/video.mp4 \
    --config pose_classifier_config.json \
    --model checkpoints/pose_classifier/best_model.pth \
    --output results.json \
    -- visualize output_annotated.mp4
```

## Resume Training

If training was interrupted, resume from checkpoint:

```bash
python run_pose_training.py \
    --mode existing \
    --config pose_classifier_config.json \
    --resume checkpoints/pose_classifier/checkpoint_epoch_50.pth
```

## Testing Only

Evaluate model without training:

```bash
python run_pose_training.py \
    --mode existing \
    --config pose_classifier_config.json \
    --test-only
```

## Configuration Tips

### For Better Accuracy

```json
{
  "training": {
    "num_epochs": 200,
    "learning_rate": 0.00005,
    "batch_size": 8
  },
  "data": {
    "sequence_length": 128,
    "stride": 4
  }
}
```

### For Faster Training

```json
{
  "training": {
    "num_epochs": 50,
    "batch_size": 32
  },
  "data": {
    "sequence_length": 32,
    "stride": 16
  }
}
```

### For Limited Memory

```json
{
  "model": {
    "hidden_dim": 128,
    "num_layers": 4,
    "num_heads": 4
  },
  "training": {
    "batch_size": 4
  }
}
```

## Common Issues and Solutions

### Out of Memory

**Problem**: CUDA out of memory during training

**Solutions**:
- Reduce `batch_size` to 8 or 4
- Reduce `sequence_length` to 32
- Reduce `hidden_dim` to 128
- Use `--device cpu` (slower but works)

### Poor Accuracy

**Problem**: Model accuracy is low

**Solutions**:
- Increase `num_epochs` to 200+
- Enable data augmentation: `"augmentation": {"enabled": true}`
- Check if your data is balanced
- Reduce learning rate: `0.00001`
- Increase model capacity: `hidden_dim: 512`

### Slow Training

**Problem**: Training takes too long

**Solutions**:
- Increase `batch_size` if memory allows
- Reduce `num_workers` if CPU is bottleneck
- Use GPU instead of CPU
- Reduce `sequence_length` or `num_epochs`

## Integration with Existing Pipeline

To use your existing pose estimation:

```python
# Replace the dummy estimator in run_pose_inference.py
from mmpose.apis import init_model as init_pose_model, inference_topdown

pose_model = init_pose_model(
    "configs/mmpose/rtmpose-t_8xb256-420e_coco-256x192.py",
    "checkpoints/rtmpose-t_*.pth", 
    device='cuda:0'
)

def real_pose_estimator(frame):
    result = inference_topdown(pose_model, frame, [[0, 0, frame.shape[1], frame.shape[0]]])
    if result and hasattr(result[0], 'pred_instances'):
        keypoints = result[0].pred_instances.keypoints[0].cpu().numpy()
        return keypoints[:, :2]  # Return x, y coordinates
    return None
```

## Next Steps

1. **Customize Architecture**: Modify `PoseTransformerEncoder` for your needs
2. **Add More Classes**: Update `num_classes` and `class_names` in config
3. **Experiment with Augmentation**: Try different augmentation parameters
4. **Hyperparameter Tuning**: Use the config file to experiment with different settings
5. **Real-time Inference**: Adapt the inference code for webcam input

## Getting Help

- Check the main README: `README_POSE_CLASSIFIER.md`
- Examine training plots in `outputs/pose_classifier/`
- Review confusion matrix to identify problematic classes
- Check logs in `logs/pose_classifier/`

## Performance Expectations

With default settings on decent hardware:

- **Training Time**: 2-4 hours for 100 epochs on GPU
- **Inference Speed**: ~30-60 FPS for real-time applications
- **Expected Accuracy**: 70-90% depending on data quality

**Note**: These are rough estimates. Actual performance depends on your hardware and data.