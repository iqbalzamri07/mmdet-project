# Pose Classification Training System

A modern, Transformer-based pose classification training system built from scratch with PyTorch.

## Features

- **Transformer Architecture**: Uses self-attention mechanisms for pose sequence processing
- **Multi-class Classification**: Supports 6 action classes by default
- **Data Augmentation**: Includes noise, scaling, and temporal jittering
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Real-time Inference**: Efficient inference engine for video processing
- **Visualization**: Training curves and confusion matrix generation

## Installation

```bash
pip install torch torchvision
pip install opencv-python numpy matplotlib seaborn scikit-learn tqdm
```

## Usage

### 1. Prepare Your Data

Organize your videos in the following structure:
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

### 2. Configure the System

Edit the `PoseConfig` class in `train_pose_classifier.py` to customize:
- Number of classes and class names
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Paths and directories

### 3. Train the Model

```bash
python train_pose_classifier.py
```

The training process will:
1. Extract pose sequences from your videos
2. Create training and validation splits
3. Train the Transformer model
4. Save checkpoints and training plots
5. Generate evaluation metrics

### 4. Use for Inference

```python
from train_pose_classifier import PoseInference, PoseConfig

# Load configuration and model
config = PoseConfig()
inference = PoseInference('checkpoints/pose_classifier/best_model.pth', config)

# Predict on a pose sequence
action, confidence, probs = inference.predict_sequence(your_pose_sequence)

# Process entire video
results = inference.predict_video('path/to/video.mp4', your_pose_estimator)
```

## Model Architecture

The system uses a Transformer encoder with:

- **Input Projection**: Maps pose keypoints to hidden dimension
- **Positional Encoding**: Adds temporal position information
- **Transformer Layers**: 6 layers with 8 attention heads
- **Classification Head**: Maps to final class predictions

### Key Parameters

- `num_keypoints`: 17 (COCO format)
- `sequence_length`: 64 frames per sequence
- `hidden_dim`: 256
- `num_heads`: 8
- `num_layers`: 6

## Training Configuration

Default training settings:

- **Batch Size**: 16
- **Epochs**: 100
- **Learning Rate**: 1e-4 with cosine annealing
- **Optimizer**: AdamW with weight decay 1e-5
- **Warmup**: 10 epochs
- **Gradient Clipping**: 1.0

## Data Augmentation

- **Gaussian Noise**: std=0.02
- **Random Scaling**: 0.9-1.1
- **Temporal Jittering**: ±3 frames

## Output Files

The system generates:

- **Checkpoints**: `checkpoints/pose_classifier/`
  - `best_model.pth` - Best validation accuracy
  - `checkpoint_epoch_*.pth` - Periodic checkpoints

- **Logs**: `logs/pose_classifier/`
  - Training progress and metrics

- **Outputs**: `outputs/pose_classifier/`
  - `training_plots_*.png` - Loss and accuracy curves
  - `confusion_matrix_*.png` - Confusion matrix visualization

- **Data**: `data/pose_sequences/`
  - Processed pose sequences in `.npz` format
  - `data_split.json` - Train/val split information

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Batch Size**: Adjust based on GPU memory
3. **Sequence Length**: Longer sequences capture more context but use more memory
4. **Data Augmentation**: Enable for better generalization
5. **Learning Rate**: Adjust based on training convergence

## Integration with Pose Estimation

To integrate with your existing pose estimation pipeline:

```python
# Replace the dummy pose estimator with your real one
from mmpose.apis import init_model as init_pose_model, inference_topdown

pose_config = "configs/mmpose/rtmpose-t_8xb256-420e_coco-256x192.py"
pose_checkpoint = "checkpoints/rtmpose-t_*.pth"
pose_model = init_pose_model(pose_config, pose_checkpoint, device='cuda:0')

def real_pose_estimator(frame):
    result = inference_topdown(pose_model, frame, [[0, 0, frame.shape[1], frame.shape[0]]])
    if result and hasattr(result[0], 'pred_instances'):
        keypoints = result[0].pred_instances.keypoints[0].cpu().numpy()
        return keypoints[:, :2]  # Return x, y coordinates
    return None

# Use in preprocessor
preprocessor = PoseDataPreprocessor(real_pose_estimator, config)
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `sequence_length`
- Reduce `hidden_dim` or `num_layers`

### Poor Accuracy
- Increase training epochs
- Adjust learning rate
- Enable data augmentation
- Check data quality and balance

### Slow Training
- Increase `batch_size` if memory allows
- Reduce `num_workers` if CPU bottleneck
- Use mixed precision training

## Future Enhancements

Potential improvements:

- [ ] Mixed precision training (FP16)
- [ ] Multi-GPU training support
- [ ] Attention visualization
- [ ] Online learning capabilities
- [ ] Export to ONNX/TensorRT
- [ ] Real-time webcam inference

## License

This training system is part of the mmdet-project and follows the same license terms.