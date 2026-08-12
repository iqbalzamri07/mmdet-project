#!/usr/bin/env python3
"""
Inference script for pose classification
Process videos and classify actions using trained model
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_pose_classifier import PoseInference, PoseConfig


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = PoseConfig()
    
    # Update configuration
    config.num_keypoints = config_dict['model']['num_keypoints']
    config.keypoint_dims = config_dict['model']['keypoint_dims']
    config.hidden_dim = config_dict['model']['hidden_dim']
    config.num_heads = config_dict['model']['num_heads']
    config.num_layers = config_dict['model']['num_layers']
    config.dropout = config_dict['model']['dropout']
    config.max_seq_len = config_dict['model']['max_seq_len']
    
    config.sequence_length = config_dict['data']['sequence_length']
    config.stride = config_dict['data']['stride']
    config.num_classes = config_dict['data']['num_classes']
    config.class_names = config_dict['data']['class_names']
    config.data_dir = config_dict['data']['data_dir']
    
    config.use_augmentation = config_dict['augmentation']['enabled']
    config.noise_std = config_dict['augmentation']['noise_std']
    config.temporal_jitter = config_dict['augmentation']['temporal_jitter']
    config.scale_range = tuple(config_dict['augmentation']['scale_range'])
    
    config.checkpoint_dir = config_dict['paths']['checkpoint_dir']
    config.log_dir = config_dict['paths']['log_dir']
    config.output_dir = config_dict['paths']['output_dir']
    
    config.device = config_dict['system']['device']
    config.num_workers = config_dict['system']['num_workers']
    
    return config


def create_simple_pose_estimator():
    """Create a simple pose estimator (replace with real one)"""
    def simple_estimator(frame):
        # This is a placeholder - replace with actual pose estimation
        h, w = frame.shape[:2]
        num_keypoints = 17
        
        # For demo, return keypoints in a grid pattern
        keypoints = np.zeros((num_keypoints, 2))
        
        # Create a simple human-like pose pattern
        # Head (1-4)
        keypoints[0] = [w//2, h//4]
        keypoints[1] = [w//2 - 20, h//4 - 20]
        keypoints[2] = [w//2 + 20, h//4 - 20]
        keypoints[3] = [w//2 - 25, h//4]
        keypoints[4] = [w//2 + 25, h//4]
        
        # Shoulders (5-6)
        keypoints[5] = [w//2 - 60, h//3]
        keypoints[6] = [w//2 + 60, h//3]
        
        # Elbows (7-8)
        keypoints[7] = [w//2 - 80, h//2]
        keypoints[8] = [w//2 + 80, h//2]
        
        # Wrists (9-10)
        keypoints[9] = [w//2 - 70, h//2 + 60]
        keypoints[10] = [w//2 + 70, h//2 + 60]
        
        # Hips (11-12)
        keypoints[11] = [w//2 - 40, h//2 + 80]
        keypoints[12] = [w//2 + 40, h//2 + 80]
        
        # Knees (13-14)
        keypoints[13] = [w//2 - 50, h//2 + 160]
        keypoints[14] = [w//2 + 50, h//2 + 160]
        
        # Ankles (15-16)
        keypoints[15] = [w//2 - 60, h - 50]
        keypoints[16] = [w//2 + 60, h - 50]
        
        return keypoints
    
    return simple_estimator


def visualize_predictions(frame, results, class_names):
    """Draw prediction results on frame"""
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Find the active prediction at current time
    # For simplicity, show the first prediction
    if results:
        result = results[0]
        action = result['action']
        confidence = result['confidence']
        
        # Draw action label
        text = f"Action: {action} ({confidence:.2f})"
        cv2.putText(vis_frame, text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw all class probabilities
        y_offset = 80
        for class_name, prob in zip(class_names, result['probabilities']):
            prob_text = f"{class_name}: {prob:.3f}"
            cv2.putText(vis_frame, prob_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
    
    return vis_frame


def process_video(args, config, inference):
    """Process video file and classify actions"""
    
    print(f"Processing video: {args.video}")
    
    # Check if video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Create pose estimator
    pose_estimator = create_simple_pose_estimator()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Extract poses
    print("Extracting poses from video...")
    poses = []
    frame_indices = []
    frames = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 2 == 0:  # Process every 2nd frame
            pose_result = pose_estimator(frame)
            if pose_result is not None:
                poses.append(pose_result)
                frame_indices.append(frame_idx)
                frames.append(frame)
            else:
                if poses:
                    poses.append(poses[-1])
                    frame_indices.append(frame_idx)
                    frames.append(frame)
                else:
                    poses.append(np.zeros((config.num_keypoints, 2)))
                    frame_indices.append(frame_idx)
                    frames.append(frame)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    
    if len(poses) < config.sequence_length:
        print(f"Error: Video too short. Need at least {config.sequence_length} frames, got {len(poses)}")
        return
    
    print(f"Extracted {len(poses)} pose frames")
    
    # Convert to numpy array
    poses = np.array(poses)
    
    # Create overlapping sequences and predict
    print("Running pose classification...")
    results = []
    
    for start_idx in range(0, len(poses) - config.sequence_length + 1, config.stride):
        end_idx = start_idx + config.sequence_length
        sequence = poses[start_idx:end_idx]
        
        action_name, confidence, probabilities = inference.predict_sequence(sequence)
        
        start_frame = frame_indices[start_idx]
        end_frame = frame_indices[end_idx - 1]
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        result = {
            'start_time': start_time,
            'end_time': end_time,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'action': action_name,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
        
        results.append(result)
        
        if len(results) % 10 == 0:
            print(f"Processed {len(results)} sequences...")
    
    print(f"Generated {len(results)} predictions")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    # Create visualization video if requested
    if args.visualize:
        print("Creating visualization...")
        
        output_video_path = args.visualize
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Create a mapping from frame to predictions
        frame_to_results = {}
        for result in results:
            for frame_idx in range(result['start_frame'], result['end_frame'] + 1):
                if frame_idx not in frame_to_results:
                    frame_to_results[frame_idx] = []
                frame_to_results[frame_idx].append(result)
        
        # Process each frame
        for i, (frame, original_frame_idx) in enumerate(zip(frames, frame_indices)):
            # Get results for this frame
            frame_results = frame_to_results.get(original_frame_idx, [])
            
            # Visualize
            vis_frame = visualize_predictions(frame, frame_results, config.class_names)
            
            # Write frame
            out.write(vis_frame)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(frames)} frames for visualization...")
        
        out.release()
        print(f"Visualization saved to: {output_video_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    # Count action occurrences
    action_counts = {}
    for result in results:
        action = result['action']
        if action not in action_counts:
            action_counts[action] = 0
        action_counts[action] += 1
    
    print("\nAction distribution:")
    for action, count in action_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {action}: {count} ({percentage:.1f}%)")
    
    print(f"\nTotal predictions: {len(results)}")
    print(f"Video duration: {total_frames / fps:.2f} seconds")
    
    # Show first few predictions
    print("\nFirst 5 predictions:")
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. {result['start_time']:.2f}s - {result['end_time']:.2f}s: "
              f"{result['action']} ({result['confidence']:.3f})")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Pose Classification Inference')
    parser.add_argument('--config', type=str, default='pose_classifier_config.json',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/pose_classifier/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save JSON results')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Path to save visualization video')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found: {args.model}")
        print("Please train the model first using: python run_pose_training.py")
        return
    
    # Load inference model
    print("Loading model...")
    inference = PoseInference(str(model_path), config)
    
    # Process video
    process_video(args, config, inference)


if __name__ == "__main__":
    main()