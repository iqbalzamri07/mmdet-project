#!/usr/bin/env python3
"""
Quick example: Train and test pose classification in 5 minutes
This script demonstrates the complete workflow with minimal configuration
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_pose_classifier import (
    PoseConfig, PoseTrainer, PoseSequenceDataset, 
    create_dummy_pose_estimator
)
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def create_quick_demo_data(num_samples=200):
    """Create small demo dataset for quick testing"""
    print("Creating quick demo data...")
    
    config = PoseConfig()
    sequences = []
    labels = []
    
    # Create distinct patterns for each class to make the demo more realistic
    for i in range(num_samples):
        label = i % config.num_classes
        
        # Create different pose patterns based on class
        if label == 0:  # smoking - hand to mouth motion
            sequence = create_smoking_pattern(config.sequence_length)
        elif label == 1:  # sitting - lower body compression
            sequence = create_sitting_pattern(config.sequence_length)
        elif label == 2:  # standing - upright posture
            sequence = create_standing_pattern(config.sequence_length)
        elif label == 3:  # walking - leg movement pattern
            sequence = create_walking_pattern(config.sequence_length)
        elif label == 4:  # calling - hand to head
            sequence = create_calling_pattern(config.sequence_length)
        else:  # playing_phone - phone interaction
            sequence = create_phone_pattern(config.sequence_length)
        
        # Add some randomness
        sequence += np.random.normal(0, 0.02, sequence.shape)
        
        # Save to temporary file
        temp_path = Path(config.data_dir) / f"demo_{i:04d}.npz"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        scores = np.ones((config.sequence_length, config.num_keypoints, 1), dtype=np.float32)
        np.savez_compressed(temp_path, keypoints=sequence, scores=scores, label=label)
        
        sequences.append(temp_path)
        labels.append(label)
    
    print(f"Created {len(sequences)} demo samples")
    return sequences, labels


def create_smoking_pattern(seq_len):
    """Create smoking-like pose pattern"""
    keypoints = np.zeros((seq_len, 17, 2))
    
    # Base standing pose
    for t in range(seq_len):
        keypoints[t] = create_base_pose(t, seq_len)
        
        # Hand to mouth motion (smoking gesture)
        mouth_y = 0.25 + 0.02 * np.sin(2 * np.pi * t / seq_len)
        hand_x = 0.5 + 0.1 * np.sin(2 * np.pi * t / seq_len)
        hand_y = mouth_y
        
        # Update right hand position (keypoint 10)
        keypoints[t, 10] = [hand_x, hand_y]
    
    return keypoints


def create_sitting_pattern(seq_len):
    """Create sitting-like pose pattern"""
    keypoints = np.zeros((seq_len, 17, 2))
    
    for t in range(seq_len):
        # Lower body compression for sitting
        keypoints[t] = create_base_pose(t, seq_len)
        
        # Compress lower body
        keypoints[t, 11:17, 1] *= 0.7  # Raise hips, knees, ankles
        keypoints[t, 11:17, 1] += 0.3  # Move up
        
        # Bent knees
        keypoints[t, 13:15, 0] = keypoints[t, 11:13, 0] + 0.05  # Knees forward
    
    return keypoints


def create_standing_pattern(seq_len):
    """Create standing-like pose pattern"""
    keypoints = np.zeros((seq_len, 17, 2))
    
    for t in range(seq_len):
        keypoints[t] = create_base_pose(t, seq_len)
        # Natural standing pose with slight swaying
        sway = 0.02 * np.sin(2 * np.pi * t / seq_len)
        keypoints[t, :, 0] += sway
    
    return keypoints


def create_walking_pattern(seq_len):
    """Create walking-like pose pattern"""
    keypoints = np.zeros((seq_len, 17, 2))
    
    for t in range(seq_len):
        keypoints[t] = create_base_pose(t, seq_len)
        
        # Leg movement pattern
        step_phase = 2 * np.pi * t / (seq_len / 2)  # Two steps per sequence
        
        # Left leg (keypoints 13, 15)
        left_leg_offset = 0.1 * np.sin(step_phase)
        keypoints[t, 13, 0] += left_leg_offset
        keypoints[t, 15, 0] += left_leg_offset * 1.5
        
        # Right leg (keypoints 14, 16) - opposite phase
        right_leg_offset = 0.1 * np.sin(step_phase + np.pi)
        keypoints[t, 14, 0] += right_leg_offset
        keypoints[t, 16, 0] += right_leg_offset * 1.5
        
        # Arm swing opposite to legs
        keypoints[t, 7, 0] -= left_leg_offset * 0.5  # Left arm
        keypoints[t, 8, 0] -= right_leg_offset * 0.5  # Right arm
    
    return keypoints


def create_calling_pattern(seq_len):
    """Create calling-like pose pattern"""
    keypoints = np.zeros((seq_len, 17, 2))
    
    for t in range(seq_len):
        keypoints[t] = create_base_pose(t, seq_len)
        
        # Hand to head (calling gesture)
        head_y = 0.2
        hand_x = 0.5 + 0.15 * np.sin(2 * np.pi * t / seq_len)
        hand_y = head_y + 0.05
        
        # Update right hand position (keypoint 10)
        keypoints[t, 10] = [hand_x, hand_y]
        
        # Elbow bent (keypoint 8)
        keypoints[t, 8] = [hand_x - 0.1, hand_y + 0.1]
    
    return keypoints


def create_phone_pattern(seq_len):
    """Create phone usage-like pose pattern"""
    keypoints = np.zeros((seq_len, 17, 2))
    
    for t in range(seq_len):
        keypoints[t] = create_base_pose(t, seq_len)
        
        # Phone holding position - lower than calling
        phone_y = 0.4 + 0.05 * np.sin(2 * np.pi * t / seq_len)
        phone_x = 0.5 + 0.1 * np.cos(2 * np.pi * t / seq_len)
        
        # Both hands potentially involved
        keypoints[t, 9] = [phone_x - 0.1, phone_y]  # Left hand
        keypoints[t, 10] = [phone_x + 0.1, phone_y]  # Right hand
        
        # Head looking down
        keypoints[t, 0, 1] += 0.05  # Head tilts down
        
        # Elbows bent
        keypoints[t, 7] = [phone_x - 0.15, phone_y + 0.1]
        keypoints[t, 8] = [phone_x + 0.15, phone_y + 0.1]
    
    return keypoints


def create_base_pose(t, seq_len):
    """Create a base human pose"""
    keypoints = np.zeros((17, 2))
    
    # Head and face
    keypoints[0] = [0.5, 0.2]  # Nose
    keypoints[1] = [0.48, 0.18]  # Left eye
    keypoints[2] = [0.52, 0.18]  # Right eye
    keypoints[3] = [0.47, 0.2]   # Left ear
    keypoints[4] = [0.53, 0.2]   # Right ear
    
    # Shoulders
    keypoints[5] = [0.4, 0.3]   # Left shoulder
    keypoints[6] = [0.6, 0.3]   # Right shoulder
    
    # Elbows
    keypoints[7] = [0.35, 0.45]  # Left elbow
    keypoints[8] = [0.65, 0.45]  # Right elbow
    
    # Wrists
    keypoints[9] = [0.3, 0.6]   # Left wrist
    keypoints[10] = [0.7, 0.6]  # Right wrist
    
    # Hips
    keypoints[11] = [0.42, 0.7]  # Left hip
    keypoints[12] = [0.58, 0.7]  # Right hip
    
    # Knees
    keypoints[13] = [0.4, 0.85]  # Left knee
    keypoints[14] = [0.6, 0.85]  # Right knee
    
    # Ankles
    keypoints[15] = [0.38, 1.0]  # Left ankle
    keypoints[16] = [0.62, 1.0]  # Right ankle
    
    return keypoints


def quick_demo():
    """Run a quick demonstration"""
    
    print("="*60)
    print("QUICK POSE CLASSIFICATION DEMO")
    print("="*60)
    print()
    
    # Create configuration with reduced parameters for speed
    config = PoseConfig()
    config.num_epochs = 20  # Reduced for quick demo
    config.batch_size = 8   # Smaller batches for speed
    config.hidden_dim = 128 # Smaller model for speed
    config.num_layers = 4   # Fewer layers for speed
    
    print("Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Model layers: {config.num_layers}")
    print(f"  Device: {config.device}")
    print()
    
    # Create demo data
    sequences, labels = create_quick_demo_data(num_samples=200)
    
    # Split data
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    print()
    
    # Create datasets and dataloaders
    train_dataset = PoseSequenceDataset(train_sequences, train_labels, config, augment=True)
    val_dataset = PoseSequenceDataset(val_sequences, val_labels, config, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for demo
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Train the model
    print("Starting training...")
    print()
    
    trainer = PoseTrainer(config)
    trainer.train(train_loader, val_loader)
    
    # Test on a few examples
    print("\n" + "="*60)
    print("TESTING ON EXAMPLES")
    print("="*60)
    
    from train_pose_classifier import PoseInference
    
    best_model_path = Path(config.checkpoint_dir) / 'best_model.pth'
    if best_model_path.exists():
        inference = PoseInference(str(best_model_path), config)
        
        # Test examples
        test_examples = [
            ("Smoking", create_smoking_pattern(64)),
            ("Sitting", create_sitting_pattern(64)),
            ("Standing", create_standing_pattern(64)),
            ("Walking", create_walking_pattern(64)),
            ("Calling", create_calling_pattern(64)),
            ("Phone", create_phone_pattern(64))
        ]
        
        print("\nPrediction Results:")
        for true_label, sequence in test_examples:
            action, confidence, probs = inference.predict_sequence(sequence)
            
            # Get top 3 predictions
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_preds = [(config.class_names[i], probs[i]) for i in top3_idx]
            
            print(f"\nTrue: {true_label}")
            print(f"Predicted: {action} (confidence: {confidence:.3f})")
            print("Top 3 predictions:")
            for name, prob in top3_preds:
                print(f"  {name}: {prob:.3f}")
    
    print("\n" + "="*60)
    print("🎉 Quick demo complete!")
    print("="*60)
    print(f"Model saved to: {config.checkpoint_dir}")
    print(f"Outputs saved to: {config.output_dir}")
    print()
    print("Next steps:")
    print("1. Try with your real data using: python run_pose_training.py")
    print("2. Customize the configuration in pose_classifier_config.json")
    print("3. Run inference on videos using: python run_pose_inference.py")


if __name__ == "__main__":
    quick_demo()