#!/usr/bin/env python3
"""
Simple training script for pose classification
Loads configuration from JSON file and runs training
"""

import argparse
import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_pose_classifier import (
    PoseConfig, PoseTrainer, PoseSequenceDataset, 
    PoseDataPreprocessor, PoseInference, create_dummy_pose_estimator
)
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create PoseConfig from dictionary
    config = PoseConfig()
    
    # Update configuration
    config.num_keypoints = config_dict['model']['num_keypoints']
    config.keypoint_dims = config_dict['model']['keypoint_dims']
    config.hidden_dim = config_dict['model']['hidden_dim']
    config.num_heads = config_dict['model']['num_heads']
    config.num_layers = config_dict['model']['num_layers']
    config.dropout = config_dict['model']['dropout']
    config.max_seq_len = config_dict['model']['max_seq_len']
    
    config.batch_size = config_dict['training']['batch_size']
    config.num_epochs = config_dict['training']['num_epochs']
    config.learning_rate = config_dict['training']['learning_rate']
    config.weight_decay = config_dict['training']['weight_decay']
    config.warmup_epochs = config_dict['training']['warmup_epochs']
    config.gradient_clip = config_dict['training']['gradient_clip']
    
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
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    
    return config


def prepare_data(config, mode='dummy'):
    """Prepare training data"""
    
    if mode == 'dummy':
        # Create dummy data for testing
        print("Creating dummy data for testing...")
        num_samples = 1000
        sequences = []
        labels = []
        
        for i in range(num_samples):
            # Create random pose sequence
            sequence = np.random.rand(
                config.sequence_length, 
                config.num_keypoints, 
                config.keypoint_dims
            ).astype(np.float32)
            
            # Create temporary file
            temp_path = Path(config.data_dir) / f"dummy_{i:05d}.npz"
            
            label = np.random.randint(0, config.num_classes)
            scores = np.ones((config.sequence_length, config.num_keypoints, 1), dtype=np.float32)
            
            np.savez_compressed(
                temp_path, 
                keypoints=sequence, 
                scores=scores,
                label=label
            )
            
            sequences.append(temp_path)
            labels.append(label)
            
        print(f"Created {len(sequences)} dummy samples")
        
    elif mode == 'process':
        # Process real videos
        print("Processing real videos...")
        
        pose_estimator = create_dummy_pose_estimator()  # Replace with real estimator
        preprocessor = PoseDataPreprocessor(pose_estimator, config)
        
        raw_video_dir = Path(config.raw_video_dir)
        if not raw_video_dir.exists():
            print(f"Error: Video directory not found: {raw_video_dir}")
            return None, None
        
        # Create label mapping from class names
        label_mapping = {name: idx for idx, name in enumerate(config.class_names)}
        
        sequences, labels = preprocessor.process_directory(raw_video_dir, label_mapping)
        
        if len(sequences) == 0:
            print("Error: No sequences created from videos")
            return None, None
            
    elif mode == 'existing':
        # Load existing processed data
        print("Loading existing processed data...")
        
        data_split_file = Path(config.data_dir) / 'data_split.json'
        if not data_split_file.exists():
            print(f"Error: Data split file not found: {data_split_file}")
            print("Please run with --mode process first")
            return None, None
        
        with open(data_split_file, 'r') as f:
            split_info = json.load(f)
        
        # Combine train and val data for resplitting
        all_data = split_info['train'] + split_info['val']
        sequences = [Path(item[0]) for item in all_data]
        labels = [item[1] for item in all_data]
        
        print(f"Loaded {len(sequences)} existing samples")
        
    else:
        print(f"Error: Unknown mode {mode}")
        return None, None
    
    # Split data
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    
    return (train_sequences, train_labels), (val_sequences, val_labels)


def main():
    parser = argparse.ArgumentParser(description='Train Pose Classification Model')
    parser.add_argument('--config', type=str, default='pose_classifier_config.json',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='dummy',
                       choices=['dummy', 'process', 'existing'],
                       help='Data preparation mode')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing, no training')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    print("="*60)
    print("POSE CLASSIFICATION TRAINING")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Classes ({config.num_classes}): {config.class_names}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Sequence length: {config.sequence_length}")
    print("="*60)
    
    # Prepare data
    train_data, val_data = prepare_data(config, args.mode)
    if train_data is None or val_data is None:
        print("Failed to prepare data")
        return
    
    train_sequences, train_labels = train_data
    val_sequences, val_labels = val_data
    
    # Create datasets
    train_dataset = PoseSequenceDataset(
        train_sequences, train_labels, config, augment=True
    )
    val_dataset = PoseSequenceDataset(
        val_sequences, val_labels, config, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    # Create trainer
    trainer = PoseTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train or test only
    if args.test_only:
        print("Running test only...")
        trainer.final_evaluation(val_loader)
    else:
        print("Starting training...")
        trainer.train(train_loader, val_loader)
        
        # Test inference
        print("\n" + "="*60)
        print("TESTING INFERENCE")
        print("="*60)
        
        best_model_path = Path(config.checkpoint_dir) / 'best_model.pth'
        if best_model_path.exists():
            inference = PoseInference(str(best_model_path), config)
            
            # Test on a sample
            sample_sequence = np.random.rand(
                config.sequence_length, 
                config.num_keypoints, 
                config.keypoint_dims
            )
            action, confidence, probs = inference.predict_sequence(sample_sequence)
            
            print(f"Sample prediction: {action} (confidence: {confidence:.4f})")
            print(f"All probabilities:")
            for class_name, prob in zip(config.class_names, probs):
                print(f"  {class_name}: {prob:.4f}")
        else:
            print(f"Warning: Best model not found at {best_model_path}")
    
    print("\n" + "="*60)
    print("✅ Complete!")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"Outputs: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()