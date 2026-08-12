"""
Modern Pose Classification Training System
A fresh implementation using Transformer-based architecture for pose sequence classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class PoseConfig:
    """Configuration for pose classification system"""
    
    def __init__(self):
        # Model Architecture
        self.num_keypoints = 17  # COCO format
        self.keypoint_dims = 2   # x, y coordinates
        self.input_dim = self.num_keypoints * self.keypoint_dims
        self.hidden_dim = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dropout = 0.1
        self.max_seq_len = 128  # Maximum sequence length
        
        # Training
        self.batch_size = 16
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.warmup_epochs = 10
        self.gradient_clip = 1.0
        
        # Data
        self.sequence_length = 64  # Frames per sequence
        self.stride = 8  # Stride for sequence extraction
        self.num_classes = 6  # Update based on your classes
        self.class_names = ["smoking", "sitting", "standing", "walking", "calling", "playing_phone"]
        
        # Paths
        self.data_dir = "data/pose_sequences"
        self.checkpoint_dir = "checkpoints/pose_classifier"
        self.log_dir = "logs/pose_classifier"
        self.output_dir = "outputs/pose_classifier"
        
        # Augmentation
        self.use_augmentation = True
        self.noise_std = 0.02
        self.temporal_jitter = 3
        self.scale_range = (0.9, 1.1)
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for pose sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class PoseTransformerEncoder(nn.Module):
    """Transformer encoder for pose sequence processing"""
    
    def __init__(self, config: PoseConfig):
        super().__init__()
        
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input pose sequences [batch_size, seq_len, num_keypoints * 2]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        if mask is not None:
            # Convert mask to format expected by transformer
            mask = mask.bool()
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps for visualization"""
        # This would require modifying the forward pass to return attention weights
        # For now, return None
        return None


class PoseSequenceDataset(Dataset):
    """Dataset for pose sequence classification"""
    
    def __init__(self, data_paths: List[Path], labels: List[int], config: PoseConfig, augment: bool = True):
        self.data_paths = data_paths
        self.labels = labels
        self.config = config
        self.augment = augment
        
        assert len(data_paths) == len(labels), "Data paths and labels must have same length"
        
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load pose sequence
        pose_data = np.load(self.data_paths[idx])
        
        # Extract keypoints and scores
        keypoints = pose_data['keypoints']  # [seq_len, num_keypoints, 2]
        scores = pose_data.get('scores', np.ones_like(keypoints[..., :1]))  # [seq_len, num_keypoints, 1]
        
        # Normalize keypoints to [-1, 1]
        if keypoints.max() > 1.0:
            keypoints = keypoints / 256.0  # Normalize by typical image size
        
        # Flatten keypoints: [seq_len, num_keypoints * 2]
        keypoints_flat = keypoints.reshape(keypoints.shape[0], -1).astype(np.float32)
        
        # Apply augmentation if enabled
        if self.augment and self.config.use_augmentation:
            keypoints_flat = self._augment_sequence(keypoints_flat)
        
        # Pad or truncate to fixed length
        keypoints_flat = self._pad_or_truncate(keypoints_flat)
        
        # Create attention mask (1 for real frames, 0 for padding)
        seq_len = min(keypoints_flat.shape[0], self.config.sequence_length)
        attention_mask = torch.ones(self.config.sequence_length)
        attention_mask[seq_len:] = 0
        
        return {
            'keypoints': torch.from_numpy(keypoints_flat),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'attention_mask': attention_mask.bool(),
            'path': str(self.data_paths[idx])
        }
    
    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply data augmentation to pose sequence"""
        # Add Gaussian noise
        noise = np.random.normal(0, self.config.noise_std, sequence.shape)
        sequence = sequence + noise
        
        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(*self.config.scale_range)
            sequence = sequence * scale
        
        # Random temporal jittering
        if np.random.rand() < 0.3:
            jitter = np.random.randint(-self.config.temporal_jitter, self.config.temporal_jitter + 1)
            if jitter != 0:
                sequence = np.roll(sequence, jitter, axis=0)
        
        return sequence
    
    def _pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to fixed length"""
        seq_len = sequence.shape[0]
        target_len = self.config.sequence_length
        
        if seq_len >= target_len:
            # Random crop if longer
            start = np.random.randint(0, seq_len - target_len + 1)
            return sequence[start:start + target_len]
        else:
            # Pad with zeros
            padding = np.zeros((target_len - seq_len, sequence.shape[1]), dtype=sequence.dtype)
            return np.concatenate([sequence, padding], axis=0)


class PoseDataPreprocessor:
    """Preprocess raw video data into pose sequences"""
    
    def __init__(self, pose_estimator, config: PoseConfig):
        self.pose_estimator = pose_estimator
        self.config = config
        self.output_dir = Path(config.data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_video(self, video_path: Path, label: int, video_id: str = None) -> List[Path]:
        """Extract pose sequences from a video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"Extracted {len(frames)} frames from {video_path}")
        
        # Extract poses for all frames
        poses = []
        for frame in tqdm(frames, desc="Extracting poses"):
            pose_result = self.pose_estimator(frame)
            if pose_result is not None:
                poses.append(pose_result)
            else:
                # Use last valid pose or zeros
                if poses:
                    poses.append(poses[-1])
                else:
                    poses.append(np.zeros((self.config.num_keypoints, 3)))  # x, y, confidence
        
        poses = np.array(poses)  # [num_frames, num_keypoints, 3]
        
        # Create sequences
        sequences = []
        for start_idx in range(0, len(poses) - self.config.sequence_length + 1, self.config.stride):
            end_idx = start_idx + self.config.sequence_length
            sequence = poses[start_idx:end_idx]
            
            # Split into keypoints and scores
            keypoints = sequence[:, :, :2]  # [seq_len, num_keypoints, 2]
            scores = sequence[:, :, 2:3]    # [seq_len, num_keypoints, 1]
            
            # Save sequence
            if video_id is None:
                video_id = video_path.stem
            
            seq_id = f"{video_id}_seq{len(sequences):04d}"
            output_path = self.output_dir / f"{seq_id}.npz"
            
            np.savez_compressed(
                output_path,
                keypoints=keypoints,
                scores=scores,
                label=label,
                video_path=str(video_path)
            )
            
            sequences.append(output_path)
        
        print(f"Created {len(sequences)} sequences from {video_path}")
        return sequences
    
    def process_directory(self, data_dir: Path, label_mapping: Dict[str, int]) -> Tuple[List[Path], List[int]]:
        """Process all videos in a directory structure"""
        all_sequences = []
        all_labels = []
        
        for class_name, label in label_mapping.items():
            class_dir = data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
            
            print(f"\nProcessing {len(video_files)} videos for class '{class_name}'")
            
            for video_file in tqdm(video_files, desc=f"Processing {class_name}"):
                sequences = self.process_video(video_file, label)
                all_sequences.extend(sequences)
                all_labels.extend([label] * len(sequences))
        
        print(f"\nTotal sequences created: {len(all_sequences)}")
        return all_sequences, all_labels


class PoseTrainer:
    """Trainer class for pose classification"""
    
    def __init__(self, config: PoseConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = PoseTransformerEncoder(config).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            keypoints = batch['keypoints'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(keypoints, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                keypoints = batch['keypoints'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(keypoints, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions for analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            # Warmup learning rate
            if epoch < self.config.warmup_epochs:
                lr_scale = (epoch + 1) / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * lr_scale
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, predictions, labels = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')
                print(f"✅ New best model saved with accuracy: {val_acc:.2f}%")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Save training plots
            if (epoch + 1) % 5 == 0:
                self.save_training_plots()
        
        print(f"\n🎉 Training complete! Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Final evaluation
        self.final_evaluation(val_loader)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_training_plots(self):
        """Save training progress plots"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.val_accuracies, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.config.output_dir, f'training_plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_path}")
    
    def final_evaluation(self, val_loader: DataLoader):
        """Final evaluation with detailed metrics"""
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        val_loss, val_acc, predictions, labels = self.validate(val_loader)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            labels, 
            predictions, 
            target_names=self.config.class_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.config.class_names,
            yticklabels=self.config.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = os.path.join(self.config.output_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
        print("="*60)


class PoseInference:
    """Inference class for pose classification"""
    
    def __init__(self, model_path: str, config: PoseConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model
        self.model = PoseTransformerEncoder(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def predict_sequence(self, keypoints: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict action from pose sequence
        
        Args:
            keypoints: Pose sequence [seq_len, num_keypoints, 2]
            
        Returns:
            action_name: Predicted action name
            confidence: Prediction confidence
            probabilities: Class probabilities
        """
        # Preprocess
        keypoints_flat = keypoints.reshape(keypoints.shape[0], -1).astype(np.float32)
        
        # Normalize
        if keypoints_flat.max() > 1.0:
            keypoints_flat = keypoints_flat / 256.0
        
        # Pad/truncate
        seq_len = min(keypoints_flat.shape[0], self.config.sequence_length)
        padded = np.zeros((self.config.sequence_length, keypoints_flat.shape[1]), dtype=np.float32)
        padded[:seq_len] = keypoints_flat[:seq_len]
        
        # Create attention mask
        attention_mask = torch.ones(self.config.sequence_length)
        attention_mask[seq_len:] = 0
        
        # Convert to tensor
        input_tensor = torch.from_numpy(padded).unsqueeze(0).to(self.device)  # [1, seq_len, dims]
        mask_tensor = attention_mask.unsqueeze(0).bool().to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor, mask_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get top prediction
        pred_idx = np.argmax(probabilities)
        action_name = self.config.class_names[pred_idx]
        confidence = float(probabilities[pred_idx])
        
        return action_name, confidence, probabilities
    
    def predict_video(self, video_path: str, pose_estimator) -> List[Dict]:
        """Predict actions for an entire video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Extract poses
        poses = []
        frame_indices = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 2 == 0:  # Process every 2nd frame for efficiency
                pose_result = pose_estimator(frame)
                if pose_result is not None:
                    poses.append(pose_result)
                    frame_indices.append(frame_idx)
                else:
                    if poses:
                        poses.append(poses[-1])
                        frame_indices.append(frame_idx)
                    else:
                        poses.append(np.zeros((self.config.num_keypoints, 2)))
                        frame_indices.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        # Create overlapping sequences
        results = []
        poses = np.array(poses)
        
        for start_idx in range(0, len(poses) - self.config.sequence_length + 1, self.config.stride):
            end_idx = start_idx + self.config.sequence_length
            sequence = poses[start_idx:end_idx]
            
            action_name, confidence, probabilities = self.predict_sequence(sequence)
            
            start_frame = frame_indices[start_idx]
            end_frame = frame_indices[end_idx - 1]
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            results.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'action': action_name,
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            })
        
        return results


def create_dummy_pose_estimator():
    """Create a dummy pose estimator for testing (replace with real one)"""
    def dummy_estimator(frame):
        # This is a placeholder - replace with actual pose estimation
        # For testing, return random poses
        h, w = frame.shape[:2]
        num_keypoints = 17
        keypoints = np.random.rand(num_keypoints, 2) * [w, h]
        return keypoints
    return dummy_estimator


def main():
    """Main training pipeline"""
    
    # Initialize configuration
    config = PoseConfig()
    print("Pose Classification Training System")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Number of classes: {config.num_classes}")
    print(f"Class names: {config.class_names}")
    print("="*60)
    
    # Create dummy pose estimator (replace with real one)
    pose_estimator = create_dummy_pose_estimator()
    
    # Create data preprocessor
    preprocessor = PoseDataPreprocessor(pose_estimator, config)
    
    # Option 1: Process videos into pose sequences
    # Uncomment this if you have raw videos to process
    """
    data_dir = Path("data/custom_actions_videos")
    label_mapping = {
        "smoking": 0,
        "sitting": 1, 
        "standing": 2,
        "walking": 3,
        "calling": 4,
        "playing_phone": 5
    }
    
    print("Processing videos into pose sequences...")
    all_sequences, all_labels = preprocessor.process_directory(data_dir, label_mapping)
    
    # Save data split info
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        all_sequences, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    # Save splits for reproducibility
    split_info = {
        'train': [(str(p), int(l)) for p, l in zip(train_sequences, train_labels)],
        'val': [(str(p), int(l)) for p, l in zip(val_sequences, val_labels)]
    }
    
    with open(os.path.join(config.data_dir, 'data_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    """
    
    # Option 2: Load existing pose sequences
    # Assuming you have already processed the data
    print("Loading existing pose sequences...")
    
    # For demo purposes, create dummy data
    # In practice, load from the data_split.json created above
    num_samples = 1000
    dummy_sequences = []
    dummy_labels = []
    
    for _ in range(num_samples):
        # Create random pose sequence
        sequence = np.random.rand(config.sequence_length, config.num_keypoints, 2).astype(np.float32)
        
        # Create temporary file
        temp_path = Path(config.data_dir) / f"temp_{len(dummy_sequences)}.npz"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        label = np.random.randint(0, config.num_classes)
        np.savez_compressed(temp_path, keypoints=sequence, scores=np.ones((config.sequence_length, config.num_keypoints, 1)))
        
        dummy_sequences.append(temp_path)
        dummy_labels.append(label)
    
    # Split data
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        dummy_sequences, dummy_labels, test_size=0.2, stratify=dummy_labels, random_state=42
    )
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    
    # Create datasets
    train_dataset = PoseSequenceDataset(train_sequences, train_labels, config, augment=True)
    val_dataset = PoseSequenceDataset(val_sequences, val_labels, config, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create trainer and start training
    trainer = PoseTrainer(config)
    trainer.train(train_loader, val_loader)
    
    # Test inference
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60)
    
    # Load best model
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    inference = PoseInference(best_model_path, config)
    
    # Test on a sample
    sample_sequence = np.random.rand(config.sequence_length, config.num_keypoints, 2)
    action, confidence, probs = inference.predict_sequence(sample_sequence)
    
    print(f"Sample prediction: {action} (confidence: {confidence:.4f})")
    print(f"All probabilities:")
    for class_name, prob in zip(config.class_names, probs):
        print(f"  {class_name}: {prob:.4f}")
    
    print("\n✅ Training and testing complete!")
    print(f"Model saved to: {config.checkpoint_dir}")
    print(f"Outputs saved to: {config.output_dir}")


if __name__ == "__main__":
    main()