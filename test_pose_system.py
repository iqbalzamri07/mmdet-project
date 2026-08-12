#!/usr/bin/env python3
"""
Test script for pose classification system
Verifies that all components work correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_pose_classifier import (
    PoseConfig, PoseTransformerEncoder, PoseSequenceDataset,
    PositionalEncoding
)


def test_configuration():
    """Test configuration loading"""
    print("Testing configuration...")
    
    config = PoseConfig()
    
    assert config.num_keypoints == 17, "Number of keypoints should be 17"
    assert config.num_classes == 6, "Number of classes should be 6"
    assert len(config.class_names) == 6, "Should have 6 class names"
    assert config.sequence_length == 64, "Sequence length should be 64"
    
    print("✓ Configuration test passed")
    return True


def test_positional_encoding():
    """Test positional encoding"""
    print("Testing positional encoding...")
    
    d_model = 256
    max_len = 128
    batch_size = 2
    seq_len = 32
    
    pos_encoding = PositionalEncoding(d_model, max_len)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_encoding(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert not torch.allclose(output, x), "Positional encoding should modify input"
    
    print("✓ Positional encoding test passed")
    return True


def test_model_architecture():
    """Test model architecture"""
    print("Testing model architecture...")
    
    config = PoseConfig()
    model = PoseTransformerEncoder(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = config.sequence_length
    input_dim = config.num_keypoints * config.keypoint_dims
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    logits = model(x, mask)
    
    expected_shape = (batch_size, config.num_classes)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    
    # Test without mask
    logits_no_mask = model(x)
    assert logits_no_mask.shape == expected_shape, "Output shape should be same without mask"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    assert total_params > 0, "Model should have parameters"
    assert total_params < 50_000_000, "Model should have less than 50M parameters for efficiency"
    
    print("✓ Model architecture test passed")
    return True


def test_dataset_creation():
    """Test dataset creation and loading"""
    print("Testing dataset creation...")
    
    config = PoseConfig()
    
    # Create dummy data files
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create temporary data files
        data_paths = []
        labels = []
        
        for i in range(10):
            # Create dummy pose sequence
            sequence = np.random.rand(config.sequence_length, config.num_keypoints, 2).astype(np.float32)
            scores = np.ones((config.sequence_length, config.num_keypoints, 1), dtype=np.float32)
            
            # Save to temporary file
            temp_path = temp_dir / f"test_{i}.npz"
            np.savez_compressed(temp_path, keypoints=sequence, scores=scores)
            
            data_paths.append(temp_path)
            labels.append(i % config.num_classes)
        
        # Create dataset
        dataset = PoseSequenceDataset(data_paths, labels, config, augment=False)
        
        assert len(dataset) == 10, "Dataset should have 10 samples"
        
        # Test data loading
        sample = dataset[0]
        
        assert 'keypoints' in sample, "Sample should contain 'keypoints'"
        assert 'labels' in sample, "Sample should contain 'labels'"
        assert 'attention_mask' in sample, "Sample should contain 'attention_mask'"
        
        assert sample['keypoints'].shape == (config.sequence_length, config.input_dim), \
            f"Keypoints shape should be ({config.sequence_length}, {config.input_dim})"
        assert sample['attention_mask'].shape == (config.sequence_length,), \
            f"Attention mask shape should be ({config.sequence_length},)"
        
        # Test data loader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        assert batch['keypoints'].shape[0] == 4, "Batch size should be 4"
        assert batch['keypoints'].shape[1] == config.sequence_length, \
            f"Sequence length should be {config.sequence_length}"
        
        print("✓ Dataset creation test passed")
        return True
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


def test_data_augmentation():
    """Test data augmentation"""
    print("Testing data augmentation...")
    
    config = PoseConfig()
    config.use_augmentation = True
    
    # Create temporary data
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create a fixed sequence
        sequence = np.random.rand(config.sequence_length, config.num_keypoints, 2).astype(np.float32)
        scores = np.ones((config.sequence_length, config.num_keypoints, 1), dtype=np.float32)
        
        temp_path = temp_dir / "test_aug.npz"
        np.savez_compressed(temp_path, keypoints=sequence, scores=scores)
        
        # Create datasets with and without augmentation
        dataset_no_aug = PoseSequenceDataset([temp_path], [0], config, augment=False)
        dataset_aug = PoseSequenceDataset([temp_path], [0], config, augment=True)
        
        sample_no_aug = dataset_no_aug[0]['keypoints']
        sample_aug = dataset_aug[0]['keypoints']
        
        # With augmentation, the samples should be different
        # (due to random operations)
        # We'll test this multiple times since augmentation is random
        differences = 0
        for _ in range(5):
            sample_aug_1 = dataset_aug[0]['keypoints']
            sample_aug_2 = dataset_aug[0]['keypoints']
            
            if not np.allclose(sample_aug_1, sample_aug_2):
                differences += 1
        
        # At least some samples should be different due to augmentation
        assert differences > 0, "Augmentation should create variations"
        
        print("✓ Data augmentation test passed")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_inference_components():
    """Test inference components"""
    print("Testing inference components...")
    
    config = PoseConfig()
    
    # Create a simple model
    model = PoseTransformerEncoder(config)
    model.eval()
    
    # Test with dummy sequence
    sequence = np.random.rand(config.sequence_length, config.num_keypoints, 2).astype(np.float32)
    
    # Simulate inference process
    sequence_flat = sequence.reshape(sequence.shape[0], -1).astype(np.float32)
    
    # Normalize
    if sequence_flat.max() > 1.0:
        sequence_flat = sequence_flat / 256.0
    
    # Pad/truncate
    seq_len = min(sequence_flat.shape[0], config.sequence_length)
    padded = np.zeros((config.sequence_length, sequence_flat.shape[1]), dtype=np.float32)
    padded[:seq_len] = sequence_flat[:seq_len]
    
    # Convert to tensor
    input_tensor = torch.from_numpy(padded).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Check output
    assert probabilities.shape == (config.num_classes,), \
        f"Probabilities should have shape ({config.num_classes},)"
    assert np.allclose(probabilities.sum(), 1.0, atol=1e-5), \
        "Probabilities should sum to 1"
    
    # Get prediction
    pred_idx = np.argmax(probabilities)
    assert 0 <= pred_idx < config.num_classes, "Prediction should be valid class index"
    
    print("✓ Inference components test passed")
    return True


def test_gpu_availability():
    """Test GPU availability and compatibility"""
    print("Testing GPU availability...")
    
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Current CUDA device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        
        # Test tensor operations on GPU
        test_tensor = torch.randn(100, 100).cuda()
        result = test_tensor @ test_tensor.T
        
        assert result.device.type == 'cuda', "Result should be on CUDA"
        
        print("✓ GPU test passed")
    else:
        print("  ⚠ CUDA not available, using CPU")
        print("✓ GPU test skipped (CPU mode)")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("POSE CLASSIFICATION SYSTEM TESTS")
    print("="*60)
    print()
    
    tests = [
        ("Configuration", test_configuration),
        ("Positional Encoding", test_positional_encoding),
        ("Model Architecture", test_model_architecture),
        ("Dataset Creation", test_dataset_creation),
        ("Data Augmentation", test_data_augmentation),
        ("Inference Components", test_inference_components),
        ("GPU Availability", test_gpu_availability),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"✗ {test_name} test failed: {str(e)}")
            results.append((test_name, False, str(e)))
    
    print()
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print()
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)