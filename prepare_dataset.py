"""
Data Preparation Script for Custom Action Recognition

This script extracts pose keypoints from your videos and saves them
in the format required for ST-GCN training.
"""

import os
import cv2
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm

# MMLab Imports
from mmengine import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.utils import register_all_modules as register_pose_modules
from mmdet.utils import register_all_modules as register_det_modules

# Configuration
DATASET_ROOT = "data/custom_actions"
OUTPUT_ROOT = "data/custom_actions_processed"
NUM_FRAMES = 100  # ST-GCN expects 100 frames
NUM_KEYPOINTS = 17  # COCO format

# Action labels
ACTION_LABELS = ["smoking", "sitting", "standing", "walking", "calling", "playing_phone"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ACTION_LABELS)}

def init_models():
    """Initialize detection and pose models"""
    print("Initializing models...")
    
    register_det_modules()
    register_pose_modules()
    
    # Detection model
    det_config = "mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
    det_ckpt = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    init_default_scope("mmdet")
    det_model = init_detector(det_config, det_ckpt, device="cuda:0")
    
    # Pose model
    pose_config = "configs/mmpose/rtmpose-t_8xb256-420e_coco-256x192.py"
    pose_ckpt = "checkpoints/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth"
    init_default_scope("mmpose")
    pose_model = init_pose_model(pose_config, pose_ckpt, device="cuda:0")
    
    return det_model, pose_model

def extract_keypoints_from_video(video_path, det_model, pose_model):
    """Extract keypoints from a single video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample frames uniformly
    if total_frames < NUM_FRAMES:
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    keypoints_list = []
    keypoint_scores_list = []
    
    for target_idx in tqdm(frame_indices, desc=f"  Processing {Path(video_path).name}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect people
        init_default_scope("mmdet")
        det_res = inference_detector(det_model, frame)
        bboxes = det_res.pred_instances.bboxes.cpu().numpy()
        scores = det_res.pred_instances.scores.cpu().numpy()
        labels = det_res.pred_instances.labels.cpu().numpy()
        
        # Filter for persons
        person_bboxes = [bboxes[i] for i, l in enumerate(labels)
                        if l == 0 and scores[i] > 0.5]
        
        if not person_bboxes:
            keypoints = np.zeros((NUM_KEYPOINTS, 2))
            keypoint_scores = np.zeros(NUM_KEYPOINTS)
        else:
            # Use largest person
            bbox_areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in person_bboxes]
            main_bbox = person_bboxes[np.argmax(bbox_areas)]
            
            init_default_scope("mmpose")
            pose_res = inference_topdown(pose_model, frame, [main_bbox])
            
            if pose_res and len(pose_res) > 0:
                result = pose_res[0]
                if hasattr(result, "pred_instances") and result.pred_instances is not None:
                    kps = result.pred_instances.keypoints
                    kps_scores = result.pred_instances.keypoint_scores
                    
                    if hasattr(kps, "cpu"):
                        kps = kps.cpu().numpy()
                    else:
                        kps = np.array(kps)
                        
                    if hasattr(kps_scores, "cpu"):
                        kps_scores = kps_scores.cpu().numpy()
                    else:
                        kps_scores = np.array(kps_scores)
                    
                    kps = kps[0] if len(kps.shape) == 3 else kps
                    kps_scores = kps_scores[0] if len(kps_scores.shape) == 2 else kps_scores
                    
                    if kps.shape[0] < NUM_KEYPOINTS:
                        kps = np.vstack([kps, np.zeros((NUM_KEYPOINTS - kps.shape[0], 2))])
                        kps_scores = np.concatenate([kps_scores, np.zeros(NUM_KEYPOINTS - kps_scores.shape[0])])
                    
                    keypoints = kps[:, :2]
                    keypoint_scores = kps_scores
                else:
                    keypoints = np.zeros((NUM_KEYPOINTS, 2))
                    keypoint_scores = np.zeros(NUM_KEYPOINTS)
            else:
                keypoints = np.zeros((NUM_KEYPOINTS, 2))
                keypoint_scores = np.zeros(NUM_KEYPOINTS)
        
        keypoints_list.append(keypoints)
        keypoint_scores_list.append(keypoint_scores)
    
    cap.release()
    
    keypoints_array = np.array(keypoints_list, dtype=np.float32)
    keypoint_scores_array = np.array(keypoint_scores_list, dtype=np.float32)
    
    if len(keypoints_array) < NUM_FRAMES:
        padding = NUM_FRAMES - len(keypoints_array)
        keypoints_array = np.vstack([keypoints_array, np.tile(keypoints_array[-1], (padding, 1, 1))])
        keypoint_scores_array = np.vstack([keypoint_scores_array, np.tile(keypoint_scores_array[-1], (padding, 1))])
    
    return keypoints_array, (height, width)

def process_dataset(split="train"):
    """Process all videos in a split"""
    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"{'='*60}")
    
    det_model, pose_model = init_models()
    
    all_keypoints = []
    all_keypoint_scores = []
    all_labels = []
    all_img_shapes = []
    video_names = []
    
    for label in ACTION_LABELS:
        label_dir = os.path.join(DATASET_ROOT, split, label)
        
        if not os.path.exists(label_dir):
            print(f"Warning: {label_dir} does not exist, skipping...")
            continue
        
        video_files = [f for f in os.listdir(label_dir)
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"\nFound {len(video_files)} videos for '{label}'")
        
        for video_file in video_files:
            video_path = os.path.join(label_dir, video_file)
            
            try:
                keypoints, img_shape = extract_keypoints_from_video(video_path, det_model, pose_model)
                
                if keypoints is not None:
                    all_keypoints.append(keypoints)
                    all_keypoint_scores.append(keypoint_scores)
                    all_labels.append(LABEL_TO_IDX[label])
                    all_img_shapes.append(img_shape)
                    video_names.append(video_file)
                    print(f"  ✓ {video_file}: shape={keypoints.shape}")
                else:
                    print(f"  ✗ {video_file}: failed to process")
                    
            except Exception as e:
                print(f"  ✗ {video_file}: error - {e}")
    
    if len(all_keypoints) == 0:
        print(f"\nNo videos processed for {split} split!")
        return None
    
    # Save data
    output_file = os.path.join(OUTPUT_ROOT, f"{split}_data.pkl")
    
    data = {
        'keypoint': np.array(all_keypoints),
        'keypoint_score': np.array(all_keypoint_scores),
        'label': np.array(all_labels),
        'img_shape': all_img_shapes,
        'video_names': video_names,
        'num_frames': NUM_FRAMES,
        'num_keypoints': NUM_KEYPOINTS,
        'total_videos': len(all_keypoints)
    }
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    # Save label mapping
    json_file = os.path.join(OUTPUT_ROOT, f"{split}_data.json")
    json_data = {
        'total_videos': len(all_keypoints),
        'num_frames': NUM_FRAMES,
        'num_keypoints': NUM_KEYPOINTS,
        'labels': [ACTION_LABELS[l] for l in all_labels],
        'video_names': video_names,
        'label_to_idx': LABEL_TO_IDX
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n✓ Saved {len(all_keypoints)} videos to {output_file}")
    print(f"  Label distribution:")
    for i, label in enumerate(ACTION_LABELS):
        count = sum(1 for l in all_labels if l == i)
        print(f"    {label}: {count} videos")
    
    return data

def main():
    """Main function"""
    print("="*60)
    print("Custom Action Recognition - Data Preparation")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_ROOT):
        print(f"\nError: Dataset directory {DATASET_ROOT} does not exist!")
        print("Please create the dataset structure first:")
        print("  data/custom_actions/train/{smoking,sitting,standing,walking,calling,playing_phone}")
        print("  data/custom_actions/val/{smoking,sitting,standing,walking,calling,playing_phone}")
        return
    
    # Process train split
    train_data = process_dataset("train")
    
    # Process val split
    val_data = process_dataset("val")
    
    # Create labels file
    label_file = os.path.join(OUTPUT_ROOT, "labels.txt")
    with open(label_file, 'w') as f:
        for label in ACTION_LABELS:
            f.write(f"{label}\n")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_ROOT}")
    print(f"Train samples: {train_data['total_videos'] if train_data else 0}")
    print(f"Val samples: {val_data['total_videos'] if val_data else 0}")
    print(f"\nNext steps:")
    print("1. Create training config: configs/custom_action_recognition.py")
    print("2. Run training: python -m mmaction.train configs/custom_action_recognition.py")
    print("="*60)

if __name__ == "__main__":
    main()
