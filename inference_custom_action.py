"""
Inference Script for Custom Action Recognition Model
Use your trained ST-GCN model to detect actions in new videos
DEFAULT: Saves processed video with annotations
"""

import os
import cv2
import numpy as np
import torch
import json
from mmengine.config import Config
from mmengine import init_default_scope
from mmaction.apis import init_recognizer, inference_skeleton
from mmaction.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmdet.utils import register_all_modules as register_det_modules
from mmpose.utils import register_all_modules as register_pose_modules
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

register_all_modules()

# Configuration
MODEL_CONFIG = "configs/custom_action_recognition.py"
MODEL_CHECKPOINT = "work_dirs/custom_action_recognition/best_acc_top1_epoch_12.pth"
LABELS_FILE = "data/custom_actions_processed/labels.txt"
OUTPUT_DIR = "inference_outputs"  # Where to save processed footage
NUM_FRAMES = 100
NUM_KEYPOINTS = 17

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load labels
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r") as f:
        ACTION_LABELS = [line.strip() for line in f.readlines()]
else:
    ACTION_LABELS = ["smoking", "sitting", "standing", "walking", "calling", "playing_phone"]

print(f"Loaded {len(ACTION_LABELS)} action labels: {ACTION_LABELS}")


def extract_keypoints_from_video(video_path, det_model, pose_model):
    """Extract keypoints from a single video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, None, None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames uniformly
    if total_frames < NUM_FRAMES:
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    keypoints_list = []
    keypoint_scores_list = []
    all_frames = []

    for target_idx in tqdm(frame_indices, desc=f"  Extracting keypoints"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        all_frames.append(frame.copy())  # Store original frame for saving later

        # Detect people
        init_default_scope("mmdet")
        det_res = inference_detector(det_model, frame)
        bboxes = det_res.pred_instances.bboxes.cpu().numpy()
        scores = det_res.pred_instances.scores.cpu().numpy()
        labels = det_res.pred_instances.labels.cpu().numpy()

        # Filter for persons
        person_bboxes = [
            bboxes[i] for i, l in enumerate(labels) if l == 0 and scores[i] > 0.5
        ]

        if not person_bboxes:
            keypoints = np.zeros((NUM_KEYPOINTS, 2))
            keypoint_scores = np.zeros(NUM_KEYPOINTS)
        else:
            # Use largest person
            bbox_areas = [
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in person_bboxes
            ]
            main_bbox = person_bboxes[np.argmax(bbox_areas)]

            init_default_scope("mmpose")
            pose_res = inference_topdown(pose_model, frame, [main_bbox])

            if pose_res and len(pose_res) > 0:
                result = pose_res[0]
                if (
                    hasattr(result, "pred_instances")
                    and result.pred_instances is not None
                ):
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
                    kps_scores = (
                        kps_scores[0] if len(kps_scores.shape) == 2 else kps_scores
                    )

                    if kps.shape[0] < NUM_KEYPOINTS:
                        kps = np.vstack(
                            [kps, np.zeros((NUM_KEYPOINTS - kps.shape[0], 2))]
                        )
                        kps_scores = np.concatenate(
                            [kps_scores, np.zeros(NUM_KEYPOINTS - kps_scores.shape[0])]
                        )

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
        if len(keypoints_array) > 0:
            keypoints_array = np.vstack(
                [keypoints_array, np.tile(keypoints_array[-1], (padding, 1, 1))]
            )
            keypoint_scores_array = np.vstack(
                [
                    keypoint_scores_array,
                    np.tile(keypoint_scores_array[-1], (padding, 1)),
                ]
            )
            # Also pad frames
            last_frame = all_frames[-1]
            for _ in range(padding):
                all_frames.append(last_frame.copy())
        else:
            keypoints_array = np.zeros((NUM_FRAMES, NUM_KEYPOINTS, 2), dtype=np.float32)
            keypoint_scores_array = np.zeros(
                (NUM_FRAMES, NUM_KEYPOINTS), dtype=np.float32
            )

    return keypoints_array, keypoint_scores_array, (height, width), all_frames, fps


def save_video_with_skeleton(frames, keypoints, action_name, confidence, output_path, fps=30):
    """Save video with skeleton overlay"""
    print(f"\n💾 Saving processed video to: {output_path}")
    
    # Get video dimensions
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, (frame, kp) in enumerate(tqdm(zip(frames, keypoints), desc="  Processing frames", total=len(frames))):
        # Draw keypoints
        for j, (x, y) in enumerate(kp):
            if x > 0 and y > 0:  # Only draw valid keypoints
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw skeleton connections (COCO format)
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for connection in skeleton_connections:
            idx1, idx2 = connection
            if idx1 < len(kp) and idx2 < len(kp):
                x1, y1 = kp[idx1]
                x2, y2 = kp[idx2]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # Draw label
        label_text = f"{action_name} ({confidence:.1f}%)"
        cv2.rectangle(frame, (10, 10), (350, 55), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        
        out.write(frame)

    out.release()
    print(f"✅ Video saved successfully!")


def main():
    print("=" * 60)
    print("Custom Action Recognition - Inference")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"\n❌ Error: Model checkpoint not found: {MODEL_CHECKPOINT}")
        print("Please train the model first: python train_custom_model.py")
        return

    # Load model
    print(f"\nLoading model from {MODEL_CHECKPOINT}")
    cfg = Config.fromfile(MODEL_CONFIG)
    model = init_recognizer(cfg, MODEL_CHECKPOINT, device="cuda:0")
    model.eval()
    print("✓ Model loaded successfully")

    # Initialize pose models
    print("\nInitializing pose models...")
    register_det_modules()
    register_pose_modules()

    det_config = "mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
    det_ckpt = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    init_default_scope("mmdet")
    det_model = init_detector(det_config, det_ckpt, device="cuda:0")

    pose_config = "configs/mmpose/rtmpose-t_8xb256-420e_coco-256x192.py"
    pose_ckpt = "checkpoints/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth"
    init_default_scope("mmpose")
    pose_model = init_pose_model(pose_config, pose_ckpt, device="cuda:0")
    print("✓ Pose models initialized")

    # Get video path
    print("\n" + "=" * 60)
    print("Enter the path to a video file to analyze:")
    print("(Example: footage/my_video.mp4)")
    video_path = input("Video path: ").strip()

    if not video_path:
        print("\n❌ Error: No video path provided")
        return

    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video not found: {video_path}")
        return

    print(f"\nProcessing video: {video_path}")

    # Extract keypoints and frames
    keypoints, keypoint_scores, img_shape, frames, fps = extract_keypoints_from_video(
        video_path, det_model, pose_model
    )

    if keypoints is None:
        print("\n❌ Error: Failed to extract keypoints")
        return

    print(f"\n✓ Extracted keypoints: shape={keypoints.shape}")

    # Prepare pose_results in the format expected by inference_skeleton
    pose_results = []
    for i in range(len(keypoints)):
        frame_result = {
            'keypoints': keypoints[i].reshape(1, NUM_KEYPOINTS, 2),  # (1, 17, 2)
            'keypoint_scores': keypoint_scores[i].reshape(1, NUM_KEYPOINTS)  # (1, 17)
        }
        pose_results.append(frame_result)

    print(f"✓ Prepared pose_results: {len(pose_results)} frames")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        result = inference_skeleton(model, pose_results, img_shape)

    # Get prediction
    pred_scores = result.pred_score.cpu().numpy()
    pred_label = np.argmax(pred_scores)
    confidence = pred_scores[pred_label] * 100

    action_name = ACTION_LABELS[pred_label]

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nPredicted Action: {action_name}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nAll probabilities:")
    for i, (label, score) in enumerate(zip(ACTION_LABELS, pred_scores)):
        marker = "  <-- PREDICTED" if i == pred_label else ""
        print(f"  {label}: {score * 100:.2f}%{marker}")
    print("=" * 60)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    output_video_path = os.path.join(OUTPUT_DIR, f"{video_name}_{action_name}_{timestamp}.mp4")
    output_json_path = os.path.join(OUTPUT_DIR, f"{video_name}_{action_name}_{timestamp}.json")

    # AUTOMATICALLY save video with annotations (DEFAULT BEHAVIOR)
    save_video_with_skeleton(frames, keypoints, action_name, confidence, output_video_path, fps)
    
    # Optionally save results to JSON
    save_json = input("\n💾 Save prediction results to JSON file? (y/n, default=n): ").strip().lower()
    
    if save_json == 'y':
        results = {
            "video_path": video_path,
            "predicted_action": action_name,
            "confidence": float(confidence),
            "all_probabilities": {
                label: float(score * 100) 
                for label, score in zip(ACTION_LABELS, pred_scores)
            },
            "timestamp": datetime.now().isoformat(),
            "model_checkpoint": MODEL_CHECKPOINT,
            "keypoints_shape": keypoints.shape
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_json_path}")

    # Optional: Live visualization
    visualize = input("\n👁️  Show live visualization? (y/n, default=n): ").strip().lower()
    if visualize == 'y':
        print("\nVisualizing... (Press 'q' to quit)")
        for frame, kp in zip(frames, keypoints):
            vis_frame = frame.copy()
            for x, y in kp:
                if x > 0 and y > 0:
                    cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            label_text = f"{action_name} ({confidence:.1f}%)"
            cv2.rectangle(vis_frame, (10, 10), (350, 55), (0, 0, 0), -1)
            cv2.putText(vis_frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            
            cv2.imshow('Action Recognition', vis_frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("✅ Inference complete!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎬 Video saved: {output_video_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
