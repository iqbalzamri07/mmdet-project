from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules as register_det_modules
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmengine import init_default_scope
from mmpose.utils import register_all_modules as register_pose_modules
import cv2
import os
import glob
import numpy as np

# 1. Register all modules for both libraries
register_det_modules()
register_pose_modules()

# 2. MMDetection Setup - USE LOCAL CHECKPOINT
det_config_file = 'configs/mmdet/faster-rcnn_r50_fpn_1x_coco.py'
# CHANGE THIS: Use local checkpoint file instead of URL
det_checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Check if local checkpoint exists, if not, download it
if not os.path.exists(det_checkpoint_file):
    print(f"Checkpoint not found at {det_checkpoint_file}, downloading...")
    os.makedirs('checkpoints', exist_ok=True)
    os.system(f"wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -P checkpoints/")

det_model = init_detector(det_config_file, det_checkpoint_file, device='cuda:0')
det_classes = det_model.dataset_meta['classes']

# 3. MMPose Setup
pose_config_file = 'configs/mmpose/rtmpose-t_8xb256-420e_coco-256x192.py'

# DYNAMIC SEARCH: This finds any file in 'checkpoints' that starts with 'rtmpose-t'
matches = glob.glob('checkpoints/rtmpose-*.pth')

if not matches:
    print("Error: Could not find an 'rtmpose-t' checkpoint file in 'checkpoints/'")
    print("Please ensure you ran 'mim download mmpose --config rtmpose-t_8xb256-420e_coco-256x192 --dest checkpoints'")
    exit()

# Automatically use the first one found
pose_checkpoint_file = matches[0]
print(f"Found and using: {pose_checkpoint_file}")

pose_model = init_pose_model(pose_config_file, pose_checkpoint_file, device='cuda:0')

# Rest of your code remains the same...
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

# Paths
input_video_path = 'footage/cam1_00_20260415010152.mp4'
output_video_path = 'process_video/cam1_00_20260415010152.mp4'

if not os.path.exists(input_video_path):
    print(f"Error: Could not find '{input_video_path}'.")
    exit()

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing '{input_video_path}' with Detection + Pose Estimation...")
print(f"Total frames: {total_frames}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    init_default_scope('mmdet')  # Set scope to mmdet before detection
    # --- STEP 1: DETECTION ---
    det_result = inference_detector(det_model, frame)
    
    bboxes = det_result.pred_instances.bboxes.cpu().numpy()
    scores = det_result.pred_instances.scores.cpu().numpy()
    labels = det_result.pred_instances.labels.cpu().numpy()

    person_bboxes = []
    
    # --- STEP 2: DRAW PHONES & GATHER PERSONS ---
    for bbox, score, label in zip(bboxes, scores, labels):
        class_name = det_classes[label]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw cell phones immediately
        if class_name == 'cell phone' and score > 0.4:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        # Save person bboxes for pose estimation
        elif class_name == 'person' and score > 0.4:
            person_bboxes.append([x1, y1, x2, y2])
    init_default_scope('mmpose')
    # --- STEP 3: POSE ESTIMATION ---
    if len(person_bboxes) > 0:
        # Send all person bounding boxes to MMPose at once
        pose_results = inference_topdown(pose_model, frame, person_bboxes)
        
        # Handle pose results
        if pose_results and len(pose_results) > 0:
            for idx, result in enumerate(pose_results):
                if hasattr(result, 'pred_instances') and result.pred_instances is not None:
                    pred_instances = result.pred_instances
                    
                    if pred_instances.keypoints is not None:
                        keypoints = pred_instances.keypoints
                        # Convert to numpy if it's a tensor
                        if hasattr(keypoints, 'cpu'):
                            keypoints = keypoints.cpu().numpy()
                        else:
                            keypoints = np.array(keypoints)
                        
                        keypoint_scores = pred_instances.keypoint_scores
                        # Convert to numpy if it's a tensor
                        if hasattr(keypoint_scores, 'cpu'):
                            keypoint_scores = keypoint_scores.cpu().numpy()
                        else:
                            keypoint_scores = np.array(keypoint_scores)
                        
                        # Draw keypoints for each person
                        for i in range(len(keypoints)):
                            kps = keypoints[i]
                            kps_scores = keypoint_scores[i]
                            
                            # Draw keypoints only (no skeleton lines)
                            for j in range(len(kps)):
                                if j < len(kps_scores) and kps_scores[j] > 0.3:
                                    pt = (int(kps[j, 0]), int(kps[j, 1]))
                                    if 0 <= pt[0] < width and 0 <= pt[1] < height:
                                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
                            
                            # Draw bounding box around the person
                            if idx < len(person_bboxes):
                                px1, py1, px2, py2 = map(int, person_bboxes[idx])
                                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

    # --- STEP 4: SAVE ---
    out.write(frame)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count}/{total_frames} frames...")

cap.release()
out.release()
print(f"Done! Saved processed video to '{output_video_path}'")