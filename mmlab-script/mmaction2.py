import cv2
import os
import torch
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from mmcv.transforms import Compose

# MMLab Imports
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules as register_det_modules
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.utils import register_all_modules as register_pose_modules
from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules as register_action_modules
from mmengine import init_default_scope
from mmengine.dataset import pseudo_collate

# 1. Register ALL modules
register_det_modules()
register_pose_modules()
register_action_modules()

# 2. Setup Paths & Models
det_config = "mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
det_ckpt = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
init_default_scope("mmdet")
det_model = init_detector(det_config, det_ckpt, device="cuda:0")
det_classes = det_model.dataset_meta["classes"]

pose_config = "configs/mmpose/rtmpose-t_8xb256-420e_coco-256x192.py"
pose_ckpt = "checkpoints/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth"
init_default_scope("mmpose")
pose_model = init_pose_model(pose_config, pose_ckpt, device="cuda:0")

action_config = (
    "configs/mmaction/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py"
)
action_ckpt = "checkpoints/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-99c60e2d.pth"
init_default_scope("mmaction")
action_model = init_recognizer(action_config, action_ckpt, device="cuda:0")

# CRITICAL: Build the test pipeline manually
mmaction_pipeline = Compose(action_model.cfg.test_pipeline)

# Load Labels
with open("labels/label_map_k710.txt", "r") as f:
    action_labels = [line.strip() for line in f.readlines()]

# 3. Skeleton connections for COCO (17 keypoints)
SKELETON_DRAW = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]

PERSON_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
]


# 4. Person Tracker Class
class PersonTracker:
    def __init__(self, max_age=30):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age

    def iou(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, detections):
        detections = np.array(detections)
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["frames_since_last_seen"] += 1
                if self.tracks[track_id]["frames_since_last_seen"] > self.max_age:
                    del self.tracks[track_id]
            return self.tracks
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = {
                    "bbox": det,
                    "frames_since_last_seen": 0,
                    "action_buffer": deque(maxlen=100),
                    "last_action": None,
                    "last_confidence": None,
                }
                self.next_id += 1
            return self.tracks
        track_ids = list(self.tracks.keys())
        cost_matrix = np.ones((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self.iou(self.tracks[track_id]["bbox"], det)
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        matched_dets = set()
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < 0.7:
                track_id = track_ids[track_idx]
                self.tracks[track_id]["bbox"] = detections[det_idx]
                self.tracks[track_id]["frames_since_last_seen"] = 0
                matched_dets.add(det_idx)
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks[self.next_id] = {
                    "bbox": det,
                    "frames_since_last_seen": 0,
                    "action_buffer": deque(maxlen=100),
                    "last_action": None,
                    "last_confidence": None,
                }
                self.next_id += 1
        for i, track_id in enumerate(track_ids):
            if i not in track_indices:
                self.tracks[track_id]["frames_since_last_seen"] += 1
                if self.tracks[track_id]["frames_since_last_seen"] > self.max_age:
                    del self.tracks[track_id]
        return self.tracks


# 5. Drawing functions
def draw_skeleton(frame, keypoints, color):
    keypoints = keypoints.astype(int)
    for connection in SKELETON_DRAW:
        idx1, idx2 = connection
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            pt1 = tuple(keypoints[idx1][:2])
            pt2 = tuple(keypoints[idx2][:2])
            if 0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0]:
                if 0 <= pt2[0] < frame.shape[1] and 0 <= pt2[1] < frame.shape[0]:
                    cv2.line(frame, pt1, pt2, color, 2)
    for i, kp in enumerate(keypoints):
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x, y), 4, color, -1)


def draw_bbox_with_action(frame, bbox, person_id, action_name, confidence, color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    id_label = f"Person {person_id}"
    (w_id, h_id), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + w_id, y1), color, -1)
    cv2.putText(
        frame, id_label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    if action_name:
        action_label = f"Action: {action_name}"
        if confidence:
            action_label += f" ({confidence:.2f})"
        (w_act, h_act), _ = cv2.getTextSize(
            action_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(frame, (x1, y2), (x1 + w_act, y2 + 20), color, -1)
        cv2.putText(
            frame,
            action_label,
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )


# 6. Initialize tracker and video
tracker = PersonTracker(max_age=30)
input_video = "footage/cam1_00_20260415010152.mp4"
output_video = "process_video/output_final.mp4"
cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(
    output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

print(f"Processing video: {input_video}")
print(f"Total frames: {total_frames}")
print("=" * 60)

frame_count = 0
window_size = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    init_default_scope("mmdet")
    det_res = inference_detector(det_model, frame)
    bboxes = det_res.pred_instances.bboxes.cpu().numpy()
    scores = det_res.pred_instances.scores.cpu().numpy()
    labels = det_res.pred_instances.labels.cpu().numpy()

    person_bboxes = [
        bboxes[i]
        for i, l in enumerate(labels)
        if det_classes[l] == "person" and scores[i] > 0.5
    ]

    if not person_bboxes:
        tracker.update([])
        out.write(frame)
        frame_count += 1
        continue

    tracks = tracker.update(person_bboxes)

    init_default_scope("mmpose")
    pose_res = inference_topdown(pose_model, frame, person_bboxes)

    pose_by_track_id = {}
    if pose_res:
        for i, result in enumerate(pose_res):
            if hasattr(result, "pred_instances") and result.pred_instances is not None:
                bbox = person_bboxes[i]
                best_match_id, best_iou = None, 0.0
                for track_id, track_data in tracks.items():
                    iou = tracker.iou(bbox, track_data["bbox"])
                    if iou > best_iou:
                        best_iou, best_match_id = iou, track_id

                if best_match_id is not None and best_iou > 0.5:
                    # --- SAFE EXTRACTION (Handles both Tensors and NumPy arrays) ---
                    keypoints = result.pred_instances.keypoints
                    if hasattr(keypoints, "cpu"):
                        keypoints = keypoints.cpu().numpy()
                    else:
                        keypoints = np.array(keypoints)

                    kps_scores = result.pred_instances.keypoint_scores
                    if hasattr(kps_scores, "cpu"):
                        kps_scores = kps_scores.cpu().numpy()
                    else:
                        kps_scores = np.array(kps_scores)
                    # ----------------------------------------------------------------

                    kps = keypoints[0] if len(keypoints.shape) == 3 else keypoints
                    kps_scores = (
                        kps_scores[0] if len(kps_scores.shape) == 2 else kps_scores
                    )
                    pose_by_track_id[best_match_id] = {
                        "keypoints": kps,
                        "scores": kps_scores,
                    }

    for track_id, track_data in tracks.items():
        color = PERSON_COLORS[track_id % len(PERSON_COLORS)]
        bbox = track_data["bbox"]
        action_buffer = track_data["action_buffer"]
        action_name = track_data.get("last_action", None)
        action_confidence = track_data.get("last_confidence", None)

        pose_data = pose_by_track_id.get(track_id, None)

        if pose_data:
            kps = pose_data["keypoints"]
            kps_scores = pose_data["scores"]
            draw_skeleton(frame, kps, color)

            kps_filtered = kps.copy()
            kps_scores_filtered = kps_scores.copy()
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            for j in range(len(kps_filtered)):
                if kps_scores[j] < 0.3:
                    kps_filtered[j, 0] = bbox_center_x
                    kps_filtered[j, 1] = bbox_center_y
                    kps_scores_filtered[j] = 0.0

            kps_pixel = kps_filtered[:, :2]

            num_kpts = kps_pixel.shape[0]
            if num_kpts < 17:
                kps_pixel = np.vstack(
                    [kps_pixel, np.tile(kps_pixel[-1], (17 - num_kpts, 1))]
                )
                kps_scores_filtered = np.concatenate(
                    [kps_scores_filtered, np.tile([0.0], (17 - num_kpts))]
                )
            elif num_kpts > 17:
                kps_pixel = kps_pixel[:17]
                kps_scores_filtered = kps_scores_filtered[:17]

            action_buffer.append((kps_pixel, kps_scores_filtered))

        if len(action_buffer) == window_size:
            try:
                all_keypoints = np.array(
                    [item[0] for item in action_buffer], dtype=np.float32
                )  # [100, 17, 2]
                all_scores = np.array(
                    [item[1] for item in action_buffer], dtype=np.float32
                )  # [100, 17]
                stgcn_input = all_keypoints[np.newaxis, ...]  # [1, 100, 17, 2]

                pipeline_input = {
                    "keypoint": stgcn_input,
                    "keypoint_score": all_scores[np.newaxis, ...],
                    "total_frames": window_size,
                    "img_shape": (height, width),
                    "start_index": 0,
                    "modality": "Pose",
                }

                # RUN PIPELINE MANUALLY (This calculates bone_motion automatically!)
                pipeline_output = mmaction_pipeline(pipeline_input)

                # Use pseudo_collate to properly format data for test_step
                test_data = pseudo_collate([pipeline_output])

                # PASS DIRECTLY TO TEST STEP
                action_result = action_model.test_step(test_data)[0]

                if hasattr(action_result, "pred_score"):
                    pred_scores = action_result.pred_score.detach().cpu().numpy()
                else:
                    pred_scores = action_result.pred_score.item.detach().cpu().numpy()

                label_idx = np.argmax(pred_scores)
                action_name = action_labels[label_idx]
                action_confidence = float(pred_scores[label_idx])

                track_data["last_action"] = action_name
                track_data["last_confidence"] = action_confidence

                top3_idx = np.argsort(pred_scores)[-3:][::-1]
                top3_actions = [(action_labels[i], pred_scores[i]) for i in top3_idx]
                print(
                    f"  Person {track_id}: {action_name} ({action_confidence:.3f}) | Top 3: {top3_actions}"
                )

            except Exception as e:
                print(f"Action recognition error for person {track_id}: {e}")
                import traceback

                traceback.print_exc()

        draw_bbox_with_action(
            frame, bbox, track_id, action_name, action_confidence, color
        )

    out.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count}/{total_frames} frames...")

cap.release()
out.release()
print(f"Done! Video saved to: {output_video}")
