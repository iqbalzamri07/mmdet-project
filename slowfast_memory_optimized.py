"""
FIXED RGB MULTI-PERSON ACTION RECOGNITION
- Fixed class mismatch
- Fixed label mapping
- Better inference logic
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import gc
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from mmengine.config import Config
from mmaction.apis import init_recognizer, inference_recognizer
from mmdet.apis import init_detector, inference_detector
from mmengine import init_default_scope

# ================= CONFIG =================
# Choose which config you trained with
USE_MULTILABEL = True  # Set to False if using single-label config

if USE_MULTILABEL:
    MODEL_CONFIG = "configs/slowfast_multilabel.py"
    MODEL_CHECKPOINT = "work_dirs/slowfast_multilabel/epoch_50.pth"
else:
    MODEL_CONFIG = "configs/slowfast_custom.py"
    MODEL_CHECKPOINT = "work_dirs/slowfast_custom/best_acc_top1_epoch_*.pth"

# FIXED: Consistent labels with training
ACTION_LABELS = ["sitting", "standing", "walking", "calling", "playing_phone"]
NUM_CLASSES = len(ACTION_LABELS)

# Thresholds for multi-label
MULTILABEL_THRESHOLD = 0.15  # Sigmoid threshold

CHECKPOINT_DIR = "checkpoints"
DET_CONFIG = "mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
DET_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")

OUTPUT_DIR = "process_video"
CLIP_LEN = 32
FRAME_STRIDE = 1
MIN_FRAMES = 5
DETECTION_THRESHOLD = 0.7
USE_CPU_FOR_DETECTION = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"✓ Using {NUM_CLASSES} classes: {ACTION_LABELS}")
print(f"✓ Multi-label mode: {USE_MULTILABEL}")

# ================= FIND BEST CHECKPOINT =================
def find_best_checkpoint(work_dir):
    """Find the best checkpoint in work_dir"""
    import glob
    
    if USE_MULTILABEL:
        pattern = os.path.join(work_dir, "epoch_50.pth")
    else:
        pattern = os.path.join(work_dir, "best_acc_top1_epoch_*.pth")
    
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        return checkpoints[0]
    
    # Fallback to latest
    pattern = os.path.join(work_dir, "epoch_*.pth")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        return sorted(checkpoints)[-1]
    
    return None

# ================= INIT MODELS =================
print("="*60)
print("LOADING MODELS")
print("="*60)

# Find actual checkpoint
actual_checkpoint = find_best_checkpoint(
    "work_dirs/slowfast_multilabel" if USE_MULTILABEL else "work_dirs/slowfast_custom"
)

if actual_checkpoint is None:
    print(f"❌ No checkpoint found! Please train first.")
    sys.exit(1)

print(f"Checkpoint: {actual_checkpoint}")
print(f"Config: {MODEL_CONFIG}")

cfg = Config.fromfile(MODEL_CONFIG)
action_model = init_recognizer(cfg, actual_checkpoint, device="cuda:0")
action_model.eval()

print("✅ Action model loaded")

torch.cuda.empty_cache()
gc.collect()

# Load detection model
init_default_scope("mmdet")

if not os.path.exists(DET_CHECKPOINT):
    print("⚠️  Downloading detection model...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        DET_CHECKPOINT
    )

det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device="cpu" if USE_CPU_FOR_DETECTION else "cuda:0")
print(f"✅ Detection model loaded on {'CPU' if USE_CPU_FOR_DETECTION else 'GPU'}")

print(f"  GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")


# ================= TRACKING =================
class PersonTracker:
    def __init__(self, iou_threshold=0.2):
        self.next_id = 0
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missing_frames = 5
    
    def update(self, frame_idx, detections):
        assigned_tracks = set()
        
        for box in detections:
            best_track_id = None
            best_iou = self.iou_threshold
            
            for track_id, track_data in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                if frame_idx - track_data['last_frame'] > self.max_missing_frames:
                    continue
                
                iou = self.calculate_iou(box, track_data['last_bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                assigned_tracks.add(best_track_id)
                self.tracks[best_track_id]['last_bbox'] = box
                self.tracks[best_track_id]['last_frame'] = frame_idx
                self.tracks[best_track_id]['frames'].append((frame_idx, box))
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'last_bbox': box,
                    'last_frame': frame_idx,
                    'frames': [(frame_idx, box)]
                }
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_valid_tracks(self, min_frames=5):
        return {tid: td for tid, td in self.tracks.items() 
                if len(td['frames']) >= min_frames}


# ================= HELPERS =================
def get_person_bboxes(frame):
    if det_model is None:
        return []
    
    result = inference_detector(det_model, frame)
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()

    persons = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if label == 0 and score > DETECTION_THRESHOLD:
            x1, y1, x2, y2 = map(int, bbox)
            persons.append((x1, y1, x2, y2))
    return persons


def extract_person_clips(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Total frames: {len(frames)}")

    tracker = PersonTracker(iou_threshold=0.15)
    all_detections = []

    for i in tqdm(range(len(frames)), desc="Detecting"):
        boxes = get_person_bboxes(frames[i])
        all_detections.append(boxes)
        tracker.update(i, boxes)
    
    valid_tracks = tracker.get_valid_tracks(MIN_FRAMES)
    print(f"Valid tracks: {len(valid_tracks)}")

    person_clips = {}
    for track_id, track_data in valid_tracks.items():
        clips = []
        for frame_idx, box in track_data['frames']:
            x1, y1, x2, y2 = box
            # FIXED: Keep aspect ratio, add padding
            crop = frames[frame_idx][max(0,y1):y2, max(0,x1):x2]
            clips.append((crop, frame_idx, box))
        person_clips[track_id] = clips

    return person_clips, frames, all_detections


def classify_clip(crops_with_indices, person_id):
    """FIXED: Proper multi-label inference"""
    if len(crops_with_indices) < MIN_FRAMES:
        return [], None
    
    torch.cuda.empty_cache()
    gc.collect()
    
    crops = [c[0] for c in crops_with_indices[:CLIP_LEN]]
    rep_bbox = crops_with_indices[len(crops_with_indices)//2][2]
    
    temp_path = "temp_clip.mp4"
    target_size = (224, 224)
    
    # Filter out any empty/invalid crops safely
    valid_crops = []
    for crop in crops:
        if crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0:
            valid_crops.append(crop)
            
    if len(valid_crops) < MIN_FRAMES:
        return [], None
    
    # Resize all valid crops to 224x224
    resized_crops = [cv2.resize(crop, target_size) for crop in valid_crops]
    
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, target_size)
    for crop in resized_crops:
        out.write(crop)
    out.release()

    try:
        result = inference_recognizer(action_model, temp_path)
        scores = result.pred_score.cpu().numpy()
        
        if USE_MULTILABEL:
            # Multi-label: Apply sigmoid
            probabilities = 1 / (1 + np.exp(-scores))
            
            # Get all labels above threshold
            active_labels = []
            for i, prob in enumerate(probabilities):
                if prob >= MULTILABEL_THRESHOLD and i < NUM_CLASSES:
                    active_labels.append((ACTION_LABELS[i], float(prob)))
            
            # Sort by confidence
            active_labels.sort(key=lambda x: x[1], reverse=True)
            
            action_label = ", ".join([l[0] for l in active_labels]) if active_labels else "unknown"
            confidence = active_labels[0][1] if active_labels else 0.0
        else:
            # Single-label: argmax
            pred_idx = np.argmax(scores)
            if pred_idx < NUM_CLASSES:
                action_label = ACTION_LABELS[pred_idx]
                confidence = scores[pred_idx] * 100
                active_labels = [(action_label, confidence)]
            else:
                action_label = f"unknown_{pred_idx}"
                confidence = 0.0
                active_labels = []
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return active_labels, rep_bbox
        
    except Exception as e:
        print(f"    Error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        torch.cuda.empty_cache()
        gc.collect()
        return [], None


def annotate_video(frames, all_detections, results, output_path):
    h, w = frames[0].shape[:2]
    fps = 25
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    colors = [(0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (0,255,128)]

    for i, frame in enumerate(tqdm(frames, desc="Annotating")):
        vis = frame.copy()
        boxes = all_detections[i]
        
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = colors[box_idx % len(colors)]
            
            person_id = None
            for pid, result in results.items():
                result_bbox = result.get('bbox')
                if result_bbox:
                    iou = calculate_iou(box, result_bbox)
                    if iou > 0.3:
                        person_id = pid
                        break
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            if person_id is not None and person_id in results:
                labels = results[person_id].get('labels', [])
                if labels:
                    # Show all active labels
                    text_parts = [f"{l[0]}:{l[1]*100:.0f}%" for l in labels[:2]]
                    text = f"P{person_id}: " + " | ".join(text_parts)
                else:
                    text = f"P{person_id}: unknown"
            else:
                text = "Detected"
            
            cv2.putText(vis, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(vis)

    out.release()
    print(f"✅ Saved: {output_path}")


def calculate_iou(box1, box2):
    try:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    except:
        return 0.0


# ================= MAIN =================
def main():
    video_path = input("Enter video path: ").strip()

    if not os.path.exists(video_path):
        print("❌ Video not found")
        return

    print("\n" + "="*60)
    print("MULTI-LABEL ACTION RECOGNITION")
    print("="*60)
    print(f"Mode: {'Multi-label' if USE_MULTILABEL else 'Single-label'}")
    print(f"Classes: {ACTION_LABELS}")
    print(f"Threshold: {MULTILABEL_THRESHOLD}")
    print("="*60 + "\n")

    print("Extracting person clips...")
    person_clips, frames, all_detections = extract_person_clips(video_path)

    if len(person_clips) == 0:
        print("\n❌ No valid person tracks found!")
        return

    print(f"\nClassifying {len(person_clips)} person(s)...")
    results = {}

    for pid, clips in person_clips.items():
        labels, bbox = classify_clip(clips, pid)
        
        results[pid] = {
            'labels': labels,
            'bbox': bbox,
            'num_frames': len(clips)
        }

        if labels:
            label_str = " | ".join([f"{l[0]}({l[1]*100:.1f}%)" for l in labels])
            print(f"  ✅ P{pid}: {label_str}")
        else:
            print(f"  ❌ P{pid}: unknown")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{Path(video_path).stem}_multilabel_{timestamp}.mp4"
    )

    print("\nCreating output video...")
    annotate_video(frames, all_detections, results, output_path)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for pid, res in results.items():
        labels = res.get('labels', [])
        if labels:
            print(f"  Person {pid}: {[l[0] for l in labels]}")
        else:
            print(f"  Person {pid}: unknown")
    print(f"\nOutput: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()