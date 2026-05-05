"""
RGB MULTI-PERSON ACTION RECOGNITION (MMAction2)
MEMORY OPTIMIZED VERSION for small GPUs (4GB or less)
- Uses CPU for detection (saves GPU memory)
- Reduces clip length (uses less memory)
- Fixes memory fragmentation
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import urllib.request
import gc
import sys

# Set environment variable to fix memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from mmengine.config import Config
from mmaction.apis import init_recognizer, inference_recognizer
from mmdet.apis import init_detector, inference_detector
from mmengine import init_default_scope

# ================= CONFIG =================
# Use the training config (important for architecture consistency)
MODEL_CONFIG = "configs/slowfast_custom.py"

# Your trained model checkpoint
MODEL_CHECKPOINT = "work_dirs/slowfast_custom/best_acc_top1_epoch_30.pth"

# Detection model paths
CHECKPOINT_DIR = "checkpoints"  # Must be defined before use!
DET_CONFIG = "mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
DET_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")

# Your custom action labels (model trained on these)
CUSTOM_LABELS = ["smoking", "sitting", "standing", "walking", "calling", "playing_phone"]

OUTPUT_DIR = "rgb_outputs"
CLIP_LEN = 32  # Match training config
FRAME_STRIDE = 1
MIN_FRAMES = 5
DETECTION_THRESHOLD = 0.7
USE_CPU_FOR_DETECTION = True

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model will predict these 6 labels directly
ACTION_LABELS = CUSTOM_LABELS
print(f"✓ Loaded {len(ACTION_LABELS)} custom action labels: {ACTION_LABELS}")

# ================= INIT MODELS =================
print("="*60)
print("LOADING CUSTOM TRAINED SLOWFAST MODEL")
print("="*60)
print(f"Checkpoint: {MODEL_CHECKPOINT}")
print(f"Config: {MODEL_CONFIG}")
print(f"Actions: {ACTION_LABELS}")
print("="*60 + "\n")

print("Loading models...")
print(f"  GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

# Load action model on GPU
cfg = Config.fromfile(MODEL_CONFIG)
action_model = init_recognizer(cfg, MODEL_CHECKPOINT, device="cuda:0")
action_model.eval()

print("✅ Custom SlowFast model loaded on GPU")

# Clear GPU memory after loading action model
torch.cuda.empty_cache()
gc.collect()

# Load detection model on CPU to save GPU memory
if USE_CPU_FOR_DETECTION:
    print("  Loading detection model on CPU (to save GPU memory)...")
    init_default_scope("mmdet")
    
    if os.path.exists(DET_CHECKPOINT):
        det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device="cpu")
        print("✅ Detection model loaded on CPU")
    else:
        print("⚠️  Detection model checkpoint not found")
        print("  Downloading...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
            DET_CHECKPOINT
        )
        det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device="cpu")
        print("✅ Detection model loaded on CPU")
else:
    init_default_scope("mmdet")
    if os.path.exists(DET_CHECKPOINT):
        det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device="cuda:0")
        print("✅ Detection model loaded on GPU")
    else:
        print("⚠️  Detection model not found")
        det_model = None

print(f"  GPU memory after loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

# ================= IMPROVED TRACKING =================
class PersonTracker:
    """Improved person tracker using IoU matching"""
    
    def __init__(self, iou_threshold=0.2):
        self.next_id = 0
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_missing_frames = 5
    
    def update(self, frame_idx, detections):
        """Update tracks with new detections"""
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
        
        return self.tracks
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
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
        """Get tracks with enough frames"""
        valid_tracks = {}
        for track_id, track_data in self.tracks.items():
            if len(track_data['frames']) >= min_frames:
                valid_tracks[track_id] = track_data
        return valid_tracks
    
    def get_track_summary(self):
        """Print summary of tracks"""
        print(f"\n📊 Tracking Summary:")
        print(f"   Total tracks created: {len(self.tracks)}")
        
        valid_tracks = self.get_valid_tracks(MIN_FRAMES)
        print(f"   Valid tracks (≥{MIN_FRAMES} frames): {len(valid_tracks)}")
        
        frame_counts = [len(t['frames']) for t in self.tracks.values()]
        if frame_counts:
            print(f"   Avg frames per track: {np.mean(frame_counts):.1f}")
            print(f"   Max frames in a track: {max(frame_counts)}")


# ================= HELPERS =================
def get_person_bboxes(frame):
    """Detect people in a frame with higher threshold"""
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
    """Extract cropped clips for all detected people with improved tracking"""
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
    for i in tqdm(range(len(frames)), desc="Detecting and tracking"):
        frame = frames[i]
        boxes = get_person_bboxes(frame)
        all_detections.append(boxes)
        tracker.update(i, boxes)
    
    valid_tracks = tracker.get_track_summary()
    valid_tracks_dict = tracker.get_valid_tracks(MIN_FRAMES)

    # Extract clips with bboxes
    person_clips = {}
    for track_id, track_data in valid_tracks_dict.items():
        clips = []
        for frame_idx, box in track_data['frames']:
            x1, y1, x2, y2 = box
            crop = frames[frame_idx][y1:y2, x1:x2]
            clips.append((crop, frame_idx, box))
        person_clips[track_id] = clips

    return person_clips, frames, all_detections


def classify_clip(crops_with_indices, person_id):
    """Classify a person's action using your custom trained SlowFast model"""
    if len(crops_with_indices) < MIN_FRAMES:
        return "unknown", 0.0, None
    
    print(f"  Classifying person {person_id}...")
    
    # Aggressive GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Check available memory before classification
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_free = (torch.cuda.get_device_properties(0).total_memory - mem_reserved) / 1024**3
        print(f"    GPU memory: {mem_allocated:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved, {mem_free:.2f} GiB free")
    
    # Take first CLIP_LEN crops
    crops = [c[0] for c in crops_with_indices[:CLIP_LEN]]
    
    # Get representative bbox
    rep_bbox = crops_with_indices[len(crops_with_indices)//2][2]
    
    # Save temp video
    temp_path = "temp_clip.mp4"
    h, w = crops[0].shape[:2]
    
    # Use same size as training (224x224)
    target_size = (224, 224)
    resized_crops = [cv2.resize(crop, target_size) for crop in crops]
    
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, target_size)
    for crop in resized_crops:
        out.write(crop)
    out.release()

    try:
        result = inference_recognizer(action_model, temp_path)
        scores = result.pred_score.cpu().numpy()
        pred_idx = np.argmax(scores)
        
        # Model directly predicts your 6 custom actions
        action_label = ACTION_LABELS[pred_idx] if pred_idx < len(ACTION_LABELS) else f"action_{pred_idx}"
        confidence = scores[pred_idx] * 100
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Aggressive cleanup after classification
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return action_label, confidence, rep_bbox
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    ⚠️  GPU OOM - skipping this person")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        else:
            print(f"    Classification error: {e}")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return "error", 0.0, None
    except Exception as e:
        print(f"    Classification error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return "error", 0.0, None


def annotate_video(frames, all_detections, results, output_path):
    """Save video with bounding boxes and action labels"""
    h, w = frames[0].shape[:2]
    fps = 25
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    colors = [(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0),(0,255,128)]

    for i, frame in enumerate(tqdm(frames, desc="Creating output video")):
        vis = frame.copy()
        boxes = all_detections[i]
        
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = colors[box_idx % len(colors)]
            
            person_id = None
            for pid, result in results.items():
                result_bbox = result.get('bbox')
                if result_bbox is not None and isinstance(result_bbox, tuple):
                    try:
                        iou = calculate_iou(box, result_bbox)
                        if iou > 0.3:
                            person_id = pid
                            break
                    except:
                        continue
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            if person_id is not None and person_id in results:
                action = results[person_id].get('action', '...')
                conf = results[person_id].get('confidence', 0)
                text = f"P{person_id}: {action} ({conf:.1f}%)"
            else:
                text = "Detected"
            
            cv2.putText(vis, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(vis)

    out.release()
    print(f"✅ Saved: {output_path}")


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
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
    print("CUSTOM TRAINED SLOWFAST MODEL")
    print("="*60)
    print(f"Model: Trained on your 6 custom actions")
    print(f"Checkpoint: {MODEL_CHECKPOINT}")
    print(f"Actions: {ACTION_LABELS}")
    print(f"Detection model: {'CPU (saves GPU memory)' if USE_CPU_FOR_DETECTION else 'GPU'}")
    print(f"Clip length: {CLIP_LEN} frames (same as training)")
    print(f"Frame stride: {FRAME_STRIDE}")
    print("="*60 + "\n")

    print("Extracting person clips with improved tracking...")
    person_clips, frames, all_detections = extract_person_clips(video_path)

    if len(person_clips) == 0:
        print("\n❌ No valid person tracks found!")
        return

    print(f"\nDetected {len(person_clips)} valid person(s) with ≥{MIN_FRAMES} frames")
    print("Running classification with your custom trained model...")
    results = {}

    for pid, clips in person_clips.items():
        action, conf, bbox = classify_clip(clips, pid)
        
        results[pid] = {
            'action': action,
            'confidence': conf,
            'bbox': bbox,
            'num_frames': len(clips)
        }

        status = "✅" if action != "error" else "❌"
        print(f"{status} Person {pid}: {action} ({conf:.2f}%) [{len(clips)} frames]")

    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{Path(video_path).stem}_custom_{timestamp}.mp4"
    )

    print("\nCreating output video...")
    annotate_video(frames, all_detections, results, output_path)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Total frames: {len(frames)}")
    print(f"People detected: {len(person_clips)}")
    print(f"People classified: {len([r for r in results.values() if r['action'] != 'error'])}")
    print(f"Output: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
