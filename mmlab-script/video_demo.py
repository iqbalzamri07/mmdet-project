from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2
import os

register_all_modules()

config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

class_names = model.dataset_meta['classes']

input_video_path = 'footage/cam1_00_20260415010152.mp4'
output_video_path = 'process_video/cam1_00_20260415010152.mp4'

if not os.path.exists(input_video_path):
    print(f"Error: Could not find '{input_video_path}'.")
    exit()

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing '{input_video_path}'")
print(f"Total frames: {total_frames}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = inference_detector(model, frame)

    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    for bbox, score, label in zip(bboxes, scores, labels):
        class_name = class_names[label]
        
        if class_name not in ['person', 'cell phone']:
            continue

        if score < 0.4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        text = f'{class_name}: {score:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    # Print progress instead of showing the video
    frame_count += 1
    if frame_count % 30 == 0: # Print every 30 frames so it doesn't spam the terminal
        print(f"Processed {frame_count}/{total_frames} frames...")

    scale_percent = 50
    new_width = int(frame.shape[1] * scale_percent / 100)
    new_height = int(frame.shape[0] * scale_percent / 100)
    display_frame = cv2.resize(frame, (new_width, new_height))


cap.release()
out.release()
print(f"Done! Saved processed video to '{output_video_path}'")