from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import cv2

# 1. Register all modules
register_all_modules()

# 2. Config + Checkpoint
config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 3. Load model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 4. Get class names from the model (e.g., 'person', 'car', etc.)
class_names = model.dataset_meta['classes']

# 5. Open webcam
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# 6. Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

print("Starting webcam loop... Press 'q' to stop.")

# 7. Process loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference (MMDetection handles BGR to RGB conversion automatically internally)
    result = inference_detector(model, frame)

    # Extract bounding boxes, scores, and labels from the result
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()

    # Draw each detection manually using standard OpenCV
    for bbox, score, label in zip(bboxes, scores, labels):
        if score < 0.4:  # Skip low confidence detections
            continue
            
        # Get box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        class_name = class_names[label]
        
        # Draw Green bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw text label above the box
        text = f'{class_name}: {score:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write and show the frame (No RGB/BGR conversion needed anymore!)
    out.write(frame)
    cv2.imshow('MMDetection Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("Done! Saved to output.mp4")