# ActionMark

Label videos, train SlowFast (multi-label posture + activity), and test with person boxes.

Full product docs (Label · Test · Compare · collab · paths · API):

→ **[video_labeler/README.md](video_labeler/README.md)**

## Install

```bash
cd /home/newadmin/mmdet-project
python -m venv venv
source venv/bin/activate

# PyTorch (CUDA 11.8 example)
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -r video_labeler/requirements.txt
```

## Run

```bash
./run_labeler.sh
```

Open **http://127.0.0.1:8765**

## What you get

| Area | Features |
|------|----------|
| **Label** | Segments, crops, Need/Done library, Next video, crop size warning, shortcuts, save history, locks for multi-user |
| **Train** | Export clips → SlowFast, epochs in UI, ONNX export |
| **Test** | Upload / test-upload library / camera, Results browser, delete uploads & results, **Compare A vs B** checkpoints |

Label library (`video_labeler/data/videos/`) and Test uploads (`video_labeler/data/tests/inputs/`) stay separate on purpose.

## Repository layout

| Path | Purpose |
|------|---------|
| `video_labeler/` | Web app (label · export · train · test) |
| `configs/slowfast_multilabel.py` | SlowFast training config |
| `mmaction2/` | MMAction2 (SlowFast train/infer) |
| `mmdetection/` | MMDetection (Faster R-CNN person detector) |
| `checkpoints/` | Pretrained Faster R-CNN + SlowFast weights |
| `data/actionmark_dataset/` | Exported training clips |
| `data/custom_actions_videos_clean/` | Train/val lists for SlowFast |
| `work_dirs/slowfast_multilabel/` | Your trained `.pth` / `.onnx` checkpoints |
