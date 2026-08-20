# ActionMark

Label videos, train SlowFast, and test posture + activity recognition with person boxes.

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

Full docs: [video_labeler/README.md](video_labeler/README.md)

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
| `work_dirs/slowfast_multilabel/` | Your trained `.pth` checkpoints |
