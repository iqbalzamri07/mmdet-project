# mmdet-project

Custom action recognition project (OpenMMLab + ActionMark UI).

## Install (base environment)

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

Or with `pyproject.toml` (after installing PyTorch as above):

```bash
pip install .
```

---

## How to run ActionMark (label · train · test)

ActionMark is the web app for labeling videos, training SlowFast, and testing with boxes + action labels.

```bash
cd /home/newadmin/mmdet-project
chmod +x run_labeler.sh
./run_labeler.sh
```

Then open **http://127.0.0.1:8765**

Manual start:

```bash
cd /home/newadmin/mmdet-project
source venv/bin/activate
export PYTHONPATH=/home/newadmin/mmdet-project
python -m uvicorn video_labeler.backend.app:app --host 0.0.0.0 --port 8765 --reload
```

Full ActionMark docs: [video_labeler/README.md](video_labeler/README.md)

---

## Other entry points

| Script / path | Purpose |
|---------------|---------|
| `./run_labeler.sh` | ActionMark web UI |
| `configs/slowfast_multilabel.py` | SlowFast train config |
| `mmaction2/tools/train.py` | Train SlowFast from CLI |
| `slowfast_memory_optimized.py` | Legacy CLI inference |
| `mmlab-script/mmaction2.py` | Skeleton (ST-GCN) pipeline |

Trained weights are saved under `work_dirs/slowfast_multilabel/`.
