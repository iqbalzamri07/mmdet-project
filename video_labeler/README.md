# ActionMark — Video labeling → SlowFast training → Test

Label action segments in videos (calling, smoking, eating, …), fine-tune SlowFast, then test with bounding boxes and action labels.

---

## How to run the server

### 1. One-time setup (first time only)

```bash
cd /home/newadmin/mmdet-project
source venv/bin/activate
pip install -r video_labeler/requirements.txt
chmod +x run_labeler.sh
```

### 2. Start ActionMark

**Option A — launcher script (recommended)**

```bash
cd /home/newadmin/mmdet-project
./run_labeler.sh
```

**Option B — manual command**

```bash
cd /home/newadmin/mmdet-project
source venv/bin/activate
export PYTHONPATH=/home/newadmin/mmdet-project
python -m uvicorn video_labeler.backend.app:app --host 0.0.0.0 --port 8765 --reload
```

### 3. Open the UI

In your browser go to:

**http://127.0.0.1:8765**

Keep the terminal open while using the app. Stop the server with `Ctrl+C`.

### Custom host / port

```bash
HOST=0.0.0.0 PORT=8765 ./run_labeler.sh
```

---

## What you can do in the UI

| Tab | Purpose |
|-----|---------|
| **Label** | Upload videos, mark action segments, export dataset, train |
| **Test** | Pick a trained checkpoint, upload a video, see boxes + labels |

### Label workflow

1. **Upload** a video  
2. Choose an **action class**  
3. Seek to the pose/action → **Mark start** → **Mark end**  
4. Optional: **Draw crop** around the person  
5. **Save segment** (repeat for more actions)  
6. **Save annotations**  
7. **Export dataset**  
8. **Train model**

### Test workflow

1. Open the **Test** tab  
2. Choose a checkpoint (e.g. `best_acc_top1_epoch_*.pth` or `epoch_50.pth`)  
3. **Upload test video** → **Run inference**  
4. Wait for the job, then play the result (boxes + labels)  
5. Predictions below 70% confidence are shown as `unknown`

### Keyboard (Label tab)

| Key | Action |
|-----|--------|
| Space | Play / pause |
| ← / → | Prev / next frame |
| S | Mark start |
| E | Mark end |
| Enter | Save segment |

---

## Default classes

`sitting`, `standing`, `walking`, `calling`, `playing_phone`, `smoking`, `eating`

Add more in the sidebar. Labels are stored in `video_labeler/data/labels.json` and synced into the SlowFast config on export/train.

---

## Where files go

| Path | Contents |
|------|----------|
| `video_labeler/data/videos/` | Uploaded labeling videos |
| `video_labeler/data/annotations/` | Segment JSON annotations |
| `data/actionmark_dataset/` | Exported training clips |
| `data/custom_actions_videos_clean/` | Train/val lists for SlowFast |
| `work_dirs/slowfast_multilabel/` | Trained model checkpoints (`.pth`) |
| `video_labeler/data/tests/outputs/` | Annotated inference result videos |

> Export only writes under `data/actionmark_dataset/` (and rebuilds the clean symlink tree). It does not delete files under `video_labeler/data/videos/`.

---

## Project layout

```
video_labeler/
  backend/     FastAPI + export + train + inference
  frontend/    ActionMark UI
  data/
    videos/        uploaded videos
    annotations/   JSON labels
    exports/       export summaries
    jobs/          training job status + logs
    tests/         test uploads + inference outputs
run_labeler.sh     start the web server
```

---

## API (optional)

- `GET /api/labels` · `PUT /api/labels`
- `GET /api/videos` · `POST /api/videos/upload`
- `GET /api/videos/{id}/file`
- `PUT /api/annotations/{id}`
- `POST /api/export`
- `POST /api/train` · `GET /api/train/status`
- `GET /api/models`
- `POST /api/test/run` · `GET /api/test/status`
- `GET /api/test/result/{job_id}/video`
