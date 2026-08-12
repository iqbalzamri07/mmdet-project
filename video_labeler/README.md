# ActionMark — Video labeling → SlowFast training

Label action segments in videos (calling, smoking, eating, …), export clips, and fine-tune SlowFast.

## Quick start

```bash
cd /home/newadmin/mmdet-project
source venv/bin/activate
pip install -r video_labeler/requirements.txt
chmod +x run_labeler.sh
./run_labeler.sh
```

Open **http://127.0.0.1:8765**

## Workflow

1. **Upload** a video
2. Choose an **action class**
3. Seek to the pose/action → **Mark start** → **Mark end**
4. Optional: **Draw crop** around the person
5. **Save segment** (repeat for more actions)
6. **Save annotations**
7. **Export dataset** (cuts clips into `data/custom_actions_videos/`)
8. **Train model** (runs SlowFast via `configs/slowfast_multilabel.py`)

### Keyboard

| Key | Action |
|-----|--------|
| Space | Play / pause |
| ← / → | Prev / next frame |
| S | Mark start |
| E | Mark end |
| Enter | Save segment |

## Default classes

`sitting`, `standing`, `walking`, `calling`, `playing_phone`, `smoking`, `eating`

Add more in the sidebar. Labels are stored in `video_labeler/data/labels.json` and synced into the SlowFast config on export/train.

## API (optional)

- `GET /api/labels` · `PUT /api/labels`
- `GET /api/videos` · `POST /api/videos/upload`
- `GET /api/videos/{id}/file`
- `PUT /api/annotations/{id}`
- `POST /api/export`
- `POST /api/train` · `GET /api/train/status`

## Layout

```
video_labeler/
  backend/     FastAPI + export + train runner
  frontend/    ActionMark UI
  data/
    videos/        uploaded videos
    annotations/   JSON labels
    exports/       export summaries
    jobs/          training job status + logs
```

Exported clips → `data/actionmark_dataset/{train,val}/<class>/`  
Clean lists → `data/custom_actions_videos_clean/`  
Checkpoints → `work_dirs/slowfast_multilabel/`

> Export only writes under `data/actionmark_dataset/` (and rebuilds the clean symlink tree). It does not delete files under `video_labeler/data/videos/`.

## Test a trained model

1. Open ActionMark → **Test** tab  
2. Choose a checkpoint (e.g. `best_acc_top1_epoch_5.pth`)  
3. **Upload test video** → **Run inference**  
4. Wait for the job (detect people → SlowFast → annotate)  
5. Play the result video with **bounding boxes + action labels**

Checkpoints are listed from `work_dirs/slowfast_multilabel/`. Result videos are saved under `video_labeler/data/tests/outputs/`.
