# ActionMark — Label · Train · Test

Web app for labeling CCTV-style action clips, exporting a SlowFast multi-label dataset, training, and testing with **person boxes + posture/activity**.

Built on FastAPI + a static frontend, with MMAction2 (SlowFast) and MMDetection (Faster R-CNN person detector).

---

## Quick start

### One-time setup

```bash
cd /home/newadmin/mmdet-project
source venv/bin/activate
pip install -r video_labeler/requirements.txt
chmod +x run_labeler.sh
```

(Project-level PyTorch / MMDetection / MMAction2 deps: see root [Readme.md](../Readme.md).)

### Start the server

**Recommended**

```bash
cd /home/newadmin/mmdet-project
./run_labeler.sh
```

**Manual**

```bash
cd /home/newadmin/mmdet-project
source venv/bin/activate
export PYTHONPATH=/home/newadmin/mmdet-project
python -m uvicorn video_labeler.backend.app:app --host 0.0.0.0 --port 8765 --reload
```

### Open the UI

**http://127.0.0.1:8765**

Custom bind:

```bash
HOST=0.0.0.0 PORT=8765 ./run_labeler.sh
```

Stop with `Ctrl+C`.

---

## What the app does

| Mode | Purpose |
|------|---------|
| **Label** | Upload/library videos, mark segments (posture + activity + optional crop), collaborate with locks, export, train, export ONNX |
| **Test** | Run inference on **test-only** videos (upload / prior uploads / camera), browse results, compare two checkpoints, delete old uploads/results |

**Important separation**

| Use | Folder | Notes |
|-----|--------|--------|
| Label library | `video_labeler/data/videos/` | For annotation only |
| Test uploads | `video_labeler/data/tests/inputs/` | Inference only — **not** Label videos |
| Test results | `video_labeler/data/tests/outputs/` + `jobs/` | Annotated result videos + job JSON |

Label videos **cannot** be used as Test inputs by design.

---

## Label mode

### Layout

- **Left rail** — Videos (Need / Done), Dataset (export / train / ONNX), Classes  
- **Center** — Player + scrubber  
- **Right rail** — Annotate (labels, range, crop, saved segments, shortcuts, save history)  
- **Mobile (~900px)** — Tabs: Videos | Player | Annotate  

### How to label (for teammates)

1. Set **Your name** in the Videos panel (locks + save history).  
2. Pick a video from **Need** (or **Done** to review).  
3. Choose **posture** + **activity** (e.g. standing + smoking).  
4. **Mark start / end** on the timeline.  
5. Optional: **Draw crop** tightly around **one** person.  
6. **Save segment** → repeat → **Save all**.  
7. **Next video** jumps to the next unlabeled clip (skips locks held by others).  

**Tip:** Prefer short, one-person crops. A soft warning appears if the crop covers a large fraction of the frame (≥40% / ≥70%).

### Collaboration (multi-user)

- Enter a display name; each browser gets a `client_id`.  
- Opening a video **locks** it (“Currently editing”).  
- Heartbeat keeps the lock alive; leaving / deselecting releases it.  
- Click the **same video again** to **deselect**, close Annotate, and release the lock.  
- Library list auto-refreshes when the shared revision bumps (does not wipe your open player).  
- Saves store `last_annotator` and an `annotation_log` in the annotation JSON.

### Dataset / train / ONNX

From the **Dataset** panel:

1. **Export dataset** — builds clips under `data/actionmark_dataset/` and train/val lists.  
2. Set **epochs** and **Train**.  
3. Checkpoints land in `work_dirs/slowfast_multilabel/` (`.pth`).  
4. **Export ONNX** from a `.pth` for faster/portable Test (`.onnx` also appears in the Test model list).

### Keyboard shortcuts (Label)

| Key | Action |
|-----|--------|
| `Space` | Play / pause |
| `←` / `→` | Previous / next frame |
| `S` | Mark start |
| `E` | Mark end |
| `Enter` | Save segment |

(Also listed under **Keyboard shortcuts** in the Annotate rail.)

---

## Test mode

### Modes

| Mode | Behavior |
|------|----------|
| **Single** | One checkpoint → one run |
| **Compare** | Checkpoint **A** then **B** on the **same** clip (sequential; safer for GPU), then side-by-side videos + prediction tables |

### Sources

| Source | Use |
|--------|-----|
| **Upload** | New file → stored under `tests/inputs/` |
| **Library** | Re-run a **previous Test upload** (no re-upload) |
| **Camera** | Live short clips (hidden in Compare mode) |

### Results

- **Results** tab lists completed inference jobs.  
- Reopen a past result, download the video, or **delete** it (×).  
- Test uploads also have **delete** (×) under Library.

### Inference pipeline (high level)

1. Faster R-CNN detects persons.  
2. Crops are classified with SlowFast (`.pth` or `.onnx`) for posture + activity.  
3. Output video + person table are shown in the UI.

Low-confidence predictions may appear as weak / unknown depending on thresholds in `backend/infer.py`.

---

## Default classes

**Posture:** `sitting`, `standing`  

**Activity:** `walking`, `calling`, `playing_phone`, `smoking`, `eating`  

Editable in **Classes**. Stored in `video_labeler/data/labels.json` and synced into the SlowFast config on export/train.

A segment is multi-label friendly: one posture **and** one activity (e.g. `standing + smoking`).

---

## Data & paths

| Path | Contents |
|------|----------|
| `video_labeler/data/videos/` | Label library videos |
| `video_labeler/data/annotations/` | Per-video JSON (segments, crops, annotator log) |
| `video_labeler/data/labels.json` | Class taxonomy |
| `video_labeler/data/exports/` | Export summaries |
| `video_labeler/data/jobs/` | Training job status / logs |
| `video_labeler/data/tests/inputs/` | Test uploads |
| `video_labeler/data/tests/outputs/` | Inference result videos |
| `video_labeler/data/tests/jobs/` | Inference job JSON + logs |
| `data/actionmark_dataset/` | Exported training clips |
| `data/custom_actions_videos_clean/` | Train/val lists for SlowFast |
| `work_dirs/slowfast_multilabel/` | Trained `.pth` / exported `.onnx` |
| `configs/slowfast_multilabel.py` | Training config |
| `checkpoints/` | Pretrained Faster R-CNN (+ upstream SlowFast weights as needed) |

Export writes under `data/actionmark_dataset/` (and rebuilds the clean list tree). It does **not** delete Label library videos.

---

## Project layout

```
video_labeler/
  backend/
    app.py           FastAPI routes
    config.py        Paths + label taxonomy
    export.py        Annotations → training clips
    train_runner.py  SlowFast training jobs
    infer.py         Test inference + test library helpers
    onnx_export.py   .pth → .onnx
    collab.py        Locks, heartbeats, library revision
    media.py         Probe / H.264 transcode helpers
  frontend/
    index.html
    css/styles.css
    js/app.js
  data/              videos, annotations, jobs, tests, …
  requirements.txt
  README.md
run_labeler.sh       Project-root launcher
```

---

## API overview

### Labels & videos (Label)

| Method | Path | Notes |
|--------|------|--------|
| GET/PUT | `/api/labels` | Taxonomy |
| GET | `/api/videos` | Paginated library (`labeled`, `q`, …) |
| POST | `/api/videos/upload` | Label upload |
| GET | `/api/videos/{id}/file` · `/meta` | Playback + metadata |
| DELETE | `/api/videos/{id}` | Delete video + annotation |
| GET/PUT | `/api/annotations/{id}` | Load / save segments |

### Collaboration

| Method | Path |
|--------|------|
| POST | `/api/collab/hello` |
| GET | `/api/collab/status` |
| POST | `/api/collab/lock/{video_id}` |
| DELETE | `/api/collab/lock/{video_id}` |
| POST | `/api/collab/heartbeat` |
| POST | `/api/collab/bye` |

### Export & train

| Method | Path |
|--------|------|
| POST | `/api/export` |
| POST | `/api/train` · GET `/api/train/status` · POST `/api/train/stop/{job_id}` |
| GET | `/api/train/log/{job_id}` |

### Models & ONNX

| Method | Path |
|--------|------|
| GET | `/api/models` |
| POST | `/api/models/export-onnx` · GET status |

### Test

| Method | Path | Notes |
|--------|------|--------|
| POST | `/api/test/upload` | Save to `tests/inputs/` |
| POST | `/api/test/run` | Form: `checkpoint` + `file` **or** `test_video` (filename under inputs only) |
| GET | `/api/test/inputs` | List test uploads |
| DELETE | `/api/test/inputs/{filename}` | Delete upload |
| GET | `/api/test/library` | Completed results |
| DELETE | `/api/test/result/{job_id}` | Delete job + output |
| GET | `/api/test/status` | Job status |
| GET | `/api/test/result/{job_id}/video` | Result MP4 |
| POST | `/api/test/live` | Webcam frame batch |

---

## Tips for better models

- Balance rare classes (**smoking**, **calling**) with more clean segments.  
- Prefer **tight one-person crops**; avoid huge boxes that cover most of the frame.  
- Use **Compare** in Test to pick between checkpoints on the same clip.  
- Keep Test uploads/results tidy with the delete (×) controls.  
- Split work with your teammate: don’t edit the same video at once — locks enforce that.

---

## Requirements (app layer)

See `video_labeler/requirements.txt` (FastAPI, uvicorn, multipart, OpenCV, …).  
Training and inference also need the project venv with MMAction2, MMDetection, and PyTorch installed (root setup).
