"""Background SlowFast training job runner."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .export import export_dataset, sync_train_config_labels


def _job_path(job_id: str) -> Path:
    return config.JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, data: Dict[str, Any]) -> None:
    path = _job_path(job_id)
    data["job_id"] = job_id
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_jobs() -> list:
    jobs = []
    for p in sorted(config.JOBS_DIR.glob("*.json"), reverse=True):
        with open(p, "r", encoding="utf-8") as f:
            jobs.append(json.load(f))
    return jobs


def get_active_job() -> Optional[Dict[str, Any]]:
    for job in list_jobs():
        if job.get("status") in ("queued", "running", "exporting"):
            # Check if process died
            pid = job.get("pid")
            if pid and job.get("status") == "running":
                try:
                    os.kill(pid, 0)
                except OSError:
                    job["status"] = "failed"
                    job["error"] = "Process exited unexpectedly"
                    _write_job(job["job_id"], job)
                    continue
            return job
    return None


def start_training(export_first: bool = True, sync_labels: bool = True) -> Dict[str, Any]:
    active = get_active_job()
    if active:
        return {"ok": False, "error": "A job is already running", "job": active}

    job_id = datetime.utcnow().strftime("train_%Y%m%d_%H%M%S")
    log_path = config.JOBS_DIR / f"{job_id}.log"

    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "export_first": export_first,
        "log_path": str(log_path),
        "pid": None,
        "export_summary": None,
        "error": None,
    }
    _write_job(job_id, job)

    # Launch detached worker via subprocess calling this module's CLI helper
    cmd = [
        str(config.VENV_PYTHON),
        "-m",
        "video_labeler.backend.train_runner",
        "--job-id",
        job_id,
    ]
    if not export_first:
        cmd.append("--skip-export")
    if not sync_labels:
        cmd.append("--skip-sync-labels")

    # Parent just spawns; child updates job file
    proc = subprocess.Popen(
        cmd,
        cwd=str(config.PROJECT_ROOT),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job["status"] = "running"
    job["pid"] = proc.pid
    _write_job(job_id, job)
    return {"ok": True, "job": job}


def stop_training(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        return {"ok": False, "error": "Job not found"}
    pid = job.get("pid")
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    job["status"] = "stopped"
    job["error"] = "Stopped by user"
    _write_job(job_id, job)
    return {"ok": True, "job": job}


def _run_job(job_id: str, skip_export: bool = False, skip_sync: bool = False) -> None:
    job = get_job(job_id) or {
        "job_id": job_id,
        "status": "running",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "log_path": str(config.JOBS_DIR / f"{job_id}.log"),
    }
    job["pid"] = os.getpid()
    job["status"] = "running"
    _write_job(job_id, job)

    try:
        if not skip_sync:
            print("[train] Syncing ACTION_LABELS into SlowFast config...")
            sync_train_config_labels()

        if not skip_export:
            job["status"] = "exporting"
            _write_job(job_id, job)
            print("[train] Exporting labeled clips...")
            summary = export_dataset()
            job["export_summary"] = summary
            _write_job(job_id, job)
            if not summary.get("ok"):
                raise RuntimeError(summary.get("error", "Export failed"))
            if summary.get("clips", 0) == 0:
                raise RuntimeError("Export produced 0 clips")

        if not config.TRAIN_SCRIPT.exists():
            raise RuntimeError(f"Train script missing: {config.TRAIN_SCRIPT}")
        if not config.TRAIN_CONFIG.exists():
            raise RuntimeError(f"Train config missing: {config.TRAIN_CONFIG}")

        job["status"] = "training"
        _write_job(job_id, job)
        print(f"[train] Starting SlowFast: {config.TRAIN_CONFIG}")
        env = os.environ.copy()
        # Reduce fragmentation on small GPUs (RTX 2050 4GB)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        cmd = [
            str(config.VENV_PYTHON),
            str(config.TRAIN_SCRIPT),
            str(config.TRAIN_CONFIG),
        ]
        result = subprocess.run(
            cmd,
            cwd=str(config.PROJECT_ROOT),
            check=False,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Training exited with code {result.returncode}")

        job["status"] = "completed"
        job["error"] = None
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        _write_job(job_id, job)
        print("[train] Completed successfully")
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = datetime.utcnow().isoformat() + "Z"
        _write_job(job_id, job)
        print(f"[train] Failed: {exc}")
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-sync-labels", action="store_true")
    args = parser.parse_args()
    _run_job(args.job_id, skip_export=args.skip_export, skip_sync=args.skip_sync_labels)


if __name__ == "__main__":
    main()
