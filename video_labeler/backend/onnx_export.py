"""Export a trained SlowFast .pth checkpoint to ONNX."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from . import config


def resolve_work_file(name: str) -> Path:
    """Resolve a checkpoint/ONNX filename inside work_dirs/slowfast_multilabel."""
    filename = Path(name or "").name
    if not filename or filename != (name or "").replace("\\", "/").split("/")[-1]:
        raise ValueError("Invalid model filename")
    path = (config.WORK_DIR / filename).resolve()
    work = config.WORK_DIR.resolve()
    if path.parent != work or not path.exists():
        raise FileNotFoundError(f"Model not found: {filename}")
    return path


class _AvgPool3d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2, -3), keepdim=True)


class SlowFastOnnxNet(nn.Module):
    """Backbone + head only, with ONNX-friendly pooling."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.backbone = base_model.backbone
        self.head = base_model.cls_head
        if getattr(self.head, "avg_pool", None) is not None:
            self.head.avg_pool = _AvgPool3d()
        if getattr(self.head, "dropout", None) is not None:
            self.head.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def export_slowfast_onnx(
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    opset: int = 13,
) -> Path:
    from mmengine import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmaction.registry import MODELS

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() != ".pth":
        raise ValueError("ONNX export requires a .pth checkpoint")
    output_path = Path(output_path) if output_path else checkpoint_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(config.TRAIN_CONFIG))
    labels_n = len(config.ACTION_LABELS)
    if hasattr(cfg, "model") and "cls_head" in cfg.model:
        cfg.model["cls_head"]["num_classes"] = labels_n

    init_default_scope(cfg.get("default_scope", "mmaction"))
    base_model = MODELS.build(cfg.model)
    load_checkpoint(base_model, str(checkpoint_path), map_location="cpu")
    base_model.eval()

    wrapped = SlowFastOnnxNet(base_model)
    wrapped.eval()

    dummy = torch.randn(1, 3, 16, 160, 160)
    with torch.no_grad():
        wrapped(dummy)

    torch.onnx.export(
        wrapped,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["cls_score"],
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "cls_score": {0: "batch"}},
    )
    if not output_path.exists():
        raise RuntimeError("ONNX file was not written")
    return output_path


def _job_path(job_id: str) -> Path:
    return config.JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, data: Dict[str, Any]) -> None:
    data["job_id"] = job_id
    data["kind"] = "onnx_export"
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(_job_path(job_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_onnx_export_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_active_onnx_export() -> Optional[Dict[str, Any]]:
    jobs = []
    for p in sorted(config.JOBS_DIR.glob("onnx_export_*.json"), reverse=True):
        with open(p, "r", encoding="utf-8") as f:
            jobs.append(json.load(f))
    for job in jobs:
        if job.get("status") in ("queued", "running"):
            return job
    return jobs[0] if jobs else None


def start_onnx_export(checkpoint_name: str) -> Dict[str, Any]:
    active = get_active_onnx_export()
    if active and active.get("status") in ("queued", "running"):
        return {"ok": False, "error": "An ONNX export is already running", "job": active}

    src = resolve_work_file(checkpoint_name)
    if src.suffix.lower() != ".pth":
        raise ValueError("Select a .pth checkpoint to convert")

    job_id = datetime.utcnow().strftime("onnx_export_%Y%m%d_%H%M%S")
    log_path = config.JOBS_DIR / f"{job_id}.log"
    dest = src.with_suffix(".onnx")
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "checkpoint": src.name,
        "output": str(dest),
        "log_path": str(log_path),
        "pid": None,
        "error": None,
    }
    _write_job(job_id, job)

    cmd = [
        str(config.VENV_PYTHON),
        "-m",
        "video_labeler.backend.onnx_export",
        "--job-id",
        job_id,
        "--checkpoint",
        str(src),
        "--output",
        str(dest),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(config.PROJECT_ROOT),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONPATH": str(config.PROJECT_ROOT)},
    )
    job["status"] = "running"
    job["pid"] = proc.pid
    _write_job(job_id, job)
    return {"ok": True, "job": job}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    job_id = args.job_id
    ckpt = Path(args.checkpoint)
    out = Path(args.output) if args.output else ckpt.with_suffix(".onnx")

    def fail(msg: str) -> None:
        if job_id:
            job = get_onnx_export_job(job_id) or {"job_id": job_id}
            job["status"] = "failed"
            job["error"] = msg
            job["finished_at"] = datetime.utcnow().isoformat() + "Z"
            _write_job(job_id, job)
        raise SystemExit(msg)

    try:
        print(f"[onnx] exporting {ckpt.name} → {out.name}")
        path = export_slowfast_onnx(ckpt, out)
        print(f"[onnx] wrote {path} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
        if job_id:
            job = get_onnx_export_job(job_id) or {"job_id": job_id}
            job["status"] = "completed"
            job["output"] = str(path)
            job["finished_at"] = datetime.utcnow().isoformat() + "Z"
            job["error"] = None
            _write_job(job_id, job)
    except Exception as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()
