#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"

echo "ActionMark → http://127.0.0.1:${PORT}"
exec "$PYTHON" -m uvicorn video_labeler.backend.app:app --host "$HOST" --port "$PORT" --reload
