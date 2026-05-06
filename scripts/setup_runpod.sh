#!/usr/bin/env bash
set -euo pipefail

# Setup Stage12 repo on a fresh RunPod container.
#
# Usage:
#   bash scripts/setup_runpod.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export UV_LINK_MODE=copy

echo "[INFO] repo root: $ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[INFO] uv version:"
uv --version

echo "[INFO] syncing environment..."
uv sync

echo "[INFO] CUDA / import smoke test..."
uv run python - <<'PY'
import sys
print("python:", sys.version)

import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

import cv2
print("cv2:", cv2.__version__)

import numpy as np
print("numpy:", np.__version__)

import open3d as o3d
print("open3d:", o3d.__version__)

import trimesh
print("trimesh:", trimesh.__version__)

import mitsuba as mi
print("mitsuba:", mi.__version__)
PY

mkdir -p downloads checkpoints work output logs

echo "[OK] setup complete"
