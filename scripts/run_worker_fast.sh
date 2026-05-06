#!/usr/bin/env bash
set -euo pipefail

# Run the Stage12 streaming worker with in-process PixelPerfect/MoGe + MVInverse.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

SHARDS="${SHARDS:-}"
LIMIT="${LIMIT:-0}"
MESH_STRIDE="${MESH_STRIDE:-2}"
OVERWRITE="${OVERWRITE:-0}"
DEBUG="${DEBUG:-0}"
DEVICE="${DEVICE:-cuda}"
MITSUBA_VARIANT="${MITSUBA_VARIANT:-cuda_ad_rgb}"
SPP="${SPP:-32}"
FINAL_SPP="${FINAL_SPP:-128}"
ITERS="${ITERS:-80}"

mkdir -p downloads work output logs

args=(
  uv run python stage12_stream_worker_fast.py
  --mesh-stride "$MESH_STRIDE"
  --device "$DEVICE"
  --mitsuba-variant "$MITSUBA_VARIANT"
  --spp "$SPP"
  --final-spp "$FINAL_SPP"
  --iters "$ITERS"
)

if [[ "$LIMIT" != "0" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ "$DEBUG" == "1" ]]; then
  args+=(--save-debug-renders --keep-temp --keep-stage1-assets)
fi

if [[ -n "$SHARDS" ]]; then
  tar_args=()
  for shard in $SHARDS; do
    tar_args+=("downloads/${shard}.tar.gz")
  done
  args+=(--tarballs "${tar_args[@]}")
fi

echo "[INFO] running fast worker:"
printf ' %q' "${args[@]}"
echo

"${args[@]}"
