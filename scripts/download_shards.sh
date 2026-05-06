#!/usr/bin/env bash
set -euo pipefail

# Download assigned dataset shards and checkpoints.
#
# Usage:
#   BASE_URL="https://xxxxx.trycloudflare.com" SHARDS="1 9" bash scripts/download_shards.sh
#
# Optional:
#   SKIP_CHECKPOINTS=1 bash scripts/download_shards.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_URL="${BASE_URL:-}"
SHARDS="${SHARDS:-}"
SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-0}"

if [[ -z "$BASE_URL" ]]; then
  echo "[ERR] BASE_URL is required."
  echo 'Example: BASE_URL="https://xxxxx.trycloudflare.com" SHARDS="1 9" bash scripts/download_shards.sh'
  exit 1
fi

mkdir -p downloads checkpoints logs

download_file() {
  local url="$1"
  local out="$2"

  if [[ -s "$out" ]]; then
    echo "[SKIP] exists: $out"
    return 0
  fi

  echo "[DOWNLOAD] $url -> $out"
  wget -c --tries=20 --timeout=30 --read-timeout=30 -O "$out" "$url"
}

if [[ -n "$SHARDS" ]]; then
  for shard in $SHARDS; do
    download_file "${BASE_URL%/}/${shard}.tar.gz" "downloads/${shard}.tar.gz"
  done
else
  echo "[WARN] SHARDS is empty; no dataset tarballs downloaded."
fi

if [[ "$SKIP_CHECKPOINTS" != "1" ]]; then
  download_file "${BASE_URL%/}/moge2.pt" "checkpoints/moge2.pt"
  download_file "${BASE_URL%/}/ppd_moge.pth" "checkpoints/ppd_moge.pth"
  download_file "${BASE_URL%/}/depth_anything_v2_vitl.pth" "checkpoints/depth_anything_v2_vitl.pth"
  download_file "${BASE_URL%/}/ppd.pth" "checkpoints/ppd.pth"
fi

echo "[OK] download complete"
du -sh downloads checkpoints || true
