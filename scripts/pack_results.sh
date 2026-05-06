#!/usr/bin/env bash
set -euo pipefail

# Pack compact results and logs for retrieval.
#
# Usage:
#   RUN_NAME=pod1 bash scripts/pack_results.sh
#
# Optional:
#   INCLUDE_WORK=1 bash scripts/pack_results.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-stage12_results}"
INCLUDE_WORK="${INCLUDE_WORK:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p packed

OUT="packed/${RUN_NAME}_${STAMP}.tar.gz"

items=()
[[ -d output ]] && items+=(output)
[[ -d logs ]] && items+=(logs)

if [[ "$INCLUDE_WORK" == "1" && -d work ]]; then
  items+=(work)
fi

if [[ "${#items[@]}" -eq 0 ]]; then
  echo "[ERR] nothing to pack; output/ and logs/ do not exist"
  exit 1
fi

echo "[PACK] ${items[*]} -> $OUT"
tar -czf "$OUT" "${items[@]}"

echo "[OK] packed: $OUT"
du -h "$OUT"
