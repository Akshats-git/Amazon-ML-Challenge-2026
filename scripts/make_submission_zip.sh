#!/usr/bin/env bash
# Package code for the final upload (the portal wants the code/notebooks as a zip,
# plus a separate 1-2 page approach document). Data, models and caches are excluded.
#
#   bash scripts/make_submission_zip.sh [team_name]
set -euo pipefail
cd "$(dirname "$0")/.."

TEAM="${1:-team}"
OUT="${TEAM}_amlc2026_code_$(date +%m%d_%H%M).zip"

zip -r "$OUT" \
  README.md pyproject.toml uv.lock \
  src scripts notebooks \
  experiments.csv \
  -x '*/__pycache__/*' '*.ipynb_checkpoints*' '*.npy' '*.pkl' '*.pt' '*.bin' '*.safetensors' \
  2>/dev/null || true

echo
unzip -l "$OUT" | tail -n +1
echo
echo "Created $OUT ($(du -h "$OUT" | cut -f1))."
echo "Check: no dataset files inside, notebooks have outputs you want to show, approach doc uploaded separately."
