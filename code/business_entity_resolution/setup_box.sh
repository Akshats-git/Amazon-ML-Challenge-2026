#!/usr/bin/env bash
# One-time setup on a fresh Linux box (SageMaker notebook terminal / EC2). Creates .venv-ber with Python 3.12
# and the pinned requirements, then prints versions, cores and RAM.
#   bash code/business_entity_resolution/setup_box.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v uv >/dev/null || pip install -q uv
uv venv -p 3.12 .venv-ber
uv pip install -p .venv-ber -r "$HERE/requirements.txt"
source .venv-ber/bin/activate
python "$HERE/src/run.py" env
df -h . | tail -1
echo "OK. Next:  source .venv-ber/bin/activate && nohup bash $HERE/run_all.sh > run_all.out 2>&1 &"
