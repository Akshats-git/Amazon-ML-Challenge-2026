#!/usr/bin/env bash
# Run INSIDE a SageMaker notebook instance terminal (JupyterLab -> File -> New -> Terminal),
# from the repo root after uploading/cloning it to ~/SageMaker/AmazonML2026.
# Reuses the preinstalled PyTorch conda env (CUDA already set up on GPU instances).
#
#   bash scripts/sagemaker_bootstrap.sh [s3://amlc26-<team>]
set -euo pipefail
cd "$(dirname "$0")/.."
BUCKET="${1:-}"

source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
ENV=$(conda env list | awk '/pytorch_p3/ {print $1}' | tail -1)
conda activate "$ENV"
echo "using conda env: $ENV"

pip install -q -U lightgbm xgboost catboost optuna polars rapidfuzz regex \
  transformers sentence-transformers accelerate peft timm opencv-python-headless
pip install -q -e . --no-deps

python -m ipykernel install --user --name amlc --display-name "Python (amlc)"

if [[ -n "$BUCKET" ]]; then
  bash scripts/team_sync.sh pull "$BUCKET"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Done. Pick kernel 'Python (amlc)'. Remember: STOP the instance when idle."
