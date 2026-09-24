"""Project paths and global settings.

Works unchanged on the laptop, a SageMaker notebook instance, or Kaggle: set the
AMLC_ROOT env var if the repo lives somewhere read-only / unusual.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("AMLC_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"              # unzipped official dataset goes here
PROCESSED = DATA / "processed"  # cleaned tables
IMAGES = DATA / "images"        # downloaded images
FEATURES = DATA / "features"    # cached embeddings / engineered features (.npy / .parquet)
MODELS = ROOT / "models"
OOF = ROOT / "oof"              # out-of-fold + test predictions, one file per experiment
SUBMISSIONS = ROOT / "submissions"
EXPERIMENT_LOG = ROOT / "experiments.csv"

SEED = 42
N_FOLDS = 5

for _p in (RAW, PROCESSED, IMAGES, FEATURES, MODELS, OOF, SUBMISSIONS):
    _p.mkdir(parents=True, exist_ok=True)
