"""Project paths and global settings.

Works unchanged on the laptop, a SageMaker notebook instance, or Kaggle: set the
AMLC_ROOT env var if the repo lives somewhere read-only / unusual.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("AMLC_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"              # official dataset: raw/train/*.tsv, raw/test/*.tsv (official file names)
PROCESSED = DATA / "processed"  # normalised sources (.parquet), cached per split
FEATURES = DATA / "features"    # cached candidate pairs / pair features (.parquet)
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"  # one folder per run: matching_results.tsv + candidate_pairs.tsv
EXPERIMENT_LOG = ROOT / "experiments.csv"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
ID_COL, NAME_COL, ADDR_COL, COUNTRY_COL = "entity_id", "business_name", "business_address", "country"
GT_S1_COL, GT_MATCH_COL = "source1_entity_id", "matched_entity_ids"
CAND_COL = "candidate_entity_ids"

BETA = 0.5   # official metric: macro F_0.5 per Source-1 entity
SEED = 42
N_FOLDS = 5

for _p in (RAW, PROCESSED, FEATURES, MODELS, SUBMISSIONS):
    _p.mkdir(parents=True, exist_ok=True)
