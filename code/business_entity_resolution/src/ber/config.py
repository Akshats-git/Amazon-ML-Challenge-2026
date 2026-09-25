"""Paths and every tunable parameter. Paths are relative to the working directory unless overridden by env."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/raw"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "work"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 4))

# Dev mode: keep the full half-country S1 index, subsample queries to DEV_FRAC of matches + distractors.
# Dev runs write pools/cands/feats/models/oof under WORK_DIR/dev so they never mix with full runs.
DEV_FRAC = float(os.environ.get("DEV_FRAC", "0"))
USE_HAND_MAPS = os.environ.get("USE_HAND_MAPS", "0") == "1"

NORM_DIR = WORK_DIR / "norm"
DICT_DIR = WORK_DIR / "dicts"
STAGE_DIR = WORK_DIR / "dev" if DEV_FRAC > 0 else WORK_DIR
POOL_DIR = STAGE_DIR / "pools"
CAND_DIR = STAGE_DIR / "cands"
FEAT_DIR = STAGE_DIR / "feats"
MODEL_DIR = STAGE_DIR / "models"
OOF_DIR = STAGE_DIR / "oof"

SPLITS = ("train", "test")
POOLS = ("P0", "P1")

# blocking
BLOCK_K = 10
DF_CAP = int(os.environ.get("DF_CAP", "10000"))
BLOCK_WEIGHTS = (0.25, 0.50, 0.25)  # name words, address uni+bigrams, name_sq char-4
BLOCK_CHUNK = 200_000
TEST_PAIR_GUARD = 110_000_000

# transliteration dictionaries
TRANSLIT_MIN_COUNT = 3
TRANSLIT_MIN_PURITY = 0.5

# features
UNMATCHED_CUTOFF = 0.8
FEAT_CHUNK = 5_000_000
WORKER_CHUNK = 250_000

# models
TRAIN_QUERY_FRAC = 0.35
HOLDOUT_FRAC = 0.10
SEED = 42
LGB_COMMON = dict(
    objective="binary",
    learning_rate=0.05,
    min_child_samples=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    seed=SEED,
    verbose=-1,
)
LGB1 = dict(LGB_COMMON, num_leaves=127)
LGB1_ROUNDS = 3000
LGB2 = dict(LGB_COMMON, num_leaves=63)
LGB2_ROUNDS = 2000
EARLY_STOP = 100

# decision policy
TIE_EPS = 1e-6
T1_DEFAULT, T2_DEFAULT = 0.70, 0.76
T1_GRID = (0.40, 0.90)
T2_GRID = (0.60, 0.94)
GRID_STEP = 0.02
