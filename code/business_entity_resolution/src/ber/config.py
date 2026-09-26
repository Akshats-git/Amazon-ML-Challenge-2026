"""Paths and every tunable parameter. Paths are relative to the working directory unless overridden by env."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/raw"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "work"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 4))
# sparse top-n matmul is cache/memory-bound: on a 6-core laptop 3-4 threads beat 8 (0.20 vs 0.26 ms/query on US)
BLOCK_THREADS = int(os.environ.get("BLOCK_THREADS", min(N_JOBS, 4)))

# Dev mode: keep the full half-country S1 index, subsample queries to DEV_FRAC of matches + distractors.
# Dev runs write pools/cands/feats/models/oof under WORK_DIR/dev so they never mix with full runs.
DEV_FRAC = float(os.environ.get("DEV_FRAC", "0"))
USE_HAND_MAPS = os.environ.get("USE_HAND_MAPS", "0") == "1"
TAG = os.environ.get("TAG", "main")  # namespace of models, OOF predictions and thresholds (R04 = main)
USE_XFEATS = os.environ.get("USE_XFEATS", "0") == "1"  # add the name-token edit features (xfeats.py) to stage 1
# lexicon features are blanked for this share of training queries, so the model also learns the label-free route it needs
# where the lexicon has no entries (France: its generator words are unseen in train)
MASK_LEX_FRAC = float(os.environ.get("MASK_LEX_FRAC", "0.3"))
# hold the training matrix as float16 (features are stored with 10 mantissa bits = float16 precision; values are clipped
# to +-65504) and feed LightGBM through lgb.Sequence in float32 batches: halves the RAM of load_train, so a 13 GB laptop
# can train on a larger TRAIN_QUERY_FRAC
X16 = os.environ.get("X16", "0") == "1"
WRITE_CANDIDATES = os.environ.get("WRITE_CANDIDATES", "1") == "1"  # 0: skip candidate_pairs.tsv (identical across runs)

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
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "2000000"))  # rows per feature parquet file (readers load one file at a time)
WORKER_CHUNK = int(os.environ.get("WORKER_CHUNK", "150000"))  # pairs per worker task (bounds per-worker RAM)
FEAT_DROP_BITS = 13  # stored features keep 10 of 23 mantissa bits (rel. err <= 5e-4, ~27% smaller files); LightGBM bins to 255 anyway
PARQUET_KW = dict(compression="zstd", compression_level=15)

# models
TRAIN_QUERY_FRAC = float(os.environ.get("TRAIN_QUERY_FRAC", "0.35"))  # laptop (13 GB RAM) runs use 0.10-0.15
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
T1_GRID = (0.10, 0.90)  # was (0.40, 0.90): the dev optimum sat on the 0.40 floor (dev R03c: 0.98292 -> 0.98624 at 0.10)
T2_GRID = (0.30, 0.96)
GRID_STEP = 0.02
