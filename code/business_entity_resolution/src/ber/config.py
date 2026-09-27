"""Paths and every tunable parameter. Paths are relative to the working directory unless overridden by env."""
import json
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
# R08: France-only format rules in normalize.py. FR_NORM sets both halves; FR_NORM_ADDR (street types, regions/departments,
# number prefixes) and FR_NORM_NAME (spaced legal forms, et -> and, ...) override it. 0/0 = R06 (best LB).
FR_NORM = os.environ.get("FR_NORM", "0") == "1"
FR_NORM_ADDR = os.environ.get("FR_NORM_ADDR", "1" if FR_NORM else "0") == "1"
FR_NORM_NAME = os.environ.get("FR_NORM_NAME", "1" if FR_NORM else "0") == "1"
# R07 label-free lexicon imputation for vocabularies unseen in train (xfeats.impute_lexicon); lost on the LB (R06 > R07)
FR_IMPUTE = os.environ.get("FR_IMPUTE", "0") == "1"
# R09 same-address rescue for test countries absent from train (rescue.py): argmax pair with equal house numbers and the
# same street -> p >= RESCUE_P, unless another candidate S1 of the query has the same address key (sibling guard).
# LOST on the LB (R09 0.976531 vs R06 0.981021): France's rejected same-address pairs are mostly distractors. Off.
FR_RESCUE = os.environ.get("FR_RESCUE", "0") == "1"
RESCUE_P = float(os.environ.get("RESCUE_P", "0.90"))
RESCUE_JAC = float(os.environ.get("RESCUE_JAC", "0.8"))
# blend members for test countries absent from train, e.g. "x1,x2" = R06's France (default: the same tags as elsewhere);
# their thresholds come from that blend's own OOF tuning (models/blend-<tags>/thresholds.json)
BLEND_TAGS_UNSEEN = tuple(os.environ["BLEND_TAGS_UNSEEN"].split(",")) if os.environ.get("BLEND_TAGS_UNSEEN") else None
# thresholds for test countries absent from train, "T1,T2" (default: the OOF-tuned ones); an LB probe for France only
T_UNSEEN = tuple(float(x) for x in os.environ["T_UNSEEN"].split(",")) if os.environ.get("T_UNSEEN") else None
# per-source caps of the generator (train GT: <= 5 S2 and <= 6 S3 matches per S1); decide() keeps the highest p
CAPS = os.environ.get("CAPS", "1") == "1"
CAP_S2, CAP_S3 = 5, 6
TAG = os.environ.get("TAG", "main")  # namespace of models, OOF predictions and thresholds (R04 = main)
USE_XFEATS = os.environ.get("USE_XFEATS", "0") == "1"  # add the name-token edit features (xfeats.py) to stage 1
USE_NFEATS = os.environ.get("USE_NFEATS", "0") == "1"  # add the number-relation / number-oracle-lexicon features (nfeats.py)
# lexicon features are blanked for this share of training queries, so the model also learns the label-free route it needs
# where the lexicon has no entries (France: its generator words are unseen in train)
MASK_LEX_FRAC = float(os.environ.get("MASK_LEX_FRAC", "0.3"))
MASK_HOLDOUT = os.environ.get("MASK_HOLDOUT", "1") == "1"  # 0: the early-stopping holdout keeps its lexicon (as test does)
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
# per-run overrides as JSON, e.g. LGB1_PARAMS='{"num_leaves": 255, "min_child_samples": 200}' (x1/x2 used the defaults)
LGB1 = {**LGB_COMMON, "num_leaves": 127, **json.loads(os.environ.get("LGB1_PARAMS", "{}"))}
LGB1_ROUNDS = 3000
LGB2 = {**LGB_COMMON, "num_leaves": 63, **json.loads(os.environ.get("LGB2_PARAMS", "{}"))}
LGB2_ROUNDS = 2000
EARLY_STOP = 100

# decision policy
TIE_EPS = 1e-6
T1_DEFAULT, T2_DEFAULT = 0.70, 0.76
T1_GRID = (0.10, 0.90)  # was (0.40, 0.90): the dev optimum sat on the 0.40 floor (dev R03c: 0.98292 -> 0.98624 at 0.10)
T2_GRID = (0.30, 0.96)
GRID_STEP = 0.02
