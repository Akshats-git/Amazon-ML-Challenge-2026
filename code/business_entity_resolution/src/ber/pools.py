"""Section 7.5 pools: P_k^c = {S1 of c with md5 half k} + {queries matched to them} + {ALL train distractors of c}.

Partitions used downstream: ("P0", c), ("P1", c) on train and ("test", c) on test, one per country present in S1.
Each partition file work/pools/{part}_{country}.npz holds int32 s1_rows and q_rows (global row keys, see io.py).
"""
import hashlib

import numpy as np
import polars as pl

from . import config as C
from . import io


def s1_half(ids) -> np.ndarray:
    return np.fromiter((int(hashlib.md5(x.encode()).hexdigest(), 16) % 2 for x in ids), dtype=np.int8, count=len(ids))


def pool_path(part: str, country: str):
    return C.POOL_DIR / f"{part}_{country}.npz"


def countries(split: str) -> list:
    return sorted(pl.read_parquet(io.norm_path(split, 1), columns=["country"])["country"].unique().to_list())


def load_partition(part: str, country: str):
    """-> (s1_rows, q_rows) int32 arrays. test partitions are built on the fly (all rows of the country)."""
    if part == "test":
        s1 = pl.read_parquet(io.norm_path("test", 1), columns=["country"]).with_row_index("r")
        q = io.load_norm("test", "q", columns=["country"]).with_row_index("r")
        return (s1.filter(pl.col("country") == country)["r"].cast(pl.Int32).to_numpy(),
                q.filter(pl.col("country") == country)["r"].cast(pl.Int32).to_numpy())
    z = np.load(pool_path(part, country))
    return z["s1_rows"], z["q_rows"]


def load_eval_s1(part: str, country: str) -> np.ndarray:
    """S1 rows the metric is computed over: all pool S1 in full mode, the sampled 10% in dev mode."""
    return np.load(pool_path(part, country))["eval_s1"]


def partitions(parts) -> list:
    out = []
    for p in parts:
        for c in countries("test" if p == "test" else "train"):
            out.append((p, c))
    return out


def build_pools():
    s1 = pl.read_parquet(io.norm_path("train", 1), columns=["entity_id", "country"]).with_row_index("s1_row")
    s1 = s1.with_columns(half=pl.Series(s1_half(s1["entity_id"])))
    q = io.load_norm("train", "q", columns=["country"]).with_row_index("q_row")
    gt = io.gt_rows()
    q = q.join(gt.join(s1.select("s1_row", "half"), on="s1_row").select("q_row", "half"), on="q_row", how="left")
    rng = np.random.default_rng(C.SEED)
    C.POOL_DIR.mkdir(parents=True, exist_ok=True)
    seen = {}
    for c in countries("train"):
        qc = q.filter(pl.col("country") == c)
        distract = qc.filter(pl.col("half").is_null())["q_row"].to_numpy()
        for k in (0, 1):
            s1_rows = s1.filter((pl.col("country") == c) & (pl.col("half") == k))["s1_row"].to_numpy()
            matched = qc.filter(pl.col("half") == k)["q_row"].to_numpy()
            d = distract
            eval_s1 = s1_rows
            if C.DEV_FRAC > 0:  # dev: full S1 index, subsample matches of DEV_FRAC of S1 + DEV_FRAC of distractors
                keep_s1 = rng.random(len(s1_rows)) < C.DEV_FRAC
                eval_s1 = s1_rows[keep_s1]
                gtk = gt.filter(pl.col("s1_row").is_in(eval_s1))
                matched = np.intersect1d(matched, gtk["q_row"].to_numpy())
                d = distract[rng.random(len(distract)) < C.DEV_FRAC]
            q_rows = np.sort(np.concatenate([matched, d])).astype(np.int32)
            np.savez(pool_path(f"P{k}", c), s1_rows=s1_rows.astype(np.int32), q_rows=q_rows,
                     eval_s1=eval_s1.astype(np.int32))
            seen[(k, c)] = set(s1_rows.tolist())
            print(f"P{k} {c}: S1 {len(s1_rows):,}  matched q {len(matched):,}  distractors {len(d):,}  "
                  f"eval S1 {len(eval_s1):,}  distractors per index S1 {len(d) / len(s1_rows):.3f}")
    for c in countries("train"):
        assert not (seen[(0, c)] & seen[(1, c)]), "pool S1 overlap"
    print("pools: S1 sets disjoint")
