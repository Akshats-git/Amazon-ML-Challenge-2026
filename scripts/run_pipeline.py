"""End-to-end baseline: normalise -> block -> pair features -> LightGBM -> decide -> outputs.

    # iterate: sample of train, out-of-fold macro F0.5 + threshold tuning (minutes)
    uv run python scripts/run_pipeline.py dev --frac 0.05

    # submit: train on a sample of train, predict the FULL test set, write + validate outputs
    uv run python scripts/run_pipeline.py test --frac 0.05 --name tfidf_lgb_v1

The test run holds ~10M normalised S2/S3 rows and ~50M candidate pairs: it needs roughly
8 GB of free RAM (close Chrome) or a bigger cloud machine.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlc.blocking import block, to_id_pairs  # noqa: E402
from amlc.config import ADDR_COL, MODELS, NAME_COL, PROCESSED  # noqa: E402
from amlc.data import dev_sample, load_source  # noqa: E402
from amlc.features import add_group_features, pair_features  # noqa: E402
from amlc.matching import decide, label_pairs, predict, train_oof, tune_threshold  # noqa: E402
from amlc.metrics import blocking_report, fbeta_macro, per_entity_fbeta  # noqa: E402
from amlc.normalize import normalize  # noqa: E402
from amlc.submit import write_outputs  # noqa: E402
from amlc.utils import log_experiment, save_pickle, seed_everything  # noqa: E402

FEAT_CHUNK = 2_000_000


def norm(df: pl.DataFrame) -> pl.DataFrame:
    return normalize(df).drop(NAME_COL, ADDR_COL)  # raw text no longer needed; saves RAM


def load_norm(split: str, source: int) -> pl.DataFrame:
    """Normalised full source, cached as parquet in data/processed/."""
    path = PROCESSED / f"{split}_source{source}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    df = norm(load_source(split, source))
    df.write_parquet(path)
    return df


def featurize(pairs: pl.DataFrame, s1n, qn) -> pl.DataFrame:
    pairs = add_group_features(pairs)
    parts = [pair_features(pairs.slice(i, FEAT_CHUNK), s1n, qn) for i in range(0, pairs.height, FEAT_CHUNK)]
    return pl.concat(parts)


def fit(args):
    """Build the training pair set from a density-preserving train sample and fit OOF models."""
    t = time.time()
    d = dev_sample(args.frac)
    s1n, qn = norm(d["s1"]), norm(pl.concat([d["s2"], d["s3"]]))
    pairs = block(s1n, qn, top_k=args.top_k, min_sim=args.min_sim, name_weight=args.name_weight)
    rep = blocking_report(to_id_pairs(pairs, s1n, qn), d["gt_pairs"], d["s1_ids"])
    df = label_pairs(featurize(pairs, s1n, qn), s1n, qn, d["gt_pairs"])
    print(f"train pairs {df.height:,}  positives {df['y'].mean():.3f}  ({time.time() - t:.0f}s)")
    oof, models = train_oof(df)
    df = df.with_columns(proba=pl.Series(oof))
    thr, f = tune_threshold(df, s1n, qn, d["gt_pairs"], d["s1_ids"])
    return dict(models=models, threshold=thr, oof_f=f, blocking=rep, df=df, s1n=s1n, qn=qn, sample=d)


def run_dev(args):
    r = fit(args)
    pred = to_id_pairs(decide(r["df"], threshold=r["threshold"]), r["s1n"], r["qn"])
    per = per_entity_fbeta(pred, r["sample"]["gt_pairs"], r["sample"]["s1_ids"])
    print("\nOOF macro F0.5 by true match count:")
    print(per.with_columns(pl.col("n_true").clip(upper_bound=6)).group_by("n_true").agg(
        pl.len(), pl.col("f").mean(), pl.col("precision").mean(), pl.col("recall").mean()).sort("n_true"))
    log_experiment(f"dev_{args.name}", r["oof_f"], notes=f"frac={args.frac} k={args.top_k} min_sim={args.min_sim} "
                   f"w={args.name_weight} thr={r['threshold']:.2f} pair_recall={r['blocking']['pair_recall']:.4f}")
    assert abs(fbeta_macro(pred, r["sample"]["gt_pairs"], r["sample"]["s1_ids"]) - r["oof_f"]) < 1e-9


def run_test(args):
    r = fit(args)
    save_pickle({k: r[k] for k in ("models", "threshold")}, MODELS / f"{args.name}.pkl")
    del r["df"], r["qn"], r["s1n"], r["sample"]
    gc.collect()

    t = time.time()
    s1n = load_norm("test", 1)
    if args.test_query_rows:  # smoke test: all S1 rows, first N rows of S2/S3 (not cached)
        qn = pl.concat([norm(load_source("test", s, n_rows=args.test_query_rows)) for s in (2, 3)])
    else:
        qn = pl.concat([load_norm("test", 2), load_norm("test", 3)])
    print(f"test normalised: S1={s1n.height:,} S2+S3={qn.height:,} ({time.time() - t:.0f}s)")
    pairs = add_group_features(block(s1n, qn, top_k=args.top_k, min_sim=args.min_sim, name_weight=args.name_weight))
    print(f"test candidate pairs: {pairs.height:,}")
    proba = np.empty(pairs.height, dtype=np.float32)
    for i in range(0, pairs.height, FEAT_CHUNK):
        proba[i:i + FEAT_CHUNK] = predict(r["models"], pair_features(pairs.slice(i, FEAT_CHUNK), s1n, qn))
    pairs = pairs.with_columns(proba=pl.Series(proba))
    matches = to_id_pairs(decide(pairs, threshold=r["threshold"]), s1n, qn)
    cands = to_id_pairs(pairs, s1n, qn)
    run_dir = write_outputs(args.name, s1n["entity_id"], matches, cands)
    log_experiment(f"test_{args.name}", r["oof_f"], notes=f"{run_dir.name} thr={r['threshold']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    ap.add_argument("--frac", type=float, default=0.05, help="share of train S1 entities used for training")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-sim", type=float, default=0.3)
    ap.add_argument("--name-weight", type=float, default=0.5)
    ap.add_argument("--test-query-rows", type=int, default=None,
                    help="smoke test: only the first N rows of test S2/S3 (outputs still cover every S1)")
    a = ap.parse_args()
    seed_everything()
    run_dev(a) if a.mode == "dev" else run_test(a)
