"""Sections 7.7-7.8 + 10: two-stage LightGBM cross-fit, OOF storage, threshold tuning, LOCO.

Everything is stored per (partition, country); rows of p1/ctx2/p2 files are aligned with the concatenation of
that partition's feature chunks. Files under OOF_DIR/{tag}/, models under MODEL_DIR/{tag}/.
LEAKAGE RULE: pool k's stage-2 features only ever use p1 from the model NOT trained on pool k.
"""
import csv
import datetime as dt
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from . import io
from .context import CTX2_FEATURES, stage2_context
from .decide import decide, loss_decomposition, save_thresholds, tune_thresholds, load_thresholds
from .features import FEATURES, feat_files
from .metrics import blocking_report
from .pools import countries, load_eval_s1, load_partition

S2_FEATURES = FEATURES + CTX2_FEATURES
OTHER = {"P0": "P1", "P1": "P0"}


def _oof(tag, kind, part, c) -> Path:
    return C.OOF_DIR / tag / f"{kind}_{part}_{c}.parquet"


def _model_path(tag, stage, pool) -> Path:
    return C.MODEL_DIR / tag / f"m{stage}_{pool}.txt"


def log_experiment(name, cv, notes=""):
    p = Path("experiments.csv")
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "name", "cv", "lb", "notes"])
        w.writerow([dt.datetime.now().strftime("%Y-%m-%d %H:%M"), name, f"{cv:.5f}", "", notes])


# ---------------------------------------------------------------- data loading
def _train_queries(part, cs):
    """35% of the pool's queries (seed 42), union over countries."""
    rng = np.random.default_rng(C.SEED)
    qs = []
    for c in cs:
        _, q = load_partition(part, c)
        qs.append(q[rng.random(len(q)) < C.TRAIN_QUERY_FRAC])
    return np.concatenate(qs)


def load_frame(part, c, stage, tag, q_keep=None, cols=None) -> pl.DataFrame:
    """q_row, s1_row, [y], is_s3 + stage features for (part, c); optional q_row filter."""
    extra = None
    if stage == 2:
        extra = pl.read_parquet(_oof(tag, "ctx2", part, c))
    out, off = [], 0
    for f in feat_files(part, c):
        df = pl.read_parquet(f)
        n = df.height
        if extra is not None:
            df = pl.concat([df, extra.slice(off, n)], how="horizontal")
        off += n
        if q_keep is not None:
            df = df.filter(pl.col("q_row").is_in(q_keep))
        if cols is not None:
            df = df.select([k for k in ["q_row", "s1_row", "y", "is_s3"] if k in df.columns and k not in cols] + cols)
        out.append(df)
    if extra is not None:
        assert off == extra.height, "ctx2 misaligned with feature chunks"
    return pl.concat(out)


# ---------------------------------------------------------------- fit / predict
def fit(df: pl.DataFrame, cols, params, rounds, label) -> lgb.Booster:
    t0 = time.time()
    q = df["q_row"].to_numpy()
    uq = np.unique(q)
    rng = np.random.default_rng(C.SEED + 1)
    h = np.isin(q, uq[rng.random(len(uq)) < C.HOLDOUT_FRAC])
    y = df["y"].to_numpy()
    X = df.select(cols).to_numpy().astype(np.float32, copy=False)
    del df
    dtr = lgb.Dataset(X[~h], y[~h], feature_name=list(cols), free_raw_data=True)
    dva = lgb.Dataset(X[h], y[h], reference=dtr)
    del X
    b = lgb.train(dict(params, num_threads=C.N_JOBS), dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["holdout"],
                  callbacks=[lgb.early_stopping(C.EARLY_STOP, verbose=False), lgb.log_evaluation(250)])
    print(f"[fit] {label}: rows {len(y):,} (pos {y.mean():.4f}), best_iter {b.best_iteration}, "
          f"holdout logloss {b.best_score['holdout']['binary_logloss']:.5f}, {time.time() - t0:.0f}s")
    imp = sorted(zip(cols, b.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(v for _, v in imp) or 1
    print("  top gain: " + ", ".join(f"{k} {v / tot:.3f}" for k, v in imp[:15]))
    return b


def predict_frame(boosters, df: pl.DataFrame, cols) -> np.ndarray:
    X = df.select(cols).to_numpy().astype(np.float32, copy=False)
    return np.mean([b.predict(X, num_iteration=b.best_iteration, num_threads=C.N_JOBS) for b in boosters], axis=0).astype(np.float32)


def predict_stage(boosters, part, c, stage, tag, out_kind):
    cols = FEATURES if stage == 1 else S2_FEATURES
    ctx = pl.read_parquet(_oof(tag, "ctx2", part, c)) if stage == 2 else None
    outs, off = [], 0
    for f in feat_files(part, c):
        df = pl.read_parquet(f)
        if ctx is not None:
            df = pl.concat([df, ctx.slice(off, df.height)], how="horizontal")
        off += df.height
        keep = [k for k in ("q_row", "s1_row", "y", "is_s3") if k in df.columns]
        outs.append(df.select(keep).with_columns(pl.Series(out_kind, predict_frame(boosters, df, cols))))
    res = pl.concat(outs)
    path = _oof(tag, out_kind, part, c)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.write_parquet(path)
    return res


def build_ctx2(part, c, tag):
    p1 = pl.read_parquet(_oof(tag, "p1", part, c))
    stage2_context(p1).write_parquet(_oof(tag, "ctx2", part, c))


def load_booster(tag, stage, pool):
    return lgb.Booster(model_file=str(_model_path(tag, stage, pool)))


def _save(b, tag, stage, pool):
    p = _model_path(tag, stage, pool)
    p.parent.mkdir(parents=True, exist_ok=True)
    b.save_model(str(p), num_iteration=b.best_iteration)


# ---------------------------------------------------------------- evaluation
def evaluate(tag, kind, pools=("P0", "P1"), cs=None, T=None, label="", decomp=True):
    """Macro F0.5 of the decision policy on OOF predictions over the given pools/countries."""
    cs = cs or countries("train")
    gt = io.gt_rows()
    frames, truths, unis = [], [], []
    for part in pools:
        for c in cs:
            df = pl.read_parquet(_oof(tag, kind, part, c)).rename({kind: "p"})
            s1_rows, q_rows = load_partition(part, c)
            uni = load_eval_s1(part, c)
            truths.append(gt.filter(pl.col("s1_row").is_in(uni) & pl.col("q_row").is_in(q_rows)))
            frames.append(df.select("q_row", "s1_row", "y", "p"))
            unis.append(uni)
    df, truth, uni = pl.concat(frames), pl.concat(truths), np.concatenate(unis)
    if T is None:
        T = load_thresholds()
    pred = decide(df, *T)
    if decomp:
        out, _ = loss_decomposition(df, pred, truth, uni, label=f"{label} T={T}")
        return out["F"], df, truth, uni
    from .metrics import macro_f05
    f = macro_f05(uni, pred["s1_row"].to_numpy(), pred["q_row"].to_numpy(), truth["s1_row"].to_numpy(), truth["q_row"].to_numpy())
    print(f"[eval] {label} T={T}: F={f:.5f}")
    return f, df, truth, uni


# ---------------------------------------------------------------- orchestration
def train_stage1(tier0=False, tag="main", cs=None, pools=None, reuse=False):
    cs = cs or countries("train")
    pools = pools or (("P0",) if tier0 else ("P0", "P1"))
    for pool in pools:
        if reuse and _model_path(tag, 1, pool).exists() and all(_oof(tag, "p1", OTHER[pool], c).exists() for c in cs):
            print(f"[train1] reuse existing {_model_path(tag, 1, pool)} and its OOF predictions")
            continue
        tq = _train_queries(pool, cs)
        df = pl.concat([load_frame(pool, c, 1, tag, q_keep=tq, cols=FEATURES) for c in cs])
        b = fit(df, FEATURES, C.LGB1, C.LGB1_ROUNDS, f"stage1 {tag} train {pool} {cs}")
        del df
        _save(b, tag, 1, pool)
        for c in cs:
            predict_stage([b], OTHER[pool], c, 1, tag, "p1")
    eval_pools = tuple(OTHER[p] for p in pools)
    best, res = _single_best(tag, "p1", eval_pools, cs)
    f76, *_ = evaluate(tag, "p1", eval_pools, cs, T=(0.76, 0.76), label=f"stage1 {tag} OOF {eval_pools}")
    return f76, best


def _single_best(tag, kind, pools, cs):
    _, df, truth, uni = evaluate(tag, kind, pools, cs, T=(0.76, 0.76), decomp=False, label=f"{kind} {tag}")
    (fb, tb), res = tune_thresholds(df, truth, uni, single=True)
    print(f"[tune] {tag} {kind} single threshold best F={fb:.5f} at {tb[0]}")
    return (fb, tb), res


def train_stage2(tag="main", cs=None):
    cs = cs or countries("train")
    for part in ("P0", "P1"):
        for c in cs:
            build_ctx2(part, c, tag)
    for pool in ("P0", "P1"):
        tq = _train_queries(pool, cs)
        df = pl.concat([load_frame(pool, c, 2, tag, q_keep=tq, cols=S2_FEATURES) for c in cs])
        b = fit(df, S2_FEATURES, C.LGB2, C.LGB2_ROUNDS, f"stage2 {tag} train {pool} {cs}")
        del df
        _save(b, tag, 2, pool)
        for c in cs:
            predict_stage([b], OTHER[pool], c, 2, tag, "p2")
    _single_best(tag, "p1", ("P0", "P1"), cs)
    return _single_best(tag, "p2", ("P0", "P1"), cs)


def tune_from_oof(tag="main"):
    cs = countries("train")
    kind = "p2" if _oof(tag, "p2", "P0", cs[0]).exists() else "p1"
    pools = ("P0", "P1") if _oof(tag, kind, "P0", cs[0]).exists() else ("P1",)
    _, df, truth, uni = evaluate(tag, kind, pools, cs, T=(0.76, 0.76), decomp=False, label=f"{kind} pre-tune")
    (f1, t1), _ = tune_thresholds(df, truth, uni, single=True)
    (f2, t2), res = tune_thresholds(df, truth, uni)
    print(f"[tune] {kind} on {pools}: single-threshold F={f1:.5f} at {t1[0]}; (T1,T2) F={f2:.5f} at {t2}")
    save_thresholds(*t2, extra=dict(kind=kind, F=f2, F_single=f1, T_single=t1[0]))
    evaluate(tag, kind, pools, cs, T=t2, label=f"{kind} tuned all")
    for c in cs:
        evaluate(tag, kind, pools, [c], T=t2, label=f"{kind} tuned {c}")
    return t2


def loco(tag_prefix="loco"):
    """Train stages 1+2 on one country's pools only, evaluate on the other country's pools; compare with in-country."""
    cs = countries("train")
    T = load_thresholds()
    res = {}
    for c in cs:  # country-only cross-fit (serves as in-country for c and cross-country for the others)
        tag = f"{tag_prefix}_{c}"
        train_stage1(tag=tag, cs=[c])
        train_stage2(tag=tag, cs=[c])
    for src in cs:
        tag = f"{tag_prefix}_{src}"
        for tgt in cs:
            if tgt == src:
                res[(src, tgt)] = evaluate(tag, "p2", ("P0", "P1"), [tgt], T=T, decomp=False, label=f"LOCO train {src} eval {tgt}")[0]
                continue
            for pool in ("P0", "P1"):
                m1 = load_booster(tag, 1, OTHER[pool])
                predict_stage([m1], pool, tgt, 1, tag, "p1")
                build_ctx2(pool, tgt, tag)
                m2 = load_booster(tag, 2, OTHER[pool])
                predict_stage([m2], pool, tgt, 2, tag, "p2")
            res[(src, tgt)] = evaluate(tag, "p2", ("P0", "P1"), [tgt], T=T, decomp=False, label=f"LOCO train {src} eval {tgt}")[0]
    for tgt in cs:
        for src in cs:
            if src != tgt:
                d = res[(src, tgt)] - res[(tgt, tgt)]
                print(f"[loco] eval {tgt}: in-country {res[(tgt, tgt)]:.5f}  trained on {src} {res[(src, tgt)]:.5f}  drop {100 * d:+.2f}pp")
    (C.MODEL_DIR / "loco.json").write_text(json.dumps({f"{a}->{b}": v for (a, b), v in res.items()}, indent=1))
    return res
