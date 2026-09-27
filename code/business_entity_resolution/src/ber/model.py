"""Sections 7.7-7.8 + 10: two-stage LightGBM cross-fit, OOF predictions, threshold tuning, LOCO.

Predictions are float32 .npy arrays aligned with the concatenated feature files of (partition, country):
OOF_DIR/{tag}/{kind}_{part}_{country}.npy (kind p1 / p2). Keys (q_row, s1_row, y, is_s3) are read back from the
feature files, and the stage-2 context is rebuilt from p1 in memory (context.Ctx2). Models under MODEL_DIR/{tag}/.
Memory: training matrices are preallocated and filled one feature file at a time (holdout rows last, so train and
holdout are views), and scoring goes file by file; no step loads a whole 40M-pair partition as a frame.
LEAKAGE RULE: pool k's stage-2 features only ever use p1 from the model NOT trained on pool k.
"""
import csv
import datetime as dt
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from . import io
from .context import CTX2_FEATURES, Ctx2
from .decide import _in, argmax_rows, assign, decide, load_thresholds, loss_decomposition, save_thresholds, tune_thresholds
from .features import FEATURES, feat_files, load_cols
from .pools import countries, load_eval_s1, load_partition
from .nfeats import N_FEATURES, nfeat_path
from .xfeats import LEX_FEATURES, X_FEATURES, xfeat_path

S1_FEATURES = FEATURES + (X_FEATURES if C.USE_XFEATS else []) + (N_FEATURES if C.USE_NFEATS else [])
S2_FEATURES = S1_FEATURES + CTX2_FEATURES
OTHER = {"P0": "P1", "P1": "P0"}


def _oof(tag, kind, part, c) -> Path:
    return C.OOF_DIR / tag / f"{kind}_{part}_{c}.npy"


def _model_path(tag, stage, pool) -> Path:
    return C.MODEL_DIR / tag / f"m{stage}_{pool}.txt"


def load_pred(tag, kind, part, c) -> np.ndarray:
    return np.load(_oof(tag, kind, part, c))


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
    """TRAIN_QUERY_FRAC of the pool's queries (seed 42), union over countries."""
    rng = np.random.default_rng(C.SEED)
    qs = []
    for c in cs:
        _, q = load_partition(part, c)
        qs.append(q[rng.random(len(q)) < C.TRAIN_QUERY_FRAC])
    return np.concatenate(qs)


class _Rows(lgb.Sequence):
    """float16 rows handed to LightGBM as float64 batches (its Sequence path samples doubles); Dataset construction
    never sees a full float32 matrix."""
    batch_size = 65536

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.asarray(self.X[idx], dtype=np.float64)


def _mask(rows, size):
    m = np.zeros(size, bool)
    m[rows] = True
    return m


def ctx2_for(tag, part, c) -> Ctx2:
    """Stage-2 context of (part, c) from its stored p1 (OOF for pools, the P0/P1 average for test)."""
    k = load_cols(part, c, ["q_row", "s1_row", "is_s3"])
    return Ctx2(k["q_row"].to_numpy(), k["s1_row"].to_numpy(), load_pred(tag, "p1", part, c), k["is_s3"].to_numpy())


def _stage1_matrix(f, df, sel=None, cols=None) -> np.ndarray:
    """Stage-1 feature matrix of feature file f (df: its frame with the base FEATURES), columns in the order of `cols`
    (default S1_FEATURES; a booster's own list when scoring): xfeats / nfeats come from their row-aligned side files."""
    cols = list(cols or S1_FEATURES)
    frames = [df.select([c for c in cols if c in FEATURES])]
    for names, path in ((X_FEATURES, xfeat_path(f)), (N_FEATURES, nfeat_path(f))):
        want = [c for c in cols if c in names]
        if want:
            side = pl.read_parquet(path, columns=want)
            assert side.height == df.height, f"{path} not aligned with {f}"
            frames.append(side)
    x = pl.concat(frames, how="horizontal").select(cols)
    return (x if sel is None else x.filter(pl.Series(sel))).to_numpy()


def stage1_cols(booster, stage) -> list:
    """The stage-1 feature list a booster was trained on (stage 2 appends CTX2_FEATURES)."""
    names = booster.feature_name()
    return names[:len(names) - len(CTX2_FEATURES)] if stage == 2 else names


def load_train(pool, cs, stage, tag) -> dict:
    """Training rows of `pool`: pairs of TRAIN_QUERY_FRAC of its queries, 10% of those queries held out for early
    stopping (same draw as before: seed 43 over the sorted training queries that have candidates).
    -> dict(X float32 C-order with the holdout rows last, y, n_tr, cols)."""
    t0 = time.time()
    cols = S1_FEATURES if stage == 1 else S2_FEATURES
    tq = _train_queries(pool, cs)
    files = {c: feat_files(pool, c) for c in cs}
    qs = {c: [pl.read_parquet(f, columns=["q_row"])["q_row"].to_numpy() for f in files[c]] for c in cs}
    size = int(max(q.max() for c in cs for q in qs[c])) + 1
    in_tq = _mask(tq[tq < size], size)
    uq = np.unique(np.concatenate([q[in_tq[q]] for c in cs for q in qs[c]]))
    rng = np.random.default_rng(C.SEED + 1)
    hold = _mask(uq[rng.random(len(uq)) < C.HOLDOUT_FRAC], size)
    lex_off = None
    if C.USE_XFEATS and C.MASK_LEX_FRAC > 0:  # feature dropout of the lexicon block, per query
        drop = np.random.default_rng(C.SEED + 2).random(len(uq)) < C.MASK_LEX_FRAC
        if not C.MASK_HOLDOUT:
            drop &= ~hold[uq]
        lex_off = _mask(uq[drop], size)
        lex_cols = [cols.index(f) for f in LEX_FEATURES]
    n_tr = sum(int((in_tq[q] & ~hold[q]).sum()) for c in cs for q in qs[c])
    n_va = sum(int((in_tq[q] & hold[q]).sum()) for c in cs for q in qs[c])
    X = np.empty((n_tr + n_va, len(cols)), np.float16 if C.X16 else np.float32)
    y = np.empty(n_tr + n_va, np.float32)
    a, b = 0, n_tr
    for c in cs:
        cx = ctx2_for(tag, pool, c) if stage == 2 else None
        off = 0
        for f, q in zip(files[c], qs[c]):
            sel = in_tq[q]
            if sel.any():
                df = pl.read_parquet(f, columns=["s1_row", "y"] + FEATURES)
                parts = [_stage1_matrix(f, df, sel)]
                if cx is not None:
                    parts.append(cx.rows(off, q, df["s1_row"].to_numpy(), df["is_s3"].to_numpy())[sel])
                x = np.hstack(parts) if len(parts) > 1 else parts[0]
                yy = df["y"].to_numpy()[sel]
                if lex_off is not None:
                    x[np.ix_(lex_off[q[sel]], lex_cols)] = np.nan
                h = hold[q[sel]]
                k1, k2 = int((~h).sum()), int(h.sum())
                if C.X16:
                    x = np.clip(x, -65504, 65504)  # NaN passes through
                X[a:a + k1], y[a:a + k1] = x[~h], yy[~h]
                X[b:b + k2], y[b:b + k2] = x[h], yy[h]
                a, b = a + k1, b + k2
                del df, parts, x
            off += len(q)
        del cx
        gc.collect()
    assert a == n_tr and b == n_tr + n_va
    print(f"[load] {pool} stage {stage}: {n_tr:,} train + {n_va:,} holdout rows x {len(cols)} ({X.nbytes / 2**30:.1f} GB) "
          f"in {time.time() - t0:.0f}s" + (f"; lexicon blanked for {C.MASK_LEX_FRAC:.0%} of queries" if lex_off is not None else ""))
    return dict(X=X, y=y, n_tr=n_tr, cols=cols)


# ---------------------------------------------------------------- fit / predict
def fit(data: dict, params, rounds, label) -> lgb.Booster:
    """data from load_train. X is popped from it so it is freed as soon as LightGBM has binned it."""
    t0 = time.time()
    X, y, n_tr, cols = data.pop("X"), data.pop("y"), data["n_tr"], data["cols"]
    p = dict(params, num_threads=C.N_JOBS)
    wrap = _Rows if X.dtype == np.float16 else (lambda a: a)
    dtr = lgb.Dataset(wrap(X[:n_tr]), y[:n_tr], feature_name=list(cols), params=p, free_raw_data=True).construct()
    dva = lgb.Dataset(wrap(X[n_tr:]), y[n_tr:], reference=dtr, params=p, free_raw_data=True).construct()
    n, pos = len(y), float(y.mean())
    del X, y
    gc.collect()
    b = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["holdout"],
                  callbacks=[lgb.early_stopping(C.EARLY_STOP, verbose=False), lgb.log_evaluation(250)])
    print(f"[fit] {label}: rows {n:,} (pos {pos:.4f}), best_iter {b.best_iteration}, "
          f"holdout logloss {b.best_score['holdout']['binary_logloss']:.5f}, {time.time() - t0:.0f}s")
    imp = sorted(zip(cols, b.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(v for _, v in imp) or 1
    print("  top gain: " + ", ".join(f"{k} {v / tot:.3f}" for k, v in imp[:15]))
    return b


def predict_part(boosters, part, c, stage, tag, out_kind) -> np.ndarray:
    """Mean of `boosters` over every pair of (part, c), file by file -> OOF_DIR/{tag}/{out_kind}_{part}_{c}.npy."""
    t0 = time.time()
    cols = stage1_cols(boosters[0], stage)
    assert all(b.feature_name() == boosters[0].feature_name() for b in boosters), "boosters with different features"
    cx = ctx2_for(tag, part, c) if stage == 2 else None
    outs, off = [], 0
    for f in feat_files(part, c):
        df = pl.read_parquet(f, columns=["q_row", "s1_row"] + FEATURES)
        x = _stage1_matrix(f, df, cols=cols)
        if cx is not None:
            x = np.hstack([x, cx.rows(off, df["q_row"].to_numpy(), df["s1_row"].to_numpy(), df["is_s3"].to_numpy())])
        x = np.ascontiguousarray(x, dtype=np.float32)
        outs.append(np.mean([b.predict(x, num_iteration=b.best_iteration, num_threads=C.N_JOBS) for b in boosters],
                            axis=0).astype(np.float32))
        off += df.height
        del df, x
    p = np.concatenate(outs)
    path = _oof(tag, out_kind, part, c)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, p)
    print(f"[predict] {tag} {out_kind} {part} {c}: {len(p):,} pairs in {time.time() - t0:.0f}s")
    return p


def load_booster(tag, stage, pool):
    return lgb.Booster(model_file=str(_model_path(tag, stage, pool)))


def _save(b, tag, stage, pool):
    p = _model_path(tag, stage, pool)
    p.parent.mkdir(parents=True, exist_ok=True)
    b.save_model(str(p), num_iteration=b.best_iteration)


# ---------------------------------------------------------------- evaluation
def partition_top(tag, kind, part, c):
    """-> (argmax row per query with ok/has_true, true candidate pairs) for one scored partition."""
    k = load_cols(part, c, ["q_row", "s1_row", "y"])
    p = load_pred(tag, kind, part, c)
    assert len(p) == k.height, f"{kind} {part} {c}: {len(p):,} predictions for {k.height:,} pairs"
    top = argmax_rows(k["q_row"].to_numpy(), k["s1_row"].to_numpy(), p, k["y"].to_numpy())
    return top, k.filter(pl.col("y") == 1).select("s1_row", "q_row")


def evaluate(tag, kind, pools=("P0", "P1"), cs=None, T=None, label="", decomp=True):
    """Macro F0.5 of the decision policy on OOF predictions over the given pools/countries."""
    cs = cs or countries("train")
    gt = io.gt_rows()
    tops, blocked, truths, unis = [], [], [], []
    for part in pools:
        for c in cs:
            top, bi = partition_top(tag, kind, part, c)
            _, q_rows = load_partition(part, c)
            uni = load_eval_s1(part, c)
            truths.append(gt.filter(_in(gt["s1_row"], uni) & _in(gt["q_row"], q_rows)))
            tops.append(top)
            blocked.append(bi)
            unis.append(uni)
    top, truth, uni, blocked_in = assign(pl.concat(tops)), pl.concat(truths), np.concatenate(unis), pl.concat(blocked)
    del tops, blocked
    if T is None:
        T = load_thresholds(tag)
    pred = decide(top, *T)
    if decomp:
        out, _ = loss_decomposition(top, blocked_in, pred, truth, uni, label=f"{label} T={T}")
        return out["F"], top, truth, uni
    from .metrics import macro_f05
    f = macro_f05(uni, pred["s1_row"].to_numpy(), pred["q_row"].to_numpy(), truth["s1_row"].to_numpy(), truth["q_row"].to_numpy())
    print(f"[eval] {label} T={T}: F={f:.5f}")
    return f, top, truth, uni


def _single_best(tag, kind, pools, cs):
    _, top, truth, uni = evaluate(tag, kind, pools, cs, T=(0.76, 0.76), decomp=False, label=f"{kind} {tag}")
    (fb, tb), res = tune_thresholds(top, truth, uni, single=True)
    print(f"[tune] {tag} {kind} single threshold best F={fb:.5f} at {tb[0]}")
    return (fb, tb), res


# ---------------------------------------------------------------- orchestration
def train_stage1(tier0=False, tag="main", cs=None, pools=None, reuse=False):
    """Fit stage 1 on each pool (P0 only for tier 0) and score the other pool out of fold. reuse: keep an existing model
    (its OOF is still produced if missing, e.g. once the other pool's features exist)."""
    cs = cs or countries("train")
    pools = pools or (("P0",) if tier0 else ("P0", "P1"))
    for pool in pools:
        other = OTHER[pool]
        if reuse and _model_path(tag, 1, pool).exists():
            print(f"[train1] reuse {_model_path(tag, 1, pool)}")
            b = load_booster(tag, 1, pool)
        else:
            b = fit(load_train(pool, cs, 1, tag), C.LGB1, C.LGB1_ROUNDS, f"stage1 {tag} train {pool} {cs}")
            _save(b, tag, 1, pool)
            for c in cs:
                _oof(tag, "p1", other, c).unlink(missing_ok=True)  # stale OOF of an older model
        for c in cs:
            if not _oof(tag, "p1", other, c).exists() and feat_files(other, c):
                predict_part([b], other, c, 1, tag, "p1")
        del b
        gc.collect()
    eval_pools = tuple(OTHER[p] for p in pools if all(_oof(tag, "p1", OTHER[p], c).exists() for c in cs))
    if not eval_pools:
        print("[train1] no out-of-fold pool has features yet; evaluation skipped")
        return None
    best, _ = _single_best(tag, "p1", eval_pools, cs)
    f76, *_ = evaluate(tag, "p1", eval_pools, cs, T=(0.76, 0.76), label=f"stage1 {tag} OOF {eval_pools}")
    return f76, best


def train_stage2(tag="main", cs=None):
    cs = cs or countries("train")
    for pool in ("P0", "P1"):
        b = fit(load_train(pool, cs, 2, tag), C.LGB2, C.LGB2_ROUNDS, f"stage2 {tag} train {pool} {cs}")
        _save(b, tag, 2, pool)
        for c in cs:
            predict_part([b], OTHER[pool], c, 2, tag, "p2")
        del b
        gc.collect()
    _single_best(tag, "p1", ("P0", "P1"), cs)
    return _single_best(tag, "p2", ("P0", "P1"), cs)


def tune_from_oof(tag="main"):
    cs = countries("train")
    kind = "p2" if _oof(tag, "p2", "P0", cs[0]).exists() else "p1"
    pools = ("P0", "P1") if _oof(tag, kind, "P0", cs[0]).exists() else ("P1",)
    _, top, truth, uni = evaluate(tag, kind, pools, cs, T=(0.76, 0.76), decomp=False, label=f"{kind} pre-tune")
    (f1, t1), _ = tune_thresholds(top, truth, uni, single=True)
    (f2, t2), res = tune_thresholds(top, truth, uni)
    del top
    print(f"[tune] {tag} {kind} on {pools}: single-threshold F={f1:.5f} at {t1[0]}; (T1,T2) F={f2:.5f} at {t2}")
    save_thresholds(*t2, extra=dict(kind=kind, F=f2, F_single=f1, T_single=t1[0]), tag=tag)
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
            if tgt != src:
                for pool in ("P0", "P1"):
                    predict_part([load_booster(tag, 1, OTHER[pool])], pool, tgt, 1, tag, "p1")
                    predict_part([load_booster(tag, 2, OTHER[pool])], pool, tgt, 2, tag, "p2")
            res[(src, tgt)] = evaluate(tag, "p2", ("P0", "P1"), [tgt], T=T, decomp=False, label=f"LOCO train {src} eval {tgt}")[0]
    for tgt in cs:
        for src in cs:
            if src != tgt:
                d = res[(src, tgt)] - res[(tgt, tgt)]
                print(f"[loco] eval {tgt}: in-country {res[(tgt, tgt)]:.5f}  trained on {src} {res[(src, tgt)]:.5f}  drop {100 * d:+.2f}pp")
    (C.MODEL_DIR / "loco.json").write_text(json.dumps({f"{a}->{b}": v for (a, b), v in res.items()}, indent=1))
    return res
