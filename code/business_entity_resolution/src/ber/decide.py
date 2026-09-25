"""Section 7.9 decision policy: per-query argmax (ties -> no link), then per S1 keep first if p >= T1, later if p >= T2.
Also the (T1, T2) grid tuning and the loss decomposition report."""
import json

import numpy as np
import polars as pl

from . import config as C
from .metrics import bucket_report, entity_counts, f05


def argmax_assign(df: pl.DataFrame, pcol="p") -> pl.DataFrame:
    """-> one row per query with a non-tied argmax: q_row, s1_row, p, rk (rank of the query within its S1 by p)."""
    top = (df.select("q_row", "s1_row", pl.col(pcol).alias("p"))
           .sort(["q_row", "p"], descending=[False, True])
           .group_by("q_row", maintain_order=True)
           .agg(pl.col("s1_row").first(), pl.col("p").first(), pl.col("p").get(1, null_on_oob=True).alias("p2nd")))
    top = top.filter(pl.col("p2nd").is_null() | ((pl.col("p") - pl.col("p2nd")) >= C.TIE_EPS)).drop("p2nd")
    return top.with_columns(rk=pl.col("p").rank("ordinal", descending=True).over("s1_row"))


def decide(df: pl.DataFrame, T1=C.T1_DEFAULT, T2=C.T2_DEFAULT, pcol="p") -> pl.DataFrame:
    top = argmax_assign(df, pcol)
    keep = pl.when(pl.col("rk") == 1).then(pl.col("p") >= T1).otherwise(pl.col("p") >= T2)
    return top.filter(keep).select("s1_row", "q_row")


def _grid(lo, hi):
    return np.round(np.arange(lo, hi + 1e-9, C.GRID_STEP), 4)


def tune_thresholds(df: pl.DataFrame, truth: pl.DataFrame, universe, pcol="p", single=False):
    """Grid (T1, T2) on pooled OOF. df: all scored pairs (q_row, s1_row, p); truth: (s1_row, q_row) true pairs."""
    universe = np.asarray(universe)
    top = argmax_assign(df, pcol).join(truth.with_columns(ok=pl.lit(True)), on=["s1_row", "q_row"], how="left")
    top = top.with_columns(pl.col("ok").fill_null(False))
    size = int(max(universe.max(), top["s1_row"].max() or 0, truth["s1_row"].max() or 0)) + 1
    lut = np.full(size, -1, np.int64)
    lut[universe] = np.arange(len(universe))
    n = len(universe)
    top = top.filter(pl.Series(lut[top["s1_row"].to_numpy()] >= 0))  # links to S1 outside the evaluated set are ignored
    tl = lut[truth["s1_row"].to_numpy()]
    nt = np.bincount(tl[tl >= 0], minlength=n)
    s = lut[top["s1_row"].to_numpy()]
    p = top["p"].to_numpy()
    ok = top["ok"].to_numpy()
    first = top["rk"].to_numpy() == 1
    fs, fp, fok = s[first], p[first], ok[first]
    ls, lp, lok = s[~first], p[~first], ok[~first]
    if single:
        grid = [(t, t) for t in _grid(*C.T1_GRID)]
    else:
        grid = [(a, b) for a in _grid(*C.T1_GRID) for b in _grid(*C.T2_GRID)]
    later = {}
    for t2 in sorted({b for _, b in grid}):
        m = lp >= t2
        later[t2] = (np.bincount(ls[m], minlength=n), np.bincount(ls[m & lok], minlength=n))
    best = (-1, None)
    res = []
    for t1, t2 in grid:
        m = fp >= t1
        npred = np.bincount(fs[m], minlength=n) + later[t2][0]
        tp = np.bincount(fs[m & fok], minlength=n) + later[t2][1]
        f = f05(nt, npred, tp).mean()
        res.append((t1, t2, f))
        if f > best[0]:
            best = (f, (float(t1), float(t2)))
    return best, res


def loss_decomposition(df: pl.DataFrame, pred: pl.DataFrame, truth: pl.DataFrame, universe, pcol="p", label=""):
    """df: all scored pairs (q_row, s1_row, p, y); pred: kept links (s1_row, q_row); truth: true pairs of the partition."""
    pred = pred.filter(pl.col("s1_row").is_in(np.asarray(universe)))
    ts, tq = truth["s1_row"].to_numpy(), truth["q_row"].to_numpy()
    ps, pq = pred["s1_row"].to_numpy(), pred["q_row"].to_numpy()
    counts = entity_counts(universe, ps, pq, ts, tq)
    f = f05(*counts).mean()
    rep = bucket_report(*counts, label=f"{label} F={f:.5f}")
    tp_pairs = pred.join(truth, on=["s1_row", "q_row"], how="semi")
    f_nofp = f05(*entity_counts(universe, tp_pairs["s1_row"].to_numpy(), tp_pairs["q_row"].to_numpy(), ts, tq)).mean()
    blocked_in = df.filter(pl.col("y") == 1).select("s1_row", "q_row")
    allfn = pl.concat([pred, blocked_in]).unique()
    f_allfn = f05(*entity_counts(universe, allfn["s1_row"].to_numpy(), allfn["q_row"].to_numpy(), ts, tq)).mean()
    fp = pred.join(truth, on=["s1_row", "q_row"], how="anti")
    q_has_true = df.group_by("q_row").agg(pl.col("y").max().alias("has_true"))
    fp = fp.join(q_has_true, on="q_row", how="left")
    fn = truth.join(pred, on=["s1_row", "q_row"], how="anti")
    am = argmax_assign(df, pcol).select("q_row", pl.col("s1_row").alias("am_s1"))
    fn = (fn.join(df.select("q_row", "s1_row").with_columns(inc=pl.lit(True)), on=["q_row", "s1_row"], how="left")
          .join(am, on="q_row", how="left"))
    n_fn_block = int(fn["inc"].is_null().sum())
    fn_in = fn.filter(pl.col("inc").is_not_null())
    n_fn_thr = int((fn_in["am_s1"] == fn_in["s1_row"]).sum())
    out = dict(F=float(f), F_noFP=float(f_nofp), F_allFN_blocked_in=float(f_allfn),
               n_fp=fp.height, fp_q_has_true=int(fp["has_true"].fill_null(0).sum()),
               fp_q_no_true=int((fp["has_true"].fill_null(0) == 0).sum()),
               n_fn=fn.height, fn_blocked_out=n_fn_block, fn_argmax_ok_below_thr=n_fn_thr,
               fn_other_s1_won_or_tie=fn_in.height - n_fn_thr)
    print(f"[loss] {label}: " + "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v:,}" for k, v in out.items()))
    return out, rep


def save_thresholds(t1, t2, extra=None):
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODEL_DIR / "thresholds.json").write_text(json.dumps(dict(T1=t1, T2=t2, **(extra or {})), indent=1))


def load_thresholds():
    p = C.MODEL_DIR / "thresholds.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d["T1"], d["T2"]
    return C.T1_DEFAULT, C.T2_DEFAULT


def tune():
    from .model import tune_from_oof
    tune_from_oof()
