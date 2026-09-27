"""Section 7.9 decision policy: per-query argmax (ties -> no link), then per S1 keep first if p >= T1, later if p >= T2.
Also the (T1, T2) grid tuning and the loss decomposition report.

Scored pairs arrive as numpy arrays grouped by query (feature-file order); argmax_rows reduces them to one row per
query, and everything downstream works on those (~13M rows for both pools instead of 130M pairs)."""
import json

import numpy as np
import polars as pl

from . import config as C
from .metrics import bucket_report, entity_counts, f05


def argmax_rows(q_row, s1_row, p, y=None) -> pl.DataFrame:
    """One row per query: q_row, s1_row (first argmax), p, p2nd (runner-up, -inf if single candidate), and with labels
    ok (the argmax pair is true) and has_true (the query has a true candidate). Rows must be grouped by query."""
    p = np.asarray(p, np.float32)
    new = np.r_[True, q_row[1:] != q_row[:-1]]
    starts = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    idx = np.flatnonzero(p == np.maximum.reduceat(p, starts)[gid])
    am = idx[np.r_[True, gid[idx][1:] != gid[idx][:-1]]]
    del gid, idx
    rest = p.copy()
    rest[am] = -np.inf
    out = {"q_row": q_row[starts], "s1_row": s1_row[am], "p": p[am], "p2nd": np.maximum.reduceat(rest, starts)}
    if y is not None:
        out["ok"] = y[am] == 1
        out["has_true"] = np.maximum.reduceat(y, starts) == 1
    return pl.DataFrame(out)


def assign(top: pl.DataFrame) -> pl.DataFrame:
    """Drop tied queries (runner-up within TIE_EPS) and rank the rest within their S1 by p (rk 1 = best)."""
    top = top.filter((pl.col("p") - pl.col("p2nd")) >= C.TIE_EPS).drop("p2nd")
    return top.with_columns(rk=pl.col("p").rank("ordinal", descending=True).over("s1_row"))


def decide(top: pl.DataFrame, T1=C.T1_DEFAULT, T2=C.T2_DEFAULT, n_s2=None) -> pl.DataFrame:
    """top from assign(); -> kept links (s1_row, q_row). n_s2 (test: number of S2 rows, q_row >= n_s2 is S3): also
    enforce the generator's per-source caps (<= CAP_S2 S2 and <= CAP_S3 S3 matches per S1), keeping the highest p."""
    keep = pl.when(pl.col("rk") == 1).then(pl.col("p") >= T1).otherwise(pl.col("p") >= T2)
    out = top.filter(keep)
    if n_s2 is not None:
        n = out.height
        out = (out.with_columns(s3=pl.col("q_row") >= n_s2)
               .with_columns(r=pl.col("p").rank("ordinal", descending=True).over("s1_row", "s3"))
               .filter(pl.col("r") <= pl.when(pl.col("s3")).then(C.CAP_S3).otherwise(C.CAP_S2)))
        print(f"[decide] per-source caps ({C.CAP_S2} S2 / {C.CAP_S3} S3 per S1) dropped {n - out.height:,} of {n:,} links")
    return out.select("s1_row", "q_row")


def _in(values, rows) -> pl.Series:
    """Boolean mask: values in rows (both int arrays)."""
    values, rows = np.asarray(values), np.asarray(rows)
    m = np.zeros(int(max(values.max(initial=0), rows.max(initial=0))) + 1, bool)
    m[rows] = True
    return pl.Series(m[values])


def _grid(lo, hi):
    return np.round(np.arange(lo, hi + 1e-9, C.GRID_STEP), 4)


def tune_thresholds(top: pl.DataFrame, truth: pl.DataFrame, universe, single=False):
    """Grid (T1, T2) on pooled OOF. top: assign() output with ok; truth: (s1_row, q_row) true pairs."""
    universe = np.asarray(universe)
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


def loss_decomposition(top: pl.DataFrame, blocked_in: pl.DataFrame, pred: pl.DataFrame, truth: pl.DataFrame, universe, label=""):
    """top: assign() output with ok/has_true; blocked_in: candidate pairs that are true (y == 1);
    pred: kept links (s1_row, q_row); truth: true pairs of the evaluated partitions."""
    pred = pred.filter(_in(pred["s1_row"], universe))
    ts, tq = truth["s1_row"].to_numpy(), truth["q_row"].to_numpy()
    ps, pq = pred["s1_row"].to_numpy(), pred["q_row"].to_numpy()
    counts = entity_counts(universe, ps, pq, ts, tq)
    f = f05(*counts).mean()
    rep = bucket_report(*counts, label=f"{label} F={f:.5f}")
    tp_pairs = pred.join(truth, on=["s1_row", "q_row"], how="semi")
    f_nofp = f05(*entity_counts(universe, tp_pairs["s1_row"].to_numpy(), tp_pairs["q_row"].to_numpy(), ts, tq)).mean()
    allfn = pl.concat([pred, blocked_in.select("s1_row", "q_row")]).unique()
    f_allfn = f05(*entity_counts(universe, allfn["s1_row"].to_numpy(), allfn["q_row"].to_numpy(), ts, tq)).mean()
    fp = pred.join(truth, on=["s1_row", "q_row"], how="anti").join(top.select("q_row", "has_true"), on="q_row", how="left")
    fn = truth.join(pred, on=["s1_row", "q_row"], how="anti")
    fn = (fn.join(blocked_in.select("s1_row", "q_row").with_columns(inc=pl.lit(True)), on=["s1_row", "q_row"], how="left")
          .join(top.select("q_row", pl.col("s1_row").alias("am_s1")), on="q_row", how="left"))
    n_fn_block = int(fn["inc"].is_null().sum())
    fn_in = fn.filter(pl.col("inc").is_not_null())
    n_fn_thr = int((fn_in["am_s1"] == fn_in["s1_row"]).sum())
    out = dict(F=float(f), F_noFP=float(f_nofp), F_allFN_blocked_in=float(f_allfn),
               n_fp=fp.height, fp_q_has_true=int(fp["has_true"].fill_null(False).sum()),
               fp_q_no_true=int((~fp["has_true"].fill_null(False)).sum()),
               n_fn=fn.height, fn_blocked_out=n_fn_block, fn_argmax_ok_below_thr=n_fn_thr,
               fn_other_s1_won_or_tie=fn_in.height - n_fn_thr)
    print(f"[loss] {label}: " + "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v:,}" for k, v in out.items()))
    return out, rep


def save_thresholds(t1, t2, extra=None, tag=None):
    p = C.MODEL_DIR / (tag or C.TAG) / "thresholds.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(T1=t1, T2=t2, **(extra or {})), indent=1))


def load_thresholds(tag=None):
    tag = tag or C.TAG
    p = C.MODEL_DIR / tag / "thresholds.json"
    if not p.exists() and tag == "main":
        p = C.MODEL_DIR / "thresholds.json"  # R04 wrote it at the top level
    if p.exists():
        d = json.loads(p.read_text())
        return d["T1"], d["T2"]
    return C.T1_DEFAULT, C.T2_DEFAULT


def tune(tag=None):
    from .model import tune_from_oof
    tune_from_oof(tag or C.TAG)
