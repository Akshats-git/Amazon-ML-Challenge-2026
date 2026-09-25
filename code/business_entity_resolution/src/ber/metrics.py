"""Macro F0.5 over S1 entities (singletons included), per-bucket report, blocking report."""
import numpy as np
import polars as pl


def f05(n_true, n_pred, tp):
    n_true = np.asarray(n_true, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, 0.0)
    both = (n_true > 0) & (n_pred > 0)
    f[both] = 1.25 * tp[both] / (0.25 * n_true[both] + n_pred[both])
    return f


def _key(s1, q):
    return (np.asarray(s1, dtype=np.int64) << 32) | np.asarray(q, dtype=np.int64)


def entity_counts(universe, pred_s1, pred_q, true_s1, true_q):
    """Per-entity (n_true, n_pred, tp) for S1 rows in `universe`. Pairs whose S1 is outside it are ignored."""
    universe = np.asarray(universe, dtype=np.int64)
    lut = np.full(int(max(universe.max(initial=0), np.max(pred_s1, initial=0), np.max(true_s1, initial=0))) + 1, -1, np.int64)
    lut[universe] = np.arange(len(universe))
    n = len(universe)
    pred_s1, pred_q, true_s1, true_q = map(np.asarray, (pred_s1, pred_q, true_s1, true_q))
    pk = np.unique(_key(pred_s1, pred_q))
    tk = np.unique(_key(true_s1, true_q))
    ps, ts = lut[pk >> 32], lut[tk >> 32]
    pk, tk = pk[ps >= 0], tk[ts >= 0]
    ps, ts = ps[ps >= 0], ts[ts >= 0]
    n_pred = np.bincount(ps, minlength=n)
    n_true = np.bincount(ts, minlength=n)
    hit = np.isin(pk, tk, assume_unique=True)
    tp = np.bincount(ps[hit], minlength=n)
    return n_true, n_pred, tp


def macro_f05(universe, pred_s1, pred_q, true_s1, true_q):
    return float(f05(*entity_counts(universe, pred_s1, pred_q, true_s1, true_q)).mean())


def macro_f05_ids(pred: dict, truth: dict) -> float:
    """String-id version for unit tests: dict s1_id -> iterable of ids; universe = truth keys."""
    s1s = list(truth)
    ids = {}
    enc = lambda x: ids.setdefault(x, len(ids))  # noqa: E731
    si = {s: i for i, s in enumerate(s1s)}
    ps, pq, ts, tq = [], [], [], []
    for s, v in pred.items():
        for x in set(v):
            ps.append(si[s]); pq.append(enc(x))
    for s, v in truth.items():
        for x in set(v):
            ts.append(si[s]); tq.append(enc(x))
    return macro_f05(np.arange(len(s1s)), ps, pq, ts, tq)


def bucket_report(n_true, n_pred, tp, label="") -> pl.DataFrame:
    f = f05(n_true, n_pred, tp)
    bucket = np.select([n_true == 0, n_true == 1, n_true <= 3], ["0", "1", "2-3"], "4+")
    df = pl.DataFrame({"bucket": bucket, "f": f, "nt": n_true, "np": n_pred, "tp": tp})
    n = len(f)
    rep = (
        df.group_by("bucket")
        .agg(
            pl.len().alias("n_s1"),
            pl.col("f").mean().alias("F"),
            (pl.col("tp").sum() / pl.col("np").sum()).alias("prec"),
            (pl.col("tp").sum() / pl.col("nt").sum()).alias("rec"),
            ((1 - pl.col("f")).sum() / n).alias("loss"),
        )
        .sort("bucket")
        .with_columns((pl.col("n_s1") / n).alias("share"))
    )
    total = pl.DataFrame({
        "bucket": ["all"], "n_s1": [n], "F": [f.mean()],
        "prec": [tp.sum() / max(n_pred.sum(), 1)], "rec": [tp.sum() / max(n_true.sum(), 1)],
        "loss": [1 - f.mean()], "share": [1.0],
    }).cast(rep.schema)
    out = pl.concat([rep, total])
    if label:
        print(f"--- {label}")
    with pl.Config(tbl_rows=20, float_precision=5):
        print(out)
    return out


def blocking_report(cands: pl.DataFrame, truth: pl.DataFrame, universe, ks=(1, 3, 5, 10), label="") -> dict:
    """cands: (q_row, s1_row, blk_rank); truth: (s1_row, q_row) restricted to the partition.
    Reports pair recall@k, oracle F (perfect matcher restricted to candidates), mean candidates per query."""
    hit = truth.join(cands.select("q_row", "s1_row", "blk_rank"), on=["q_row", "s1_row"], how="left")
    rk = hit["blk_rank"].fill_null(10_000).to_numpy()
    out = {f"R@{k}": float((rk <= k).mean()) for k in ks}
    found = hit.filter(pl.col("blk_rank").is_not_null())
    ts, tq = truth["s1_row"].to_numpy(), truth["q_row"].to_numpy()
    out["oracle_F"] = macro_f05(universe, found["s1_row"].to_numpy(), found["q_row"].to_numpy(), ts, tq)
    nq = cands["q_row"].n_unique()
    out["pairs"] = cands.height
    out["cands_per_query"] = cands.height / max(nq, 1)
    out["n_true"] = truth.height
    if label:
        print(f"[blocking] {label}: " + "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()))
    return out
