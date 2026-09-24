"""Official metric (macro F_0.5 per Source-1 entity) + blocking diagnostics.

All functions take *pair frames* with columns s1_id, cand_id (one row per link).

Official rules reproduced exactly:
  * per S1 entity: F_b = (1+b^2) * TP / (b^2 * |truth| + |pred|)
  * truth empty and pred empty  -> 1.0   (correct singleton)
  * truth empty and pred non-empty, or pred empty and truth non-empty -> 0.0
  * averaged over ALL S1 entities in the evaluation set (singletons included)
Check: pred {47,193,812}, truth {47,812} -> 1.25*2 / (0.25*2 + 3) = 0.714 (matches the PDF example).
"""
import polars as pl

from .config import BETA


def _counts(pairs: pl.DataFrame, name: str) -> pl.DataFrame:
    return pairs.unique(["s1_id", "cand_id"]).group_by("s1_id").len(name=name)


def per_entity_fbeta(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids, beta: float = BETA) -> pl.DataFrame:
    """One row per S1 id: n_pred, n_true, tp, precision, recall, f."""
    b2 = beta * beta
    tp = pred.unique(["s1_id", "cand_id"]).join(truth.unique(["s1_id", "cand_id"]), on=["s1_id", "cand_id"])
    base = pl.DataFrame({"s1_id": pl.Series(s1_ids, dtype=pl.Utf8)}).unique()
    df = (
        base.join(_counts(pred, "n_pred"), on="s1_id", how="left")
        .join(_counts(truth, "n_true"), on="s1_id", how="left")
        .join(tp.group_by("s1_id").len(name="tp"), on="s1_id", how="left")
        .fill_null(0)
    )
    return df.with_columns(
        precision=pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")),
        recall=pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")),
        f=pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
        .otherwise((1 + b2) * pl.col("tp") / (b2 * pl.col("n_true") + pl.col("n_pred"))),
    )


def fbeta_macro(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids, beta: float = BETA) -> float:
    return float(per_entity_fbeta(pred, truth, s1_ids, beta)["f"].mean())


def blocking_report(cands: pl.DataFrame, truth: pl.DataFrame, s1_ids, beta: float = BETA) -> dict:
    """How good is the candidate set, before any model?

    pair_recall   share of true links that survived blocking (the recall ceiling)
    oracle_f      macro F_0.5 of a perfect matcher restricted to these candidates
    pairs_per_s1  candidate volume (drives feature/model cost)
    """
    hit = truth.join(cands.select("s1_id", "cand_id").unique(), on=["s1_id", "cand_id"], how="semi")
    n_s1 = len(set(s1_ids))
    rep = {
        "pair_recall": hit.height / max(truth.height, 1),
        "oracle_f": fbeta_macro(hit, truth, s1_ids, beta),
        "n_pairs": cands.height,
        "pairs_per_s1": cands.height / max(n_s1, 1),
    }
    print("blocking: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v:,}" for k, v in rep.items()))
    return rep
