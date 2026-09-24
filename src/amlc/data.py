"""Loading the official TSVs + building dev-sized samples of the training data.

All files are TAB separated with no quoting (names/addresses contain commas and quotes),
so always read through these helpers.

Pair frames used everywhere have two columns: s1_id, cand_id (one row per S1 -> S2/S3 link).
"""
import numpy as np
import polars as pl

from .config import GT_MATCH_COL, GT_S1_COL, ID_COL, RAW, SEED


def read_tsv(path, **kw) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, **kw)


def load_source(split: str, source: int, n_rows: int | None = None) -> pl.DataFrame:
    """Columns: entity_id, business_name, business_address, country (all Utf8, nulls allowed)."""
    return read_tsv(RAW / split / f"{split}_source{source}.tsv", n_rows=n_rows)


def load_ground_truth() -> pl.DataFrame:
    """Raw GT: source1_entity_id, matched_entity_ids (comma list, null when singleton)."""
    return read_tsv(RAW / "train" / "train_ground_truth.tsv")


def explode_ids(df: pl.DataFrame, list_col: str, s1_col: str = GT_S1_COL) -> pl.DataFrame:
    """Comma-list frame -> pair frame (s1_id, cand_id). Singletons produce no rows."""
    return (
        df.select(pl.col(s1_col).alias("s1_id"), pl.col(list_col).fill_null("").str.split(",").alias("cand_id"))
        .explode("cand_id")
        .filter(pl.col("cand_id").is_not_null() & (pl.col("cand_id") != ""))
    )


def gt_pairs() -> pl.DataFrame:
    return explode_ids(load_ground_truth(), GT_MATCH_COL)


def dev_sample(frac: float = 0.05, seed: int = SEED) -> dict:
    """A density-preserving slice of the training data for fast iteration.

    Takes `frac` of the S1 entities, all of their true S2/S3 matches, and the same `frac`
    of the S2/S3 records that match nothing (distractors), so the ratio of true matches
    to distractors is the same as in the full data. Returns dict(s1, s2, s3, gt_pairs, s1_ids).

    Note: a smaller S1 pool is less confusable than the full test set, so dev scores are
    somewhat optimistic; compare runs against each other, not against the leaderboard.
    """
    rng = np.random.default_rng(seed)
    s1 = load_source("train", 1)
    keep_s1 = s1.filter(pl.Series(rng.random(s1.height) < frac))
    all_pairs = gt_pairs()
    pairs = all_pairs.join(keep_s1.select(pl.col(ID_COL).alias("s1_id")), on="s1_id", how="semi")
    all_matched = all_pairs.select("cand_id")
    out = {"s1": keep_s1, "gt_pairs": pairs, "s1_ids": keep_s1[ID_COL]}
    for src in (2, 3):
        df = load_source("train", src)
        matched_here = df.join(pairs.select(pl.col("cand_id").alias(ID_COL)), on=ID_COL, how="semi")
        distract = df.join(all_matched.select(pl.col("cand_id").alias(ID_COL)), on=ID_COL, how="anti")
        distract = distract.filter(pl.Series(rng.random(distract.height) < frac))
        out[f"s{src}"] = pl.concat([matched_here, distract])
    print(f"dev sample: S1={out['s1'].height:,} S2={out['s2'].height:,} S3={out['s3'].height:,} "
          f"true pairs={pairs.height:,}")
    return out
