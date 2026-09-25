"""TSV IO (tab-separated, quoting disabled), GT parsing, normalized-frame loading, id <-> row maps.

Row identity convention used everywhere downstream:
  s1_row = position in {split}_source1.tsv
  q_row  = position in the concatenation [source2 rows ; source3 rows] of the split
Both are int32 and are mapped back to entity ids exactly once, in submit.py.
"""
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def raw_path(split: str, src: int) -> Path:
    return C.DATA_DIR / split / f"{split}_source{src}.tsv"


def gt_path() -> Path:
    return C.DATA_DIR / "train" / "train_ground_truth.tsv"


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def read_source(split: str, src: int) -> pl.DataFrame:
    df = read_tsv(raw_path(split, src))
    assert df.columns == SOURCE_COLS, df.columns
    return df.with_columns(pl.col(c).fill_null("") for c in SOURCE_COLS)


def read_queries_raw(split: str) -> pl.DataFrame:
    """Raw S2 then S3 rows (q_row order)."""
    return pl.concat([read_source(split, 2), read_source(split, 3)])


def read_gt_pairs() -> pl.DataFrame:
    """Train GT exploded to (s1_id, cand_id); singletons contribute no rows."""
    gt = read_tsv(gt_path())
    assert gt.columns == ["source1_entity_id", "matched_entity_ids"], gt.columns
    return (
        gt.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"})
        .with_columns(pl.col("cand_id").fill_null("").str.split(","))
        .explode("cand_id")
        .with_columns(pl.col("cand_id").str.strip_chars())
        .filter(pl.col("cand_id") != "")
    )


def norm_path(split: str, src: int) -> Path:
    return C.NORM_DIR / f"{split}_s{src}.parquet"


def load_norm(split: str, which: str, columns=None) -> pl.DataFrame:
    """which='s1' -> source1 frame; which='q' -> source2+source3 concatenated with is_s3 (q_row order)."""
    if which == "s1":
        return pl.read_parquet(norm_path(split, 1), columns=columns)
    parts = []
    for src in (2, 3):
        df = pl.read_parquet(norm_path(split, src), columns=columns)
        parts.append(df.with_columns(pl.lit(src == 3, dtype=pl.Int8).alias("is_s3")))
    return pl.concat(parts)


def gt_rows() -> pl.DataFrame:
    """Train GT as int32 (s1_row, q_row) keys."""
    s1 = pl.read_parquet(norm_path("train", 1), columns=["entity_id"]).with_row_index("s1_row")
    q = load_norm("train", "q", columns=["entity_id"]).with_row_index("q_row")
    gt = read_gt_pairs()
    out = (
        gt.join(s1.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
        .join(q.rename({"entity_id": "cand_id"}), on="cand_id", how="left")
    )
    assert out["s1_row"].null_count() == 0 and out["q_row"].null_count() == 0, "GT ids missing from sources"
    return out.select(pl.col("s1_row").cast(pl.Int32), pl.col("q_row").cast(pl.Int32))


def write_id_lists(path, header, s1_ids: pl.Series, pairs: pl.DataFrame):
    """One row per s1 id in the given order; pairs = (s1_id, cand_id). Ids sorted, comma-joined, empty if none."""
    agg = (
        pairs.unique()
        .group_by("s1_id")
        .agg(pl.col("cand_id").sort().str.join(",").alias("ids"))
    )
    out = (
        pl.DataFrame({"s1_id": s1_ids})
        .join(agg, on="s1_id", how="left", maintain_order="left")
        .with_columns(pl.col("ids").fill_null(""))
    )
    assert out.height == len(s1_ids)
    out.columns = list(header)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", null_value="")
    return out


def save_np(path, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
