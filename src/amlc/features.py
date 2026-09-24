"""Pair features for the matcher (vectorised: rapidfuzz.cpdist + polars list ops).

Input: blocked pairs (qi, si, blk_score, blk_rank) + the normalised S1 / query frames.
Order: pairs = add_group_features(pairs) once, then pair_features(chunk, s1n, qn) per chunk.
No country one-hot on purpose: France appears only in test, so every feature is a
similarity or a structural signal that transfers across countries.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURES = [
    "blk_score", "blk_rank", "blk_gap", "q_n_cands", "s1_n_top1",
    "name_ratio", "name_tsort", "name_tset", "name_partial", "name_jw", "full_name_ratio",
    "name_tok_jacc", "legal_eq", "legal_both_empty", "name_len_diff",
    "addr_ratio", "addr_tset", "addr_partial_tset", "addr_missing",
    "num_jacc", "num_first_eq", "num_any", "is_s3", "non_latin_any",
]

_SCORERS = {
    "name_ratio": ("name_core", fuzz.ratio),
    "name_tsort": ("name_core", fuzz.token_sort_ratio),
    "name_tset": ("name_core", fuzz.token_set_ratio),
    "name_partial": ("name_core", fuzz.partial_ratio),
    "name_jw": ("name_core", JaroWinkler.normalized_similarity),
    "full_name_ratio": ("name_norm", fuzz.ratio),
    "addr_ratio": ("addr_norm", fuzz.ratio),
    "addr_tset": ("addr_norm", fuzz.token_set_ratio),
    "addr_partial_tset": ("addr_norm", fuzz.partial_token_set_ratio),
}


def _jaccard(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    inter = a.list.set_intersection(b).list.len()
    union = a.list.set_union(b).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None)


def pair_features(pairs: pl.DataFrame, s1n: pl.DataFrame, qn: pl.DataFrame, workers: int = -1) -> pl.DataFrame:
    """Returns `pairs` with all FEATURES columns added (float32)."""
    si, qi = pairs["si"], pairs["qi"]
    cols = {}
    for feat, (col, scorer) in _SCORERS.items():
        a = s1n[col].gather(si).fill_null("").to_list()
        b = qn[col].gather(qi).fill_null("").to_list()
        cols[feat] = process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)

    side = pl.DataFrame({
        "a_tok": s1n["name_core"].gather(si).str.split(" "),
        "b_tok": qn["name_core"].gather(qi).str.split(" "),
        "a_legal": s1n["legal"].gather(si), "b_legal": qn["legal"].gather(qi),
        "a_nums": s1n["addr_nums"].gather(si), "b_nums": qn["addr_nums"].gather(qi),
        "a_len": s1n["name_core"].gather(si).str.len_chars(), "b_len": qn["name_core"].gather(qi).str.len_chars(),
        "b_addr": qn["addr_norm"].gather(qi),
        "is_s3": qn["entity_id"].gather(qi).str.starts_with("S3-"),
        "non_latin_any": s1n["non_latin"].gather(si) | qn["non_latin"].gather(qi),
    }).select(
        name_tok_jacc=_jaccard(pl.col("a_tok"), pl.col("b_tok")),
        legal_eq=(pl.col("a_legal") == pl.col("b_legal")) & (pl.col("a_legal") != ""),
        legal_both_empty=(pl.col("a_legal") == "") & (pl.col("b_legal") == ""),
        name_len_diff=(pl.col("a_len") - pl.col("b_len")).abs(),
        addr_missing=pl.col("b_addr").fill_null("") == "",
        num_jacc=_jaccard(pl.col("a_nums"), pl.col("b_nums")),
        num_first_eq=pl.when((pl.col("a_nums").list.len() > 0) & (pl.col("b_nums").list.len() > 0))
        .then(pl.col("a_nums").list.first() == pl.col("b_nums").list.first()),
        num_any=pl.col("a_nums").list.set_intersection(pl.col("b_nums")).list.len() > 0,
        is_s3=pl.col("is_s3"),
        non_latin_any=pl.col("non_latin_any"),
    )

    out = pairs.with_columns(**{k: pl.Series(v) for k, v in cols.items()}).hstack(side)
    return out.with_columns(pl.col(FEATURES).cast(pl.Float32))


def add_group_features(pairs: pl.DataFrame) -> pl.DataFrame:
    """Features that look across all candidates of a query / an S1 entity.

    Call ONCE on the complete blocked pair set (before any chunking), then pair_features per chunk.
    """
    return pairs.with_columns(
        blk_gap=pl.col("blk_score").max().over("qi") - pl.col("blk_score"),
        q_n_cands=pl.len().over("qi"),
        # how many queries rank this S1 first: "popular" S1s attract distractors
        s1_n_top1=(pl.col("blk_rank") == 1).sum().over("si"),
    )
