"""Section 7.8 stage-2 context features computed from p1 over ALL pairs of a partition (row order preserved).

Per query: q_p1_max, q_p1_second, q_margin, pair_rank_in_q, is_q_argmax.
Per S1 s over queries whose p1-argmax is s (self excluded): s_cnt_argmax_ge50, s_sum_p1, s_max_p1_other,
pair_rank_in_s, s_cnt_same_source_ge50.
"""
import polars as pl

CTX2_FEATURES = ["p1", "q_p1_max", "q_p1_second", "q_margin", "pair_rank_in_q", "is_q_argmax",
                 "s_cnt_argmax_ge50", "s_sum_p1", "s_max_p1_other", "pair_rank_in_s", "s_cnt_same_source_ge50"]


def stage2_context(df: pl.DataFrame) -> pl.DataFrame:
    """df: q_row, s1_row, p1, is_s3 (any order). Returns CTX2_FEATURES in the same row order."""
    d = df.select("q_row", "s1_row", "p1", "is_s3").with_row_index("_i")
    d = d.sort(["q_row", "p1"], descending=[False, True])
    d = d.with_columns(
        pair_rank_in_q=pl.int_range(1, pl.len() + 1).over("q_row"),
        q_p1_max=pl.col("p1").first().over("q_row"),
        q_p1_second=pl.col("p1").get(1, null_on_oob=True).over("q_row").fill_null(0.0),
    ).with_columns(
        q_margin=pl.col("q_p1_max") - pl.col("q_p1_second"),
        is_q_argmax=(pl.col("pair_rank_in_q") == 1),
    )
    A = d.filter("is_q_argmax")
    sagg = A.group_by("s1_row").agg(
        (pl.col("p1") >= 0.5).sum().alias("cnt50"),
        ((pl.col("p1") >= 0.5) & (pl.col("is_s3") == 1)).sum().alias("cnt50_s3"),
        pl.col("p1").sum().alias("sum_p1"),
        pl.col("p1").max().alias("top1"),
        pl.col("p1").sort(descending=True).get(1, null_on_oob=True).fill_null(0.0).alias("top2"),
    )
    d = d.join(sagg, on="s1_row", how="left").with_columns(pl.col("cnt50", "cnt50_s3", "sum_p1", "top1", "top2").fill_null(0))
    self50 = (pl.col("is_q_argmax") & (pl.col("p1") >= 0.5)).cast(pl.Int64)
    d = d.with_columns(
        s_cnt_argmax_ge50=pl.col("cnt50") - self50,
        s_cnt_same_source_ge50=pl.when(pl.col("is_s3") == 1).then(pl.col("cnt50_s3")).otherwise(pl.col("cnt50") - pl.col("cnt50_s3")) - self50,
        s_sum_p1=pl.col("sum_p1") - pl.when("is_q_argmax").then(pl.col("p1")).otherwise(0.0),
        s_max_p1_other=pl.when(pl.col("is_q_argmax") & (pl.col("p1") >= pl.col("top1"))).then(pl.col("top2")).otherwise(pl.col("top1")),
    )
    # rank of this pair's p1 among the argmax queries of s (argmax rows first on ties)
    d = d.sort(["s1_row", "p1", "is_q_argmax"], descending=[False, True, True]).with_columns(
        _ca=pl.col("is_q_argmax").cast(pl.Int32).cum_sum().over("s1_row"))
    d = d.with_columns(pair_rank_in_s=pl.when("is_q_argmax").then(pl.col("_ca")).otherwise(pl.col("_ca") + 1))
    return d.sort("_i").select(pl.col(c).cast(pl.Float32) for c in CTX2_FEATURES)
