"""Name-token edit features (added after R04), stored beside the stage-1 features.

Unmatched records ("distractors") are generated from an S1 record: the house number is shifted and the name gets words
from a small vocabulary (holdings, group, partners, central, ...), while true-match noise adds other tokens (the, lnc,
...). String similarity cannot tell "X Holdings" from "The X", so we learn a smoothed log-odds lexicon of the query's
extra name_core tokens (not in the S1 name) and the S1's missing ones (not in the query), on labelled pool pairs with a
small name edit. Cross-fitted: P0 rows use the lexicon learned on P1 and vice versa; test uses both pools. Counts are
per query (a query votes true for a token if any of its true pairs has it), because unmatched queries appear in BOTH
pools: with pair counts a token seen in only a few distractors would carry their own labels into the other pool.

Lexicon imputation (unseen vocabularies): in each partition, a token that makes up >= 0.5% of the top-1 pairs as their
single appended edit (appended last in >= 90% of them) has the generator profile. If it does not have that profile in
the train pools, the train lexicon (learned where the word is not a generator word, or never seen) cannot know it, so
its value is raised to the median log-odds of the train generator-profile words. Label-free, never fires on the pools;
it recovers France (groupe, developement, france, participations, international, holding, distribution).

Language-independent edit features (France has its own generator words: groupe, holding, participations, ...): the
generator appends its word at the END of the core name, true-match noise tends to be prepended (the, shri, ...), so we
flag whether the extra token is the query's last / first token, and give its label-free frequency as the single extra
token of near top-1 pairs within the same partition (generator words are extremely frequent there).

Unmatched-number features: distractors shift the house number but keep e.g. the unit number, so "any shared number"
says match. We compare only the numbers of each address that the other lacks: counts, their minimum offset (a
distractor shift is small), and whether one is a prefix/suffix of the other (the dropped-digit noise of true matches).

Output: work/xfeats/{part}/{country}_part-NNN.parquet, row-aligned with work/feats/{part}/{country}_part-NNN.parquet.
"""
import time

import numpy as np
import polars as pl

from . import config as C
from . import io
from .features import feat_files, round_mantissa
from .pools import partitions

X_FEATURES = ["nx_extra_n", "nx_missing_n", "nx_extra_lo_max", "nx_extra_lo_min", "nx_missing_lo_max", "nx_missing_lo_min",
              "nx_num_unm_q", "nx_num_unm_s", "nx_num_unm_absdiff", "nx_num_unm_affix",
              "nx_extra_is_last", "nx_extra_is_first", "nx_missing_is_last", "nx_extra_freq"]
LEX_FEATURES = ["nx_extra_lo_max", "nx_extra_lo_min", "nx_missing_lo_max", "nx_missing_lo_min"]
MAX_EXTRA, MAX_MISSING = 2, 1  # lexicon pairs: >= 1 shared token, <= 2 extra and <= 1 missing (a small name edit)
MIN_COUNT = 50  # distinct queries: one query moves a token's statistic by <= 2% (distractors sit in both pools)
SMOOTH = 1.0
LEX_OF = {"P0": ("P1",), "P1": ("P0",), "test": ("P0", "P1")}
# lexicon imputation for vocabularies unseen in train (France): a token absent from the lexicon that dominates the
# partition's single appended edits behaves like the generator words, so it gets their median log-odds
GEN_FREQ, GEN_LAST = 0.005, 0.9


def xfeat_path(f):
    return C.STAGE_DIR / "xfeats" / f.parent.name / f.name


def lex_path(pool):
    return C.STAGE_DIR / "xfeats" / f"lexicon_{pool}.parquet"


def _names(split, col="name_core"):
    return (pl.read_parquet(io.norm_path(split, 1), columns=[col])[col], io.load_norm(split, "q", columns=[col])[col])


def _num_feats(q_row, s1_row, s1_nums, q_nums) -> dict:
    """Numbers each address has that the other lacks: counts, minimum |offset| and prefix/suffix relation."""
    d = (pl.DataFrame({"qn": q_nums.gather(q_row).str.split(" ").list.unique(),
                       "sn": s1_nums.gather(s1_row).str.split(" ").list.unique()})
         .select(uq=pl.col("qn").list.set_difference("sn").list.filter(pl.element() != ""),
                 us=pl.col("sn").list.set_difference("qn").list.filter(pl.element() != ""))
         .with_row_index("i"))
    n = d.height
    out = {"nx_num_unm_q": d["uq"].list.len().cast(pl.Float32).to_numpy(),
           "nx_num_unm_s": d["us"].list.len().cast(pl.Float32).to_numpy()}
    x = (d.select("i", pl.col("uq").alias("a")).explode("a").drop_nulls("a")
         .join(d.select("i", pl.col("us").alias("b")).explode("b").drop_nulls("b"), on="i")
         .with_columns(la=pl.col("a").str.len_chars(), lb=pl.col("b").str.len_chars())
         .with_columns(diff=(pl.col("a").str.slice(0, 15).cast(pl.Int64) - pl.col("b").str.slice(0, 15).cast(pl.Int64)).abs(),
                       affix=pl.when(pl.col("la") <= pl.col("lb"))
                       .then((pl.col("la") >= 2) & (pl.col("b").str.starts_with(pl.col("a")) | pl.col("b").str.ends_with(pl.col("a"))))
                       .otherwise((pl.col("lb") >= 2) & (pl.col("a").str.starts_with(pl.col("b")) | pl.col("a").str.ends_with(pl.col("b")))))
         .group_by("i").agg(pl.col("diff").min(), pl.col("affix").any()))
    idx = x["i"].to_numpy()
    for name, col in (("nx_num_unm_absdiff", "diff"), ("nx_num_unm_affix", "affix")):
        v = np.full(n, np.nan, np.float32)
        v[idx] = np.minimum(x[col].cast(pl.Float64).to_numpy(), 1e6)
        out[name] = v
    return out


def _edits(q_row, s1_row, s1_names, q_names) -> pl.DataFrame:
    """Per pair: extra (query tokens not in the S1 name) and missing (S1 tokens not in the query) name_core tokens."""
    return (pl.DataFrame({"qt": q_names.gather(q_row).str.split(" ").list.filter(pl.element() != ""),
                          "st": s1_names.gather(s1_row).str.split(" ").list.filter(pl.element() != "")})
            .select(extra=pl.col("qt").list.set_difference("st"),
                    missing=pl.col("st").list.set_difference("qt"),
                    shared=pl.col("qt").list.set_intersection("st").list.len(),
                    q_first=pl.col("qt").list.first(), q_last=pl.col("qt").list.last(), s_last=pl.col("st").list.last()))


def extra_freq(part, country, s1_names, q_names) -> pl.DataFrame:
    """Label-free: tok -> freq (share of the partition's top-1 pairs whose name edit is exactly one extra token tok) and
    last (share of those where tok is the query's last core token). Cached under work/xfeats/."""
    path = C.STAGE_DIR / "xfeats" / f"freq_{part}_{country}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    acc, n = [], 0
    for f in feat_files(part, country):
        k = pl.read_parquet(f, columns=["q_row", "s1_row", "blk_rank"]).filter(pl.col("blk_rank") == 1)
        n += k.height
        e = _edits(k["q_row"], k["s1_row"], s1_names, q_names)
        e = e.filter((pl.col("extra").list.len() == 1) & (pl.col("missing").list.len() == 0) & (pl.col("shared") >= 1))
        acc.append(e.select(tok=pl.col("extra").list.first(), last=pl.col("extra").list.first() == pl.col("q_last"))
                   .group_by("tok").agg(pl.len(), pl.col("last").sum()))
    out = (pl.concat(acc).group_by("tok").agg(pl.col("len").sum(), pl.col("last").sum())
           .select("tok", freq=pl.col("len") / max(n, 1), last=pl.col("last") / pl.col("len")))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    return out


def _profile(fq: pl.DataFrame) -> pl.DataFrame:
    return fq.filter((pl.col("freq") >= GEN_FREQ) & (pl.col("last") >= GEN_LAST)).select("tok").unique()


def generator_lo(lo_extra: pl.DataFrame):
    """-> (median lexicon log-odds of the train tokens with the generator profile, list of those tokens)."""
    s1_names, q_names = _names("train")
    fq = pl.concat([extra_freq(p, c, s1_names, q_names) for p, c in partitions(["P0", "P1"])])
    gen = _profile(fq)
    vals = gen.join(lo_extra, on="tok", how="inner")["lo"]
    print(f"[xfeats] generator-profile train tokens: {vals.len()} ({sorted(gen['tok'].to_list())}), median lo {vals.median():.2f}")
    return float(vals.median()), gen["tok"].to_list()


def impute_lexicon(lo_extra: pl.DataFrame, fq: pl.DataFrame, lo_gen: float, train_gen, label="") -> pl.DataFrame:
    """Raise generator-profile tokens of this partition that are not train generator words to >= lo_gen."""
    new = (_profile(fq).filter(~pl.col("tok").is_in(list(train_gen))).join(lo_extra, on="tok", how="left")
           .select("tok", old=pl.col("lo"), lo=pl.max_horizontal(pl.col("lo").fill_null(lo_gen), pl.lit(lo_gen))))
    if new.height:
        print(f"[xfeats] {label}: lexicon raised to {lo_gen:.2f} for "
              + ", ".join(f"{t} ({'new' if o is None else f'{o:.2f}'})" for t, o in new.select("tok", "old").iter_rows()))
    return pl.concat([lo_extra.join(new.select("tok"), on="tok", how="anti"), new.select("tok", "lo")])


def learn_lexicon(pool):
    """Per-query token counts on labelled pairs of `pool` whose names differ by a small edit (MAX_EXTRA, MAX_MISSING)."""
    t0 = time.time()
    s1_names, q_names = _names("train")
    acc = []
    for part, c in partitions([pool]):
        for f in feat_files(part, c):
            k = pl.read_parquet(f, columns=["q_row", "s1_row", "y"])
            e = _edits(k["q_row"], k["s1_row"], s1_names, q_names).with_columns(y=k["y"], q_row=k["q_row"])
            e = e.filter((pl.col("shared") >= 1) & (pl.col("extra").list.len() <= MAX_EXTRA)
                         & (pl.col("missing").list.len() <= MAX_MISSING))
            for kind in ("extra", "missing"):
                acc.append(e.select("q_row", pl.col(kind).alias("tok"), "y").explode("tok").drop_nulls("tok")
                           .group_by("q_row", "tok").agg(pl.col("y").max()).with_columns(kind=pl.lit(kind)))
    # a query's pairs can straddle feature files: vote per (query, token) across files, then count queries
    lex = (pl.concat(acc).group_by("kind", "q_row", "tok").agg(pl.col("y").max())
           .group_by("kind", "tok").agg(n1=pl.col("y").cast(pl.Int64).sum(), n=pl.len()))
    lex_path(pool).parent.mkdir(parents=True, exist_ok=True)
    lex.write_parquet(lex_path(pool))
    print(f"[xfeats] lexicon {pool}: {lex.height:,} (kind, token) rows in {time.time() - t0:.0f}s")
    return lex


def log_odds(pools) -> dict:
    """kind -> DataFrame(tok, lo): smoothed log-odds of 'not a true pair', centred on the kind's overall odds."""
    lex = pl.concat([pl.read_parquet(lex_path(p)) for p in pools]).group_by("kind", "tok").agg(pl.col("n1").sum(), pl.col("n").sum())
    out = {}
    for kind in ("extra", "missing"):
        d = lex.filter(pl.col("kind") == kind).with_columns(n0=pl.col("n") - pl.col("n1"))
        base = np.log((d["n0"].sum() + SMOOTH) / (d["n1"].sum() + SMOOTH))
        out[kind] = (d.filter(pl.col("n") >= MIN_COUNT)
                     .select("tok", lo=((pl.col("n0") + SMOOTH) / (pl.col("n1") + SMOOTH)).log() - base))
    return out


def build_xfeats(parts=("P0", "P1", "test")):
    for p in {q for part in parts for q in LEX_OF[part]}:
        if not lex_path(p).exists():
            learn_lexicon(p)
    for split in ("train", "test"):
        todo = [(p, c) for p, c in partitions(parts) if (p == "test") == (split == "test")]
        if not todo:
            continue
        s1_names, q_names = _names(split)
        s1_nums, q_nums = _names(split, "addr_nums")
        lo_gen = None
        for part, c in todo:
            t0 = time.time()
            lo = log_odds(LEX_OF[part])
            fq = extra_freq(part, c, s1_names, q_names)
            if lo_gen is None:
                lo_gen, train_gen = generator_lo(log_odds(("P0", "P1"))["extra"])
            lo["extra"] = impute_lexicon(lo["extra"], fq, lo_gen, train_gen, label=f"{part} {c}")
            n = 0
            for f in feat_files(part, c):
                k = pl.read_parquet(f, columns=["q_row", "s1_row"])
                e = _edits(k["q_row"], k["s1_row"], s1_names, q_names).with_row_index("i")
                cols = {"nx_extra_n": e["extra"].list.len().cast(pl.Float32).to_numpy(),
                        "nx_missing_n": e["missing"].list.len().cast(pl.Float32).to_numpy()}
                for kind in ("extra", "missing"):
                    agg = (e.select("i", pl.col(kind).alias("tok")).explode("tok").join(lo[kind], on="tok", how="inner")
                           .group_by("i").agg(mx=pl.col("lo").max(), mn=pl.col("lo").min()))
                    for stat in ("mx", "mn"):
                        v = np.full(k.height, np.nan, np.float32)
                        v[agg["i"].to_numpy()] = agg[stat].to_numpy()
                        cols[f"nx_{kind}_lo_{'max' if stat == 'mx' else 'min'}"] = v
                cols.update(_num_feats(k["q_row"], k["s1_row"], s1_nums, q_nums))
                pos = e.select(last=pl.col("extra").list.contains(pl.col("q_last")),
                               first=pl.col("extra").list.contains(pl.col("q_first")),
                               mlast=pl.col("missing").list.contains(pl.col("s_last")))
                cols["nx_extra_is_last"] = pos["last"].cast(pl.Float32).fill_null(0).to_numpy()
                cols["nx_extra_is_first"] = pos["first"].cast(pl.Float32).fill_null(0).to_numpy()
                cols["nx_missing_is_last"] = pos["mlast"].cast(pl.Float32).fill_null(0).to_numpy()
                one = (e.filter(pl.col("extra").list.len() == 1).select("i", tok=pl.col("extra").list.first())
                       .join(fq, on="tok", how="inner"))
                v = np.full(k.height, np.nan, np.float32)
                v[one["i"].to_numpy()] = one["freq"].to_numpy()
                cols["nx_extra_freq"] = v
                out = xfeat_path(f)
                out.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame({x: round_mantissa(cols[x]) for x in X_FEATURES}).write_parquet(out, **C.PARQUET_KW)
                n += k.height
            print(f"[xfeats] {part} {c}: {n:,} pairs in {time.time() - t0:.0f}s (lexicon of {LEX_OF[part]})")
