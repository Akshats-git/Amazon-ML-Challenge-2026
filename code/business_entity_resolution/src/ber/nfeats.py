"""R09 number-relation (nr_*) and number-oracle-lexicon (nol_*) features, stored beside the stage-1 features.

The generator (LOG R08c): a distractor copies an S1 record and shifts its house number UPWARD by one of GEN = {1, 2, 3,
4, 5, 7, 9, 11, 13, 21}, then may edit the name (append a word from a per-country list, swap a word, change the legal
form). True-match number noise is symmetric (+-1, +-2, +-10, +-20, dropped digits, digit edits). The R04/R05 number
features only see |diff|; these see the sign, the shift set and the edit type of the unmatched numbers.

Number relations use addr_nums (leading zeros stripped) and the alphabetic tokens of addr_norm. q_only / s_only are
the numbers one address has and the other lacks; every (q_only, s_only) pair is compared as q - s.

Number-oracle lexicon (label-free, per partition, identical code for P0, P1 and test): on the partition's blocking
rank-1 pairs, a single +GEN shift on the same street marks the distractor proxy D, equal numbers on the same street the
true proxy T. For each name_core token w that the query adds (extra) or drops (missing),
    nol(w) = log((cD(w) + 1) / N_D) - log((cT(w) + 1) / N_T)       (per kind; NaN unless cD + cT >= NOL_MIN)
so generator words score high and true-noise words low without labels, also in countries absent from train (France).

Output: work/nfeats/{part}/{country}_part-NNN.parquet, row-aligned with work/feats/{part}/{country}_part-NNN.parquet;
the per-partition lexicon goes to work/nfeats/nol_{part}_{country}.parquet.
"""
import time

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from . import config as C
from . import io
from .features import feat_files, round_mantissa
from .pools import load_partition, partitions

GEN = (1, 2, 3, 4, 5, 7, 9, 11, 13, 21)
NEG_SMALL = (-1, -2)
NEG_GEN = tuple(-g for g in GEN if g > 2)
SAME_STREET = 0.8
NOL_MIN = 20
NR_FEATURES = ["nr_all_equal", "nr_n_shared", "nr_gen_pos", "nr_neg_small", "nr_neg_gen", "nr_signed_min", "nr_digit_edit",
               "nr_transposed", "nr_len_diff", "nr_street_jac", "nr_eq_same_street", "nr_gen_same_street"]
NOL_FEATURES = ["nol_extra_max", "nol_extra_min", "nol_missing_max", "nol_missing_min"]
N_FEATURES = NR_FEATURES + NOL_FEATURES
TEXT = ["addr_nums", "addr_norm", "name_core"]


def nfeat_path(f):
    return C.STAGE_DIR / "nfeats" / f.parent.name / f.name


def nol_path(part, country):
    return C.STAGE_DIR / "nfeats" / f"nol_{part}_{country}.parquet"


def tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Per record: nums (sorted number multiset), words (alphabetic address tokens), core (name_core tokens)."""
    return df.select(
        nums=pl.col("addr_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.sort(),
        words=pl.col("addr_norm").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]+$"))).list.unique(),
        core=pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique(),
    )


def number_relations(d: pl.DataFrame) -> dict:
    """d: per pair qn/sn (sorted number lists) and qw/sw (address word lists). -> dict of NR_FEATURES float32 arrays."""
    n = d.height
    e = d.select(
        all_eq=(pl.col("qn").list.len() > 0) & (pl.col("sn").list.len() > 0) & (pl.col("qn") == pl.col("sn")),
        shared=pl.col("qn").list.set_intersection("sn").list.len(),
        uq=pl.col("qn").list.set_difference("sn"),
        us=pl.col("sn").list.set_difference("qn"),
        inter=pl.col("qw").list.set_intersection("sw").list.len(),
        union=pl.col("qw").list.set_union("sw").list.len(),
    ).with_row_index("i")
    jac = e.select(pl.when(pl.col("union") > 0).then(pl.col("inter") / pl.col("union")))
    jac = jac.to_series().cast(pl.Float32).to_numpy()
    x = (e.select("i", a=pl.col("uq")).explode("a").drop_nulls("a")
         .join(e.select("i", b=pl.col("us")).explode("b").drop_nulls("b"), on="i"))
    x = x.with_columns(diff=pl.col("a").str.slice(0, 15).cast(pl.Int64) - pl.col("b").str.slice(0, 15).cast(pl.Int64),
                       lend=pl.col("a").str.len_chars().cast(pl.Int32) - pl.col("b").str.len_chars().cast(pl.Int32))
    lev = cpdist(x["a"].to_list(), x["b"].to_list(), scorer=Levenshtein.distance, workers=C.N_JOBS, dtype=np.int32) if x.height else np.zeros(0, np.int32)
    x = x.with_columns(lev1=pl.Series(lev == 1),
                       trans=(pl.col("a") != pl.col("b")) & (pl.col("lend") == 0)
                       & (pl.col("a").str.split("").list.sort() == pl.col("b").str.split("").list.sort()))
    g = x.group_by("i").agg(
        gen_pos=pl.col("diff").is_in(GEN).sum(), neg_small=pl.col("diff").is_in(NEG_SMALL).sum(),
        neg_gen=pl.col("diff").is_in(NEG_GEN).sum(), signed_min=pl.col("diff").sort_by(pl.col("diff").abs()).first(),
        len_diff=pl.col("lend").sort_by(pl.col("diff").abs()).first(), digit_edit=pl.col("lev1").sum(),
        trans=pl.col("trans").sum())
    idx = g["i"].to_numpy()
    out = {"nr_all_equal": e["all_eq"].cast(pl.Float32).to_numpy(), "nr_n_shared": e["shared"].cast(pl.Float32).to_numpy()}
    for name, col, fill in (("nr_gen_pos", "gen_pos", 0.0), ("nr_neg_small", "neg_small", 0.0), ("nr_neg_gen", "neg_gen", 0.0),
                            ("nr_signed_min", "signed_min", np.nan), ("nr_digit_edit", "digit_edit", 0.0),
                            ("nr_transposed", "trans", 0.0), ("nr_len_diff", "len_diff", np.nan)):
        v = np.full(n, fill, np.float32)
        v[idx] = np.clip(g[col].cast(pl.Float64).to_numpy(), -1e4, 1e4)
        out[name] = v
    out["nr_street_jac"] = jac
    same = np.nan_to_num(jac, nan=0.0) >= SAME_STREET
    one = (e["uq"].list.len().to_numpy() == 1) & (e["us"].list.len().to_numpy() == 1)
    out["nr_eq_same_street"] = (out["nr_all_equal"].astype(bool) & same).astype(np.float32)
    out["nr_gen_same_street"] = (one & (out["nr_gen_pos"] == 1) & same).astype(np.float32)
    return out


def name_edits(d: pl.DataFrame) -> pl.DataFrame:
    """d: per pair qc/sc (name_core token lists) -> extra (query tokens not in the S1 name) and missing (the reverse)."""
    return d.select(extra=pl.col("qc").list.set_difference("sc"), missing=pl.col("sc").list.set_difference("qc"))


class Partition:
    """Tokenized text of one (part, country), gathered per feature file by global row keys."""

    def __init__(self, part, country, s1_norm: pl.DataFrame, q_norm: pl.DataFrame):
        self.s1_rows, self.q_rows = load_partition(part, country)
        self.s = tokens(s1_norm[self.s1_rows])
        self.q = tokens(q_norm[self.q_rows])

    def pairs(self, q_row, s1_row) -> pl.DataFrame:
        qi, si = np.searchsorted(self.q_rows, q_row), np.searchsorted(self.s1_rows, s1_row)
        assert (self.q_rows[qi] == q_row).all() and (self.s1_rows[si] == s1_row).all(), "pair outside its partition"
        q, s = self.q[qi], self.s[si]
        return pl.DataFrame({"qn": q["nums"], "sn": s["nums"], "qw": q["words"], "sw": s["words"],
                             "qc": q["core"], "sc": s["core"]})


def learn_nol(part, country, P: Partition) -> pl.DataFrame:
    """Label-free lexicon of one partition from its blocking rank-1 pairs -> (kind, tok, cD, cT, nol)."""
    acc, n_d, n_t = [], 0, 0
    for f in feat_files(part, country):
        k = pl.read_parquet(f, columns=["q_row", "s1_row", "blk_rank"]).filter(pl.col("blk_rank") == 1)
        d = P.pairs(k["q_row"].to_numpy(), k["s1_row"].to_numpy())
        nr = number_relations(d.select("qn", "sn", "qw", "sw"))
        e = name_edits(d).with_columns(D=pl.Series(nr["nr_gen_same_street"] == 1), T=pl.Series(nr["nr_eq_same_street"] == 1))
        n_d, n_t = n_d + int(e["D"].sum()), n_t + int(e["T"].sum())
        e = e.filter(pl.col("D") | pl.col("T"))
        for kind in ("extra", "missing"):
            acc.append(e.select(pl.col(kind).alias("tok"), "D", "T").explode("tok").drop_nulls("tok")
                       .group_by("tok").agg(cD=pl.col("D").sum(), cT=pl.col("T").sum()).with_columns(kind=pl.lit(kind)))
    lex = (pl.concat(acc).group_by("kind", "tok").agg(pl.col("cD").sum(), pl.col("cT").sum())
           .with_columns(nol=pl.when(pl.col("cD") + pl.col("cT") >= NOL_MIN)
                         .then(((pl.col("cD") + 1) / max(n_d, 1)).log() - ((pl.col("cT") + 1) / max(n_t, 1)).log())))
    lex = lex.with_columns(n_D=pl.lit(n_d), n_T=pl.lit(n_t))
    path = nol_path(part, country)
    path.parent.mkdir(parents=True, exist_ok=True)
    lex.write_parquet(path)
    top = lex.filter(pl.col("kind") == "extra").drop_nulls("nol").sort("nol", descending=True)
    print(f"[nfeats] NOL {part} {country}: D {n_d:,} / T {n_t:,} rank-1 pairs; {lex['nol'].is_not_null().sum():,} tokens with "
          f"a value; top extra: {', '.join(f'{t} {v:.1f}' for t, v in top.head(10).select('tok', 'nol').iter_rows())}; "
          f"bottom extra: {', '.join(f'{t} {v:.1f}' for t, v in top.tail(6).select('tok', 'nol').iter_rows())}")
    return lex


def nol_feats(e: pl.DataFrame, lex: pl.DataFrame) -> dict:
    """e: name_edits() of a set of pairs -> NOL_FEATURES (max/min over the pair's tokens that have a value)."""
    n = e.height
    e = e.with_row_index("i")
    out = {}
    for kind in ("extra", "missing"):
        lk = lex.filter((pl.col("kind") == kind) & pl.col("nol").is_not_null()).select("tok", "nol")
        agg = (e.select("i", tok=pl.col(kind)).explode("tok").join(lk, on="tok", how="inner")
               .group_by("i").agg(mx=pl.col("nol").max(), mn=pl.col("nol").min()))
        for stat, nm in (("mx", "max"), ("mn", "min")):
            v = np.full(n, np.nan, np.float32)
            v[agg["i"].to_numpy()] = agg[stat].to_numpy()
            out[f"nol_{kind}_{nm}"] = v
    return out


def build_partition_nfeats(part, country, s1_norm, q_norm):
    t0 = time.time()
    P = Partition(part, country, s1_norm, q_norm)
    lex = learn_nol(part, country, P)
    n = 0
    for f in feat_files(part, country):
        k = pl.read_parquet(f, columns=["q_row", "s1_row"])
        d = P.pairs(k["q_row"].to_numpy(), k["s1_row"].to_numpy())
        cols = number_relations(d.select("qn", "sn", "qw", "sw"))
        cols.update(nol_feats(name_edits(d), lex))
        out = nfeat_path(f)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({x: round_mantissa(cols[x]) for x in N_FEATURES}).write_parquet(out, **C.PARQUET_KW)
        n += k.height
    print(f"[nfeats] {part} {country}: {n:,} pairs in {time.time() - t0:.0f}s")


def build_nfeats(parts=("P0", "P1", "test"), only=None):
    """only: restrict to these countries (e.g. a France-only variant in a parallel WORK_DIR)."""
    for split in ("train", "test"):
        todo = [(p, c) for p, c in partitions(parts) if (p == "test") == (split == "test") and (not only or c in only)]
        if not todo:
            continue
        s1_norm = io.load_norm(split, "s1", columns=TEXT)
        q_norm = io.load_norm(split, "q", columns=TEXT)
        for part, c in todo:
            build_partition_nfeats(part, c, s1_norm, q_norm)
        del s1_norm, q_norm
