"""R09 same-address rescue for test countries absent from train (France).

France has no labels, and the R06 model misses ~40-48k of its true links (LOG R08c): queries at the S1's exact address
(same house numbers, same street) whose name edit is an organisation-word swap (club <-> comite <-> amicale ...) or an
"& Fils" / "& Associes" append. To the model these edits look like US/India distractor edits. The distractor generator,
however, always shifts the house number (upward by one of 1, 2, 3, 4, 5, 7, 9, 11, 13, 21), and on the train pools equal
numbers + same street are 96-99% true whatever the name edit (there the model already links them).

Rule, applied to each query's argmax pair in countries absent from train:
  - the S1 has >= 1 number and the query's number multiset equals the S1's;
  - the Jaccard of their alphabetic address tokens is >= RESCUE_JAC;
  - sibling guard: no other candidate S1 of the query also satisfies both (another business at the same address).
Then p = max(p, RESCUE_P), so the (T1, T2) policy links it. Addresses are compared under the France address rules
(normalize.fr_addr: street-type abbreviations, regions/departments dropped), whatever normalization the features use.
Keys are computed from the raw test TSVs and cached in WORK_DIR/rescue/test_{country}.parquet.
"""
import time
from multiprocessing import Pool

import numpy as np
import polars as pl

from . import config as C
from . import io
from . import normalize as N


def _keys_chunk(args):
    names, addrs, fr = args
    N.load_dicts()
    nums, street = [], []
    for nm, a in zip(names, addrs):
        an, _, nu = N.norm_addr(a, N.is_nonlatin(nm, a), fr)
        nums.append(" ".join(sorted(nu.split())))
        street.append(" ".join(sorted({t for t in an.split() if t.isalpha()})))
    return nums, street


def _keys(df: pl.DataFrame, fr: bool, chunk=50_000) -> pl.DataFrame:
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    tasks = [(names[i:i + chunk], addrs[i:i + chunk], fr) for i in range(0, len(names), chunk)]
    with Pool(C.N_JOBS) as p:
        res = p.map(_keys_chunk, tasks)
    return pl.DataFrame({"row": df["row"], "nums": [v for r in res for v in r[0]], "street": [v for r in res for v in r[1]]},
                        schema={"row": pl.Int32, "nums": pl.Utf8, "street": pl.Utf8})


def addr_keys(country, split="test"):
    """-> (S1 keys, query keys): row (s1_row / q_row), nums (sorted number multiset), street (sorted alphabetic token set)."""
    path = C.WORK_DIR / "rescue" / f"{split}_{country}.parquet"
    if not path.exists():
        t0 = time.time()
        fr = country == "France"
        s1 = io.read_source(split, 1).with_row_index("row").filter(pl.col("country") == country)
        n2 = pl.scan_parquet(io.norm_path(split, 2)).select(pl.len()).collect().item()
        q = pl.concat([io.read_source(split, 2).with_row_index("row").filter(pl.col("country") == country),
                       io.read_source(split, 3).with_row_index("row", offset=n2).filter(pl.col("country") == country)])
        out = pl.concat([_keys(s1, fr).with_columns(is_s1=pl.lit(True)), _keys(q, fr).with_columns(is_s1=pl.lit(False))])
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path)
        print(f"[rescue] address keys {split} {country}: {s1.height:,} S1, {q.height:,} queries in {time.time() - t0:.0f}s")
    d = pl.read_parquet(path)
    return d.filter(pl.col("is_s1")).drop("is_s1"), d.filter(~pl.col("is_s1")).drop("is_s1")


def _pos(rows: np.ndarray, keys: np.ndarray) -> np.ndarray:
    i = np.searchsorted(rows, keys)
    assert (i < len(rows)).all() and (rows[np.minimum(i, len(rows) - 1)] == keys).all(), "pair outside the key table"
    return i


def same_address(q_row, s1_row, country, split="test") -> np.ndarray:
    """Per candidate pair: S1 has numbers, equal number multisets, street Jaccard >= RESCUE_JAC."""
    sk, qk = addr_keys(country, split)
    si, qi = _pos(sk["row"].to_numpy(), np.asarray(s1_row)), _pos(qk["row"].to_numpy(), np.asarray(q_row))
    code = pl.concat([sk["nums"], qk["nums"]]).cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    code[(pl.concat([sk["nums"], qk["nums"]]) == "").to_numpy()] = -1
    sn, qn = code[:sk.height][si], code[sk.height:][qi]
    eq = (sn >= 0) & (sn == qn)
    idx = np.flatnonzero(eq)
    d = pl.DataFrame({"a": sk["street"].gather(si[idx]), "b": qk["street"].gather(qi[idx])})
    jac = d.select(j=pl.when(pl.col("a") == pl.col("b")).then(pl.when(pl.col("a") != "").then(1.0).otherwise(0.0))
                   .otherwise(pl.col("a").str.split(" ").list.set_intersection(pl.col("b").str.split(" ")).list.len()
                              / pl.col("a").str.split(" ").list.set_union(pl.col("b").str.split(" ")).list.len().clip(1))
                   )["j"].to_numpy()
    out = np.zeros(len(q_row), bool)
    out[idx] = jac >= C.RESCUE_JAC
    return out


def rescue_mask(top: pl.DataFrame, q_row, s1_row, country, split="test") -> np.ndarray:
    """Per top row: the rule fires (argmax pair same-address, no other same-address candidate)."""
    m = same_address(q_row, s1_row, country, split)
    new = np.r_[True, q_row[1:] != q_row[:-1]]
    gid = np.cumsum(new) - 1
    assert (q_row[new] == top["q_row"].to_numpy()).all(), "top not aligned with the candidate pairs"
    n_same = np.bincount(gid, weights=m, minlength=top.height)
    top_s1 = top["s1_row"].to_numpy()
    am_same = np.zeros(top.height, bool)
    am_same[gid[m]] |= s1_row[m] == top_s1[gid[m]]
    return am_same & (n_same == 1), am_same, n_same


def rescue_top(top: pl.DataFrame, q_row, s1_row, country, T=None) -> pl.DataFrame:
    """top: argmax_rows(q_row, s1_row, p) of one test country (pairs grouped by query, same order as top).
    -> top with p raised to RESCUE_P (at least the thresholds) on rescued argmax pairs."""
    t0 = time.time()
    fire, am_same, n_same = rescue_mask(top, q_row, s1_row, country)
    p = top["p"].to_numpy()
    rp = max(C.RESCUE_P, *(T or (0,)))
    lo = fire & (p < rp)
    print(f"[rescue] {country}: argmax same-address {int(am_same.sum()):,} (sibling-guarded {int((am_same & (n_same > 1)).sum()):,}); "
          f"rescued {int(fire.sum()):,}, raised {int(lo.sum()):,} (p < {rp:.2f}; p quartiles "
          f"{np.round(np.quantile(p[lo], [0.25, 0.5, 0.75]), 3).tolist() if lo.any() else []}) in {time.time() - t0:.0f}s")
    return top.with_columns(p=pl.Series(np.where(fire, np.maximum(p, rp), p).astype(np.float32)))


def unseen_countries() -> set:
    from .pools import countries
    return set(countries("test")) - set(countries("train"))
