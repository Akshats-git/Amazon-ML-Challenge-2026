"""Build-sequence checks: Step 1 (metric) and Step 2 (data facts)."""
import numpy as np
import polars as pl

from . import io
from .metrics import macro_f05, macro_f05_ids

EXPECTED_ROWS = {
    ("train", 1): {"US": 1_323_633, "India": 883_188},
    ("train", 2): {"US": 3_016_817, "India": 2_017_799},
    ("train", 3): {"US": 3_170_056, "India": 2_115_547},
    ("test", 1): {"US": 663_106, "India": 809_986, "France": 259_452},
    ("test", 2): {"US": 1_871_330, "India": 2_312_565, "France": 703_378},
    ("test", 3): {"US": 1_945_701, "India": 2_405_000, "France": 731_615},
}
EXPECTED_TRUE_PAIRS = {"S2": 3_693_619, "S3": 3_944_746}
EXPECTED_SINGLETON_SHARE = 0.05585


def _raw_ids():
    s1 = io.read_source("train", 1).select("entity_id", "country").with_row_index("s1_row")
    q = io.read_queries_raw("train").select("entity_id", "country").with_row_index("q_row")
    return s1, q


def step1():
    f = macro_f05_ids({"a": ["S2-47", "S2-193", "S3-812"]}, {"a": ["S2-47", "S3-812"]})
    print(f"unit test F = {f:.4f} (want 0.714)")
    assert abs(f - 0.7142857) < 1e-4
    assert macro_f05_ids({"a": [], "b": ["x"]}, {"a": [], "b": ["x"]}) == 1.0
    assert macro_f05_ids({"a": ["x"]}, {"a": []}) == 0.0

    s1, q = _raw_ids()
    gt = io.read_gt_pairs()
    gtr = gt.join(s1.select(pl.col("entity_id").alias("s1_id"), "s1_row"), on="s1_id").join(
        q.select(pl.col("entity_id").alias("cand_id"), "q_row"), on="cand_id")
    assert gtr.height == gt.height
    uni = s1["s1_row"].to_numpy()
    ts, tq = gtr["s1_row"].to_numpy(), gtr["q_row"].to_numpy()
    f_gt = macro_f05(uni, ts, tq, ts, tq)
    f_empty = macro_f05(uni, np.array([], np.int64), np.array([], np.int64), ts, tq)
    print(f"predict GT -> {f_gt:.6f} (want 1.0); predict empty -> {f_empty:.5f} (want {EXPECTED_SINGLETON_SHARE})")
    assert f_gt == 1.0
    assert abs(f_empty - EXPECTED_SINGLETON_SHARE) < 5e-5
    print("STEP 1 PASS")


def step2():
    for (split, src), exp in EXPECTED_ROWS.items():
        df = io.read_source(split, src)
        got = dict(df.group_by("country").len().iter_rows())
        print(f"{split} S{src}: {df.height:,} {got}")
        assert got == exp, (split, src, got, exp)
        assert df["entity_id"].str.starts_with(f"S{src}-").all()
        assert df["entity_id"].n_unique() == df.height
        if src == 1:
            assert (df["business_address"].str.strip_chars() != "").all(), "S1 address empty"
    gt = io.read_tsv(io.gt_path())
    assert gt.height == 2_206_821 and gt["source1_entity_id"].n_unique() == gt.height
    pairs = io.read_gt_pairs()
    by_src = dict(pairs.group_by(pl.col("cand_id").str.slice(0, 2)).len().iter_rows())
    print(f"GT true pairs {pairs.height:,} {by_src}")
    assert pairs.height == 7_638_365 and by_src == EXPECTED_TRUE_PAIRS
    assert pairs["cand_id"].n_unique() == pairs.height, "a candidate appears in two GT lists"
    s1, q = _raw_ids()
    j = pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("country").alias("c1")), on="s1_id").join(
        q.select(pl.col("entity_id").alias("cand_id"), pl.col("country").alias("c2")), on="cand_id")
    assert j.height == pairs.height, "GT id missing from sources"
    cross = int((j["c1"] != j["c2"]).sum())
    print(f"cross-country true pairs: {cross}")
    assert cross == 0
    nq = q.height
    distract = nq - pairs.height
    print(f"distractors {distract:,} ({distract / s1.height:.3f} per S1)")
    print("STEP 2 PASS")


def step8(parts=("P0", "P1"), n_show=20):
    """Feature sanity: positive rate, positive addr token_set median, constant features, raw-text spot check.
    Streams the feature files (a full pool is ~65M pairs)."""
    from .features import FEATURES, feat_files

    ok = True
    last = None
    for part in parts:
        files = feat_files(part)
        if not files:
            print(f"[step8] {part}: no feature files, skipped")
            continue
        n = pos = 0
        med, lo, hi = [], {}, {}
        for f in files:
            df = pl.read_parquet(f)
            n += df.height
            pos += int(df["y"].sum())
            med.append(df.filter((pl.col("y") == 1) & (pl.col("q_addr_null") == 0))["addr_tset"].to_numpy())
            mm = df.select([pl.col(c).fill_nan(None).min().alias(f"{c}|lo") for c in FEATURES]
                           + [pl.col(c).fill_nan(None).max().alias(f"{c}|hi") for c in FEATURES]).row(0, named=True)
            for c in FEATURES:
                for d, k, fn in ((lo, "lo", min), (hi, "hi", max)):
                    v = mm[f"{c}|{k}"]
                    if v is not None:
                        d[c] = v if c not in d else fn(d[c], v)
            last = df
        m = float(np.median(np.concatenate(med)))
        const = [c for c in FEATURES if c not in lo or lo[c] == hi[c]]
        print(f"[step8] {part}: pairs {n:,}  positive rate {pos / n:.4f}  "
              f"positives' median addr token_set (non-null) {m:.1f}  constant features {const}")
        ok &= m >= 90 and not const
    if last is None:
        raise SystemExit("no feature files")
    s1 = io.read_source("train", 1)
    q = io.read_queries_raw("train")
    for y in (1, 0):
        samp = last.filter(pl.col("y") == y).sample(n_show, seed=C_SEED)
        print(f"--- {n_show} random {'positives' if y else 'negatives'} ({part}): name_tset addr_tset blk_rank")
        for r in samp.iter_rows(named=True):
            a, b = s1.row(r["s1_row"], named=True), q.row(r["q_row"], named=True)
            print(f"  {r['name_tset']:5.1f} {r['addr_tset'] if r['addr_tset'] == r['addr_tset'] else float('nan'):5.1f} {int(r['blk_rank']):2d}  "
                  f"{a['business_name']!r} | {a['business_address']!r}  <->  {b['business_name']!r} | {b['business_address']!r}")
    print("STEP 8", "PASS" if ok else "FAIL")


C_SEED = 7
