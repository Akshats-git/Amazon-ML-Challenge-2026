"""Sections 7.3-7.4: per-(partition, country) sparse TF-IDF blocking, query-side top-K by cosine.

Blocks fit on the partition's S1 only (sublinear tf, float32, idf = ln((1+N)/(1+df)) + 1, absolute DF cap):
  N = word unigrams of name_norm, A = word uni+bigrams of addr_norm, C = char 4-grams of name_sq.
Each block is L2-normalized, scaled by sqrt(w), hstacked. Output work/cands/{part}_{country}.parquet:
  q_row, s1_row (global int32 keys), blk_score (float32), blk_rank (int8, 1 = best).
"""
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config as C
from . import io
from .metrics import blocking_report
from .pools import load_eval_s1, load_partition, partitions

BLOCK_COLS = ["name_norm", "addr_norm", "name_sq"]


def cand_path(part, country):
    return C.CAND_DIR / f"{part}_{country}.parquet"


def _vectorizers():
    common = dict(lowercase=False, sublinear_tf=True, dtype=np.float32, max_df=C.DF_CAP)
    return [
        ("name_norm", TfidfVectorizer(token_pattern=r"\S+", min_df=1, **common)),
        ("addr_norm", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=1, **common)),
        ("name_sq", TfidfVectorizer(analyzer="char", ngram_range=(4, 4), min_df=2, **common)),
    ]


def block_one(s1df: pl.DataFrame, qdf: pl.DataFrame, s1_rows, q_rows, k=C.BLOCK_K, verbose=True) -> pl.DataFrame:
    """s1df/qdf: normalized frames of the partition (row i <-> s1_rows[i] / q_rows[i])."""
    t0 = time.time()
    vecs = _vectorizers()
    S_blocks, fitted = [], []
    for (col, v), w in zip(vecs, C.BLOCK_WEIGHTS):
        try:
            S_blocks.append(v.fit_transform(s1df[col].to_list()) * np.float32(np.sqrt(w)))
            fitted.append((col, v, w))
        except ValueError:  # empty vocabulary after pruning (tiny partitions only)
            print(f"  block {col}: empty vocabulary, skipped")
    S = sp.hstack(S_blocks, format="csr", dtype=np.float32)
    ST = S.T.tocsr()
    if verbose:
        print(f"  fit: S1 {S.shape[0]:,} x {S.shape[1]:,} feats ({', '.join(f'{c}={len(v.vocabulary_):,}' for c, v, _ in fitted)}) "
              f"nnz/row {S.nnz / max(S.shape[0], 1):.1f}  {time.time() - t0:.1f}s")
    out = []
    t1 = time.time()
    for off in range(0, qdf.height, C.BLOCK_CHUNK):
        sl = qdf.slice(off, C.BLOCK_CHUNK)
        Q = sp.hstack([v.transform(sl[col].to_list()) * np.float32(np.sqrt(w)) for col, v, w in fitted], format="csr", dtype=np.float32)
        R = sp_matmul_topn(Q, ST, top_n=k, threshold=0.0, sort=True, n_threads=C.N_JOBS).tocsr()
        R.eliminate_zeros()
        counts = np.diff(R.indptr)
        qi = np.repeat(np.arange(R.shape[0]), counts)
        # rank within each row: data is sorted descending per row
        rank = np.arange(R.nnz) - np.repeat(R.indptr[:-1], counts) + 1
        out.append(pl.DataFrame({
            "q_row": q_rows[off + qi].astype(np.int32),
            "s1_row": s1_rows[R.indices].astype(np.int32),
            "blk_score": R.data.astype(np.float32),
            "blk_rank": rank.astype(np.int8),
        }))
    cands = pl.concat(out) if out else pl.DataFrame(schema={"q_row": pl.Int32, "s1_row": pl.Int32, "blk_score": pl.Float32, "blk_rank": pl.Int8})
    dt = time.time() - t1
    if verbose:
        print(f"  query: {qdf.height:,} queries -> {cands.height:,} pairs in {dt:.1f}s ({1000 * dt / max(qdf.height, 1):.3f} ms/query)")
    return cands


def truth_for(s1_rows, q_rows) -> pl.DataFrame:
    gt = io.gt_rows()
    return gt.filter(pl.col("s1_row").is_in(s1_rows) & pl.col("q_row").is_in(q_rows))


def block_partitions(parts=("P0", "P1", "test")):
    C.CAND_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    test_pairs = 0
    for split in ("train", "test"):
        todo = [(p, c) for p, c in partitions(parts) if (p == "test") == (split == "test")]
        if not todo:
            continue
        s1n = io.load_norm(split, "s1", columns=BLOCK_COLS)
        qn = io.load_norm(split, "q", columns=BLOCK_COLS)
        gt = io.gt_rows() if split == "train" else None
        for part, c in todo:
            s1_rows, q_rows = load_partition(part, c)
            print(f"[block] {part} {c}: S1 {len(s1_rows):,}  queries {len(q_rows):,}")
            cands = block_one(s1n[s1_rows], qn[q_rows], s1_rows, q_rows)
            cands.write_parquet(cand_path(part, c))
            if split == "test":
                test_pairs += cands.height
                assert test_pairs <= C.TEST_PAIR_GUARD, f"test candidate volume {test_pairs:,} > guard"
            else:
                truth = gt.filter(pl.col("s1_row").is_in(s1_rows) & pl.col("q_row").is_in(q_rows))
                reports[(part, c)] = blocking_report(cands, truth, load_eval_s1(part, c), label=f"{part} {c}")
            del cands
    if test_pairs:
        print(f"[block] test total pairs {test_pairs:,}")
    return reports


def handmap_ab(frac=0.10):
    """Step 7: recall@10 on a 10% query subsample of P1 with vs without hand maps (normalization re-run in-process)."""
    from . import normalize as N

    rng = np.random.default_rng(C.SEED)
    gt = io.gt_rows()
    res = {}
    for c in partitions(["P1"]):
        s1_rows, q_rows = load_partition(*c)
        q_sub = np.sort(q_rows[rng.random(len(q_rows)) < frac])
        truth = gt.filter(pl.col("s1_row").is_in(s1_rows) & pl.col("q_row").is_in(q_sub))
        raw_s1 = io.read_source("train", 1)[s1_rows]
        raw_q = io.read_queries_raw("train")[q_sub]
        for flag in (False, True):
            C.USE_HAND_MAPS = flag
            N.C.USE_HAND_MAPS = flag
            s1n, qn = N.normalize_frame(raw_s1), N.normalize_frame(raw_q)
            cands = block_one(s1n, qn, s1_rows, q_sub, verbose=False)
            r = blocking_report(cands, truth, s1_rows, label=f"P1 {c[1]} hand_maps={flag}")
            res[(c[1], flag)] = r["R@10"]
        C.USE_HAND_MAPS = False
        print(f"hand-map delta R@10 {c[1]}: {100 * (res[(c[1], True)] - res[(c[1], False)]):+.3f}pp")
    return res
