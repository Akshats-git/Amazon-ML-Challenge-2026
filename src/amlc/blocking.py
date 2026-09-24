"""Candidate generation (blocking): TF-IDF char n-grams + sparse top-k, per country.

Every S2/S3 record ("query") is matched against the S1 records of the same country and keeps
its `top_k` most similar S1 entities above `min_sim`. Querying from the S2/S3 side is natural here
because each S2/S3 record belongs to at most one S1 entity (verified on the train GT).

Similarity = name_weight * cos(name_core) + (1 - name_weight) * cos(addr_norm), computed in one
sparse product by stacking sqrt-weighted, L2-normalised name and address vectors.

Works on integer row positions, not string ids:
    s1n : normalised S1 frame        (row position = si)
    qn  : normalised concat(S2, S3)  (row position = qi)
    ->  pairs frame qi, si, blk_score, blk_rank  (rank 1 = best S1 for that query)
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from tqdm.auto import tqdm

from .config import COUNTRY_COL


def _vectorizer(ngram=(3, 3)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2, sublinear_tf=True,
                           dtype=np.float32)


def _stack(name_m, addr_m, w):
    return sp.hstack([name_m * np.float32(np.sqrt(w)), addr_m * np.float32(np.sqrt(1 - w))], format="csr")


def block(
    s1n: pl.DataFrame,
    qn: pl.DataFrame,
    top_k: int = 5,
    min_sim: float = 0.3,
    name_weight: float = 0.5,
    chunk: int = 250_000,
    n_threads: int = 8,
) -> pl.DataFrame:
    s1_country = s1n[COUNTRY_COL].fill_null("?").to_numpy()
    q_country = qn[COUNTRY_COL].fill_null("?").to_numpy()
    parts = []
    for country in sorted(set(s1_country)):
        si_idx = np.flatnonzero(s1_country == country)
        qi_idx = np.flatnonzero(q_country == country)
        if len(qi_idx) == 0:
            continue
        s1c = s1n[si_idx]
        vn, va = _vectorizer(), _vectorizer()
        S = _stack(vn.fit_transform(s1c["name_core"].to_list()), va.fit_transform(s1c["addr_norm"].to_list()),
                   name_weight)
        ST = S.T.tocsr()
        for start in tqdm(range(0, len(qi_idx), chunk), desc=f"block {country}"):
            idx = qi_idx[start:start + chunk]
            qc = qn[idx]
            Q = _stack(vn.transform(qc["name_core"].to_list()), va.transform(qc["addr_norm"].to_list()), name_weight)
            C = sp_matmul_topn(Q, ST, top_n=top_k, threshold=min_sim, sort=True, n_threads=n_threads).tocoo()
            parts.append(pl.DataFrame({
                "qi": idx[C.row].astype(np.int32),
                "si": si_idx[C.col].astype(np.int32),
                "blk_score": C.data.astype(np.float32),
            }))
    pairs = pl.concat(parts) if parts else pl.DataFrame(schema={"qi": pl.Int32, "si": pl.Int32, "blk_score": pl.Float32})
    return pairs.with_columns(
        blk_rank=pl.col("blk_score").rank("ordinal", descending=True).over("qi").cast(pl.UInt8)
    )


def to_id_pairs(pairs: pl.DataFrame, s1n: pl.DataFrame, qn: pl.DataFrame) -> pl.DataFrame:
    """Integer pairs -> official-id pairs (s1_id, cand_id) for metrics / output files."""
    return pl.DataFrame({
        "s1_id": s1n["entity_id"].gather(pairs["si"]),
        "cand_id": qn["entity_id"].gather(pairs["qi"]),
    })
