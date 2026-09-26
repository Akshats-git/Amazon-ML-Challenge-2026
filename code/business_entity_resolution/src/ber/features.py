"""Section 7.6 stage-1 pair features (float32, NaN where undefined, country-agnostic).

Per partition: context features (blocking context, frequencies, per-query flags, OOV) as per-query / per-S1 arrays
gathered chunk by chunk, string features in forked workers that read the partition's text from shared byte blobs.
Output: work/feats/{part}/{country}_part-NNN.parquet (FEAT_CHUNK rows each, sorted by query then blk_rank) with
q_row, s1_row, y (pools only) + FEATURES, float features rounded to 10 mantissa bits.
"""
import gc
import math
import multiprocessing as mp
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from . import config as C
from . import io
from .blocking import cand_path
from .pools import load_partition, partitions

CTX_FEATURES = ["blk_score", "blk_rank", "q_top1_score", "blk_gap", "q_second_score", "q_n_cands", "s1_n_top1", "s1_n_cands",
                "name_oov_frac_q", "s1_name_freq", "q_name_freq", "s1_addr_freq",
                "q_alias", "q_web", "q_idtag", "q_nonlatin", "is_s3", "q_addr_null"]
STR_FEATURES = ["name_ratio", "name_tsort", "name_tset", "name_partial", "name_jw", "name_full_ratio", "sq_ratio", "sq_partial",
                "name_me_q2s", "name_me_s2q", "name_unm_idf_q", "name_unm_idf_s", "name_len_diff",
                "legal_eq", "legal_conflict", "legal_both_empty",
                "addr_ratio", "addr_tset", "addr_ptset", "addr_me_q2s", "addr_me_s2q", "addr_unm_idf_q",
                "addr_bigram_shared", "addr_max_shared_idf", "addr_tok_ratio",
                "num_exact_any", "first_num_exact", "num_soft_any", "num_min_absdiff", "both_have_nums", "num_conflict", "longnum_exact"]
FEATURES = CTX_FEATURES + STR_FEATURES

Q_TEXT = ["name_core", "name_norm", "name_sq", "legal", "alias_l", "alias_r", "addr_norm", "addr_nums"]
S_TEXT = ["name_core", "name_norm", "name_sq", "legal", "addr_norm", "addr_nums"]

_G: dict = {}  # partition context shared with forked workers
_MEMO: dict = {}
NAN = float("nan")


def feat_dir(part):
    return C.FEAT_DIR / part


def feat_files(part, country=None):
    d = feat_dir(part)
    pat = f"{country}_part-*.parquet" if country else "*_part-*.parquet"
    return sorted(d.glob(pat))


# ---------------------------------------------------------------- shared text blobs
def _blob(s: pl.Series):
    lens = s.str.len_bytes().to_numpy().astype(np.int64)
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return "".join(s.to_list()).encode("utf-8"), off


def _dec(blob, idx):
    d, o = blob
    return [d[a:b].decode("utf-8") for a, b in zip(o[idx].tolist(), o[idx + 1].tolist())]


# ---------------------------------------------------------------- soft token matching
def _subseq(short, long):
    it = iter(long)
    return all(ch in it for ch in short)


def _sim_raw(a: str, b: str) -> float:
    """Soft token score for a != b (see module doc / brief 7.6)."""
    ad, bd = a.isdigit(), b.isdigit()
    s_, l_ = (a, b) if len(a) <= len(b) else (b, a)
    if ad or bd:
        if not (ad and bd):
            return 0.0
        if len(s_) >= 2 and (l_.startswith(s_) or l_.endswith(s_)):
            return 0.8
        return 0.7 if abs(int(a[:15]) - int(b[:15])) <= 2 else 0.0
    if s_[0] == l_[0] and len(l_) >= 3 and _subseq(s_, l_):
        return 0.9
    # normalized Levenshtein >= 0.8 needs len(short) >= 4 and a length gap <= 20% of the longer
    if len(s_) >= 4 and 5 * (len(l_) - len(s_)) <= len(l_):
        x = Levenshtein.normalized_similarity(a, b)
        return x if x >= 0.8 else 0.0
    return 0.0


def sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    key = (a, b) if a < b else (b, a)
    v = _MEMO.get(key)
    if v is None:
        v = _MEMO[key] = _sim_raw(a, b)
    return v


MAX_SOFT = 8  # soft-compare at most the 8 highest-IDF leftover tokens per side when both sides are long
MEMO_CAP = 500_000  # ~125 MB per worker; cleared when full (4M entries x 12 workers exhausted a 13 GB laptop)


def me_both(qt, st, idf, maxidf):
    """Soft Monge-Elkan both ways from one leftover similarity matrix (sim is symmetric).
    -> (ME q->s, unmatched idf q, ME s->q, unmatched idf s). Exact matches via sets; leftovers soft-compared."""
    if not qt or not st:
        return NAN, NAN, NAN, NAN
    qset, sset = set(qt), set(st)
    lq = [t for t in qset if t not in sset]
    ls = [u for u in sset if u not in qset]
    bq = dict.fromkeys(lq, 0.0)
    bs = dict.fromkeys(ls, 0.0)
    if lq and ls:
        if len(lq) * len(ls) > MAX_SOFT * MAX_SOFT:
            lq = sorted(lq, key=lambda t: (-idf.get(t, maxidf), t))[:MAX_SOFT]  # token tie-break: hash-seed independent
            ls = sorted(ls, key=lambda t: (-idf.get(t, maxidf), t))[:MAX_SOFT]
        memo = _MEMO
        if len(memo) > MEMO_CAP:
            memo.clear()
        for t in lq:
            t0 = t[0]
            tl = len(t)
            td = t.isdigit()
            for u in ls:
                # cheap zero: different first char, not both numeric, too short / too different in length for Levenshtein
                if u[0] != t0 and not td and (tl < 4 or len(u) < 4 or 5 * abs(tl - len(u)) > max(tl, len(u))):
                    continue
                k = (t, u) if t < u else (u, t)
                v = memo.get(k)
                if v is None:
                    v = memo[k] = _sim_raw(t, u)
                if v > bq[t]:
                    bq[t] = v
                if v > bs[u]:
                    bs[u] = v
    cut = C.UNMATCHED_CUTOFF
    num = den = unm_q = 0.0
    for t in qt:
        w = idf.get(t, maxidf)
        b = 1.0 if t in sset else bq[t]
        num += w * b
        den += w
        if b < cut:
            unm_q += w
    me_q = num / den if den > 0 else NAN
    num = den = unm_s = 0.0
    for u in st:
        w = idf.get(u, maxidf)
        b = 1.0 if u in qset else bs[u]
        num += w * b
        den += w
        if b < cut:
            unm_s += w
    me_s = num / den if den > 0 else NAN
    return me_q, unm_q, me_s, unm_s


def _num_feats(qn, sn):
    """num_exact_any, first_num_exact, num_soft_any, num_min_absdiff, both_have_nums, num_conflict, longnum_exact"""
    if not qn or not sn:
        return NAN, NAN, NAN, NAN, 0.0, 0.0, NAN
    qs, ss = set(qn), set(sn)
    shared = qs & ss
    exact = 1.0 if shared else 0.0
    soft = exact
    mind = 10000
    for a in qs:
        ia = int(a[:15])
        for b in ss:
            d = abs(ia - int(b[:15]))
            if d < mind:
                mind = d
            if not soft:
                s_, l_ = (a, b) if len(a) <= len(b) else (b, a)
                if d <= 2 or (len(s_) >= 2 and (l_.startswith(s_) or l_.endswith(s_))):
                    soft = 1.0
    longnum = 1.0 if any(len(x) >= 5 for x in shared) else 0.0
    return exact, float(qn[0] == sn[0]), soft, float(min(mind, 10000)), 1.0, 1.0 - soft, longnum


def _rf(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32)


# ---------------------------------------------------------------- worker
def _work(args):
    qi, si = args
    G = _G
    n = len(qi)
    uq, inv = np.unique(qi, return_inverse=True)
    us, sinv = np.unique(si, return_inverse=True)
    Q = {c: _dec(G["q_" + c], uq) for c in Q_TEXT}
    S = {c: _dec(G["s_" + c], us) for c in S_TEXT}
    qa = lambda c: [Q[c][j] for j in inv]  # noqa: E731
    sa = lambda c: [S[c][j] for j in sinv]  # noqa: E731
    qcore, score = qa("name_core"), sa("name_core")
    qsq, ssq = qa("name_sq"), sa("name_sq")
    qaddr, saddr = qa("addr_norm"), sa("addr_norm")
    out = {}
    out["name_ratio"] = _rf(qcore, score, fuzz.ratio)
    out["name_tsort"] = _rf(qcore, score, fuzz.token_sort_ratio)
    out["name_tset"] = _rf(qcore, score, fuzz.token_set_ratio)
    out["name_partial"] = _rf(qcore, score, fuzz.partial_ratio)
    out["name_jw"] = _rf(qcore, score, JaroWinkler.normalized_similarity)
    out["name_full_ratio"] = _rf(qa("name_norm"), sa("name_norm"), fuzz.ratio)
    out["sq_ratio"] = _rf(qsq, ssq, fuzz.ratio)
    out["sq_partial"] = _rf(qsq, ssq, fuzz.partial_ratio)
    # alias queries: max over parts
    alias_q = G["q_alias"][uq][inv].astype(bool)
    ai = np.flatnonzero(alias_q)
    if len(ai):
        sc_a = [score[i] for i in ai]
        ssq_a = [ssq[i] for i in ai]
        for part in ("alias_l", "alias_r"):
            pa = [Q[part][inv[i]] for i in ai]
            psq = [x.replace(" ", "") for x in pa]
            for f, scorer in (("name_ratio", fuzz.ratio), ("name_tsort", fuzz.token_sort_ratio), ("name_tset", fuzz.token_set_ratio),
                              ("name_partial", fuzz.partial_ratio), ("name_jw", JaroWinkler.normalized_similarity)):
                out[f][ai] = np.maximum(out[f][ai], _rf(pa, sc_a, scorer))
            out["sq_ratio"][ai] = np.maximum(out["sq_ratio"][ai], _rf(psq, ssq_a, fuzz.ratio))
            out["sq_partial"][ai] = np.maximum(out["sq_partial"][ai], _rf(psq, ssq_a, fuzz.partial_ratio))
    q_null = G["q_addr_null"][uq][inv].astype(bool)
    for f, scorer in (("addr_ratio", fuzz.ratio), ("addr_tset", fuzz.token_set_ratio), ("addr_ptset", fuzz.partial_token_set_ratio)):
        v = _rf(qaddr, saddr, scorer)
        v[q_null] = np.nan
        out[f] = v

    rest = {f: np.full(n, np.nan, np.float32) for f in STR_FEATURES if f not in out}
    nidf, nmax, aidf, amax = G["name_idf"], G["name_maxidf"], G["addr_idf"], G["addr_maxidf"]
    qlegal, slegal, qnums, snums = qa("legal"), sa("legal"), qa("addr_nums"), sa("addr_nums")
    ql, qr = qa("alias_l"), qa("alias_r")
    for p in range(n):
        st = score[p].split()
        parts = [qcore[p]]
        if alias_q[p]:
            parts += [x for x in (ql[p], qr[p]) if x]
        best = None
        for part in parts:
            r = me_both(part.split(), st, nidf, nmax)
            if best is None or (r[0] == r[0] and (best[0] != best[0] or r[0] > best[0])):
                best = r
        me_q, unm_q, me_s, unm_s = best
        rest["name_me_q2s"][p], rest["name_unm_idf_q"][p] = me_q, unm_q
        rest["name_me_s2q"][p], rest["name_unm_idf_s"][p] = me_s, unm_s
        rest["name_len_diff"][p] = len(qcore[p]) - len(score[p])
        lq, ls = set(qlegal[p].split()), set(slegal[p].split())
        rest["legal_eq"][p] = float(bool(lq) and lq == ls)
        rest["legal_conflict"][p] = float(bool(lq) and bool(ls) and not (lq & ls))
        rest["legal_both_empty"][p] = float(not lq and not ls)
        if q_null[p]:
            continue
        qat, sat = qaddr[p].split(), saddr[p].split()
        me_q, unm_q, me_s, _ = me_both(qat, sat, aidf, amax)
        rest["addr_me_q2s"][p], rest["addr_unm_idf_q"][p], rest["addr_me_s2q"][p] = me_q, unm_q, me_s
        qb = set(zip(qat, qat[1:]))
        rest["addr_bigram_shared"][p] = len(qb & set(zip(sat, sat[1:])))
        sh = set(qat) & set(sat)
        rest["addr_max_shared_idf"][p] = max((aidf.get(t, amax) for t in sh), default=0.0)
        rest["addr_tok_ratio"][p] = len(qat) / max(len(sat), 1)
        (rest["num_exact_any"][p], rest["first_num_exact"][p], rest["num_soft_any"][p], rest["num_min_absdiff"][p],
         rest["both_have_nums"][p], rest["num_conflict"][p], rest["longnum_exact"][p]) = _num_feats(qnums[p].split(), snums[p].split())
    out.update(rest)
    return {f: round_mantissa(out[f]) for f in STR_FEATURES}


# ---------------------------------------------------------------- partition-level
FEAT_COLS = ["name_core", "name_norm", "name_sq", "legal", "alias_l", "alias_r", "addr_norm", "addr_nums",
             "alias", "web", "idtag", "nonlatin", "addr_null"]


def round_mantissa(a, drop=C.FEAT_DROP_BITS) -> np.ndarray:
    """Round float32 values to (23 - drop) mantissa bits, NaN/inf untouched: stored features compress ~27% better."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    r = (u + np.uint32(1 << (drop - 1))) & np.uint32((0xFFFFFFFF << drop) & 0xFFFFFFFF)
    return np.where(np.isfinite(a), r, u).view(np.float32)


def _idf(series: pl.Series, n_docs: int):
    df = (pl.DataFrame({"t": series.str.split(" ")}).with_row_index("d").explode("t")
          .filter(pl.col("t") != "").unique(["d", "t"]).group_by("t").len())
    idf = (np.log((1 + n_docs) / (1 + df["len"].to_numpy())) + 1).astype(np.float64)
    return dict(zip(df["t"].to_list(), idf.tolist())), math.log(1 + n_docs) + 1


def q_true_s1(split="train") -> np.ndarray:
    """q_row -> its true s1_row, -1 for distractors (GT fact: every query matches at most one S1)."""
    n_q = sum(pl.scan_parquet(io.norm_path(split, s)).select(pl.len()).collect().item() for s in (2, 3))
    gt = io.gt_rows()
    q, s = gt["q_row"].to_numpy(), gt["s1_row"].to_numpy()
    assert len(np.unique(q)) == len(q), "a query matches two S1"
    out = np.full(n_q, -1, np.int32)
    out[q] = s
    return out


class PairContext:
    """Stage-1 context features of one partition, kept as per-query / per-S1 arrays (a few hundred MB even for the
    47M-pair partitions); rows(o, n) gathers them for a slice of pairs. Pairs are sorted by (qi, blk_rank), and
    blk_rank 1 is the highest blk_score (block_one's output order). Same values as the former polars joins."""

    def __init__(self, qi, si, blk_score, blk_rank, s1p: pl.DataFrame, qp: pl.DataFrame):
        nq, ns = qp.height, s1p.height
        self.qi, self.si, self.score, self.rank = qi, si, blk_score, blk_rank
        new = np.r_[True, qi[1:] != qi[:-1]]
        assert (np.diff(blk_score)[~new[1:]] <= 0).all(), "blk_score not descending within a query"
        starts = np.flatnonzero(new)
        sizes = np.diff(np.r_[starts, len(qi)])
        qid = qi[starts]
        second = np.zeros(len(starts), np.float32)
        m = sizes > 1
        second[m] = blk_score[starts[m] + 1]
        self.q_top1, self.q_second, self.q_n = (np.zeros(nq, np.float32) for _ in range(3))
        self.q_top1[qid], self.q_second[qid], self.q_n[qid] = blk_score[starts], second, sizes
        self.s_top1 = np.bincount(si[blk_rank == 1], minlength=ns).astype(np.float32)
        self.s_cands = np.bincount(si, minlength=ns).astype(np.float32)
        # frequencies within the partition's S1
        f = s1p.select(nf=pl.len().over("name_core"), af=pl.len().over("addr_norm"))
        self.s_nf, self.s_af = f["nf"].to_numpy().astype(np.float32), f["af"].to_numpy().astype(np.float32)
        name_cnt = s1p.group_by("name_core").len()
        qf = qp.select("name_core").with_row_index("qi").join(name_cnt, on="name_core", how="left").sort("qi")
        self.q_nf = qf["len"].fill_null(0).to_numpy().astype(np.float32)
        # share of the query's name_norm tokens absent from the partition's S1 name vocabulary (NaN if no tokens)
        vocab = pl.DataFrame({"t": s1p["name_norm"].str.split(" ")}).explode("t").unique().with_columns(inv=pl.lit(1))
        oov = (qp.select(pl.col("name_norm").str.split(" ").alias("t")).with_row_index("qi").explode("t")
               .filter(pl.col("t") != "").join(vocab, on="t", how="left")
               .group_by("qi").agg((1 - pl.col("inv").fill_null(0).mean()).alias("oov")))
        self.q_oov = np.full(nq, np.nan, np.float32)
        self.q_oov[oov["qi"].to_numpy()] = oov["oov"].to_numpy()
        self.q_flags = {k: qp[c].cast(pl.Float32).to_numpy() for k, c in (
            ("q_alias", "alias"), ("q_web", "web"), ("q_idtag", "idtag"), ("q_nonlatin", "nonlatin"),
            ("is_s3", "is_s3"), ("q_addr_null", "addr_null"))}

    def rows(self, o, n) -> dict:
        q, s, sc = self.qi[o:o + n], self.si[o:o + n], self.score[o:o + n]
        top1 = self.q_top1[q]
        out = {"blk_score": sc, "blk_rank": self.rank[o:o + n].astype(np.float32), "q_top1_score": top1,
               "blk_gap": top1 - sc, "q_second_score": self.q_second[q], "q_n_cands": self.q_n[q],
               "s1_n_top1": self.s_top1[s], "s1_n_cands": self.s_cands[s], "name_oov_frac_q": self.q_oov[q],
               "s1_name_freq": self.s_nf[s], "q_name_freq": self.q_nf[q], "s1_addr_freq": self.s_af[s]}
        out.update({k: v[q] for k, v in self.q_flags.items()})
        return {f: out[f] for f in CTX_FEATURES}


def load_pairs(part, country, s1_rows, q_rows):
    """Candidate pairs of a partition as numpy arrays sorted by (qi, blk_rank), plus local keys qi/si."""
    cands = pl.read_parquet(cand_path(part, country))
    q_row, s1_row = cands["q_row"].to_numpy(), cands["s1_row"].to_numpy()
    score, rank = cands["blk_score"].to_numpy(), cands["blk_rank"].to_numpy()
    del cands
    qi = np.searchsorted(q_rows, q_row).astype(np.int32)
    si = np.searchsorted(s1_rows, s1_row).astype(np.int32)
    assert (q_rows[qi] == q_row).all() and (s1_rows[si] == s1_row).all(), "candidate outside its partition"
    same = qi[1:] == qi[:-1]
    if not ((qi[1:] >= qi[:-1]).all() and (rank[1:][same] > rank[:-1][same]).all()):
        order = np.lexsort((rank, qi))
        q_row, s1_row, score, rank, qi, si = (a[order] for a in (q_row, s1_row, score, rank, qi, si))
    return q_row, s1_row, score, rank, qi, si


def build_partition(part, country, split, q_true):
    t0 = time.time()
    s1_rows, q_rows = load_partition(part, country)
    s1p = io.load_norm(split, "s1", columns=FEAT_COLS)[s1_rows]  # only this partition's rows stay in memory
    qp = io.load_norm(split, "q", columns=FEAT_COLS)[q_rows]
    q_row, s1_row, score, rank, qi, si = load_pairs(part, country, s1_rows, q_rows)
    ctx = PairContext(qi, si, score, rank, s1p, qp)
    keys = {"q_row": q_row, "s1_row": s1_row}
    if q_true is not None:
        keys["y"] = (q_true[q_row] == s1_row).astype(np.int8)
    # partition context for workers
    _G.clear()
    for c in Q_TEXT:
        _G["q_" + c] = _blob(qp[c])
    for c in S_TEXT:
        _G["s_" + c] = _blob(s1p[c])
    _G["q_alias"] = qp["alias"].to_numpy().astype(np.int8)
    _G["q_addr_null"] = qp["addr_null"].to_numpy().astype(np.int8)
    _G["name_idf"], _G["name_maxidf"] = _idf(s1p["name_norm"], s1p.height)
    _G["addr_idf"], _G["addr_maxidf"] = _idf(s1p["addr_norm"], s1p.height)
    del s1p, qp
    gc.collect()
    print(f"[feats] {part} {country}: {len(qi):,} pairs; context done in {time.time() - t0:.1f}s")
    W = C.WORKER_CHUNK
    tasks = [(qi[o:o + W], si[o:o + W]) for o in range(0, len(qi), W)]
    d = feat_dir(part)
    d.mkdir(parents=True, exist_ok=True)
    for f in feat_files(part, country):
        f.unlink()
    buf, done, part_no = [], 0, 0
    t1 = time.time()
    with mp.get_context("fork").Pool(C.N_JOBS) as pool:
        for k, res in enumerate(pool.imap(_work, tasks)):
            o, n = k * W, len(res[STR_FEATURES[0]])
            cols = {c: a[o:o + n] for c, a in keys.items()}
            cols.update({f: round_mantissa(v) for f, v in ctx.rows(o, n).items()})
            cols.update(res)
            buf.append(pl.DataFrame(cols))
            done += n
            if sum(b.height for b in buf) >= C.FEAT_CHUNK or k == len(tasks) - 1:
                pl.concat(buf).write_parquet(d / f"{country}_part-{part_no:03d}.parquet", **C.PARQUET_KW)
                part_no += 1
                buf = []
                el = time.time() - t1
                print(f"  {done:,}/{len(qi):,} pairs  {done / max(el, 1e-9):,.0f} pairs/s")
    _G.clear()
    print(f"[feats] {part} {country}: done in {time.time() - t0:.1f}s")


def build_features(parts=("P0", "P1", "test")):
    for split in ("train", "test"):
        todo = [(p, c) for p, c in partitions(parts) if (p == "test") == (split == "test")]
        if not todo:
            continue
        q_true = q_true_s1() if split == "train" else None
        for part, c in todo:
            build_partition(part, c, split, q_true)
            gc.collect()


def load_cols(part, country, columns) -> pl.DataFrame:
    """Selected columns of all feature files of (part, country), in file order (the row order of OOF arrays)."""
    return pl.concat([pl.read_parquet(f, columns=columns) for f in feat_files(part, country)])
