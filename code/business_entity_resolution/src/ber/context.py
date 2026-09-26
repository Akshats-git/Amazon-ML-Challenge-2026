"""Section 7.8 stage-2 context features computed from p1 over ALL pairs of a partition (numpy, never stored).

Per query: q_p1_max, q_p1_second, q_margin, pair_rank_in_q, is_q_argmax.
Per S1 s over queries whose p1-argmax is s (self excluded): s_cnt_argmax_ge50, s_sum_p1, s_max_p1_other,
pair_rank_in_s, s_cnt_same_source_ge50.
Ctx2 keeps only per-query / per-S1 arrays plus p1 and the within-query rank (~0.5 GB for a 47M-pair partition);
rows(off, keys) returns the 11 features for a slice of pairs in feature-file order.
"""
import numpy as np

CTX2_FEATURES = ["p1", "q_p1_max", "q_p1_second", "q_margin", "pair_rank_in_q", "is_q_argmax",
                 "s_cnt_argmax_ge50", "s_sum_p1", "s_max_p1_other", "pair_rank_in_s", "s_cnt_same_source_ge50"]
_LO32 = np.int64(0xFFFFFFFF)


def _desc_key(s, p):
    """int64 key ordering by S1 ascending, then p1 descending (p1 >= 0, so its float bits are monotone)."""
    return (np.asarray(s, np.int64) << 32) | (_LO32 - np.asarray(p, np.float32).view(np.uint32).astype(np.int64))


class Ctx2:
    def __init__(self, q_row, s1_row, p1, is_s3):
        """All pairs of one partition in feature-file order: rows grouped by query, q_row non-decreasing."""
        n = len(p1)
        assert len(q_row) == len(s1_row) == len(is_s3) == n and n > 0
        assert (q_row[1:] >= q_row[:-1]).all(), "pairs not grouped by query"
        p1 = np.asarray(p1, np.float32)
        new = np.r_[True, q_row[1:] != q_row[:-1]]
        starts = np.flatnonzero(new)
        gid = (np.cumsum(new) - 1).astype(np.int32)
        order = np.lexsort((-p1, gid))  # by query, p1 descending; ties keep blk_rank order
        rank = np.empty(n, np.int32)
        rank[order] = np.arange(n, dtype=np.int32) - starts[gid[order]].astype(np.int32) + 1
        self.rank_q = rank.astype(np.int16)
        am = order[starts]  # argmax row of each query
        sizes = np.diff(np.r_[starts, n])
        self.uq = q_row[starts]
        self.q_max = p1[am]
        self.q_second = np.where(sizes > 1, p1[order[np.minimum(starts + 1, n - 1)]], 0).astype(np.float32)
        del order, rank, gid
        # per S1, over argmax rows
        a_s, a_p, a_s3 = s1_row[am], p1[am], is_s3[am] == 1
        m = int(s1_row.max()) + 1
        ge = a_p >= 0.5
        self.cnt50 = np.bincount(a_s, weights=ge, minlength=m)
        self.cnt50_s3 = np.bincount(a_s, weights=ge & a_s3, minlength=m)
        self.sum_p1 = np.bincount(a_s, weights=a_p, minlength=m)
        self.KA = np.sort(_desc_key(a_s, a_p))
        ks = self.KA >> 32
        kp = (_LO32 - (self.KA & _LO32)).astype(np.uint32).view(np.float32)
        first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        self.top1 = np.zeros(m, np.float32)
        self.top2 = np.zeros(m, np.float32)
        self.top1[ks[first]] = kp[first]
        nxt = np.minimum(first + 1, len(ks) - 1)
        has2 = (first + 1 < len(ks)) & (ks[nxt] == ks[first])
        self.top2[ks[first[has2]]] = kp[first[has2] + 1]
        self.p1 = p1

    def rows(self, off, q_row, s1_row, is_s3) -> np.ndarray:
        """(n, 11) float32 CTX2_FEATURES for pairs off .. off+n-1 (their keys given, same order as the constructor's)."""
        n = len(q_row)
        p, rq = self.p1[off:off + n], self.rank_q[off:off + n]
        g = np.searchsorted(self.uq, q_row)
        assert (self.uq[g] == q_row).all(), "rows do not belong to this partition"
        a = rq == 1
        s = np.asarray(s1_row, np.int64)
        self50 = (a & (p >= 0.5)).astype(np.float64)
        cnt50, cnt50_s3 = self.cnt50[s], self.cnt50_s3[s]
        top1 = self.top1[s]
        seg = np.searchsorted(self.KA, s << 32, "left")
        key = _desc_key(s, p)
        n_gt = np.searchsorted(self.KA, key, "left") - seg  # argmax queries of s with a higher p1
        n_ge = np.searchsorted(self.KA, key, "right") - seg  # ... with p1 >= this pair's
        out = np.empty((n, len(CTX2_FEATURES)), np.float32)
        out[:, 0] = p
        out[:, 1] = self.q_max[g]
        out[:, 2] = self.q_second[g]
        out[:, 3] = out[:, 1] - out[:, 2]
        out[:, 4] = rq
        out[:, 5] = a
        out[:, 6] = cnt50 - self50
        out[:, 7] = self.sum_p1[s] - np.where(a, p, 0)
        out[:, 8] = np.where(a & (p >= top1), self.top2[s], top1)
        out[:, 9] = np.where(a, n_gt, n_ge) + 1
        out[:, 10] = np.where(np.asarray(is_s3) == 1, cnt50_s3, cnt50 - cnt50_s3) - self50
        return out
