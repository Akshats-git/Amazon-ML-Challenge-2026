"""Section 7.2: name/address transliteration dictionaries learned from train GT only.

name: true pairs whose query raw NAME is non-Latin; equal token counts -> positional alignment q_tok -> s1_tok.
addr: true pairs whose query raw ADDRESS is non-Latin; |A| = |B| = 1 residual tokens -> (a -> b).
Keep q_tok -> most frequent target when count >= 3 and share >= 0.5.
"""
import json
from collections import Counter, defaultdict

import polars as pl

from . import config as C
from . import io
from . import normalize as N
from .pools import s1_half


def _select(counts: dict, min_count=C.TRANSLIT_MIN_COUNT, purity=C.TRANSLIT_MIN_PURITY) -> dict:
    out = {}
    for k, c in counts.items():
        tgt, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= purity:
            out[k] = tgt
    return out


def _true_pairs_nonlatin():
    """Train true pairs where the query name or address is non-Latin, with raw query text and S1 norm text."""
    s1 = pl.read_parquet(io.norm_path("train", 1), columns=["entity_id", "name_norm", "addr_norm"])
    s1 = s1.with_columns(half=pl.Series(s1_half(s1["entity_id"])))
    q = io.read_queries_raw("train").select("entity_id", "business_name", "business_address")
    pat = "[^\u0000-ɏ -⃏]"
    q = q.filter(pl.col("business_name").str.contains(pat) | pl.col("business_address").str.contains(pat))
    gt = io.read_gt_pairs()
    j = (gt.join(q.rename({"entity_id": "cand_id"}), on="cand_id")
         .join(s1.rename({"entity_id": "s1_id"}), on="s1_id"))
    j = j.with_columns(
        name_nl=pl.col("business_name").str.contains(pat),
        addr_nl=pl.col("business_address").str.contains(pat),
    )
    return j


def learn(pairs: pl.DataFrame):
    name_counts, addr_counts = defaultdict(Counter), defaultdict(Counter)
    for raw, s1n in pairs.filter("name_nl").select("business_name", "name_norm").iter_rows():
        qt, st = N.name_tokens_pre(raw), s1n.split()
        if len(qt) == len(st):
            for a, b in zip(qt, st):
                name_counts[a][b] += 1
    for raw, s1a in pairs.filter("addr_nl").select("business_address", "addr_norm").iter_rows():
        qt, st = N.addr_tokens_pre(raw), s1a.split()
        A, B = set(qt) - set(st), set(st) - set(qt)
        if len(A) == 1 and len(B) == 1:
            addr_counts[A.pop()][B.pop()] += 1
    return _select(name_counts), _select(addr_counts)


def _share_core_token(pairs: pl.DataFrame, name_dict: dict) -> float:
    N._NAME_DICT = name_dict
    hits = n = 0
    for raw, s1n in pairs.select("business_name", "name_norm").iter_rows():
        qcore = set(N._name_post(N.name_tokens_pre(raw), True)[1].split())
        scoretoks = set(N._name_post(s1n.split(), False)[1].split())
        hits += bool(qcore & scoretoks)
        n += 1
    return hits / max(n, 1)


def _test_coverage(name_dict: dict) -> float:
    pat = "[^\u0000-ɏ -⃏]"
    tot = cov = 0
    for src in (2, 3):
        df = io.read_source("test", src).filter(pl.col("business_name").str.contains(pat))
        for raw in df["business_name"].to_list():
            for t in N.name_tokens_pre(raw):
                tot += 1
                cov += t in name_dict
    return cov / max(tot, 1)


def learn_and_check():
    pairs = _true_pairs_nonlatin()
    print(f"non-Latin true pairs: name {pairs['name_nl'].sum():,}  addr {pairs['addr_nl'].sum():,}")
    # held-out check: learn on S1 half 0, evaluate on half 1
    nd0, ad0 = learn(pairs.filter(pl.col("half") == 0))
    held = pairs.filter((pl.col("half") == 1) & pl.col("name_nl"))
    before = _share_core_token(held, {})
    after = _share_core_token(held, nd0)
    print(f"held-out name shares a core token: before {before:.4f} -> after {after:.4f} (dict learned on half 0: {len(nd0)} entries)")
    # final dictionaries on all train pairs
    nd, ad = learn(pairs)
    cov = _test_coverage(nd)
    n_ident = sum(k == v for k, v in nd.items())
    print(f"name dict {len(nd)} entries ({n_ident} identity), addr dict {len(ad)} entries; test non-Latin name-token coverage {cov:.4f}")
    C.DICT_DIR.mkdir(parents=True, exist_ok=True)
    (C.DICT_DIR / "name.json").write_text(json.dumps(nd, ensure_ascii=False, sort_keys=True, indent=0))
    (C.DICT_DIR / "addr.json").write_text(json.dumps(ad, ensure_ascii=False, sort_keys=True, indent=0))
    print("sample name:", list(nd.items())[:12])
    print("sample addr:", list(ad.items())[:12])
    assert after >= 0.99, "held-out shared-token rate below 0.99"
    assert cov >= 0.80, "test coverage below 0.80"
    return nd, ad
