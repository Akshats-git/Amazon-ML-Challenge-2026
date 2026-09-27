"""Section 7.10 / Step 13-14: test inference, output files, asserts, label-free monitors, validator, packaging.

Memory: decisions run per country on numpy arrays; candidate_pairs.tsv (~100M ids) is written from int32 row keys
sorted once, 50k S1 rows at a time. The official validator builds Python sets of every candidate id (~10 GB here), so
it checks matching_results.tsv (with --check-ids) and check_candidates() streams the candidate file with the same rules.
"""
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C
from . import io
from .blocking import cand_path
from .decide import argmax_rows, assign, decide, load_thresholds
from .features import load_cols
from .model import _model_path, load_booster, load_pred, predict_part
from .pools import countries
from .rescue import rescue_top, unseen_countries

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")
N_TEST_S1 = 1_732_544


def _n_s2():
    """Number of test S2 rows (q_row >= it is S3) when the per-source caps are on, else None."""
    return pl.scan_parquet(io.norm_path("test", 2)).select(pl.len()).collect().item() if C.CAPS else None


def _key(s1, q):
    return (np.asarray(s1, np.int64) << 32) | np.asarray(q, np.int64)


def predict_and_write(tier0=False, tag="main", check_ids=False):
    has_s2 = all(_model_path(tag, 2, p).exists() for p in ("P0", "P1")) and not tier0
    pools1 = ("P0",) if tier0 else ("P0", "P1")
    m1 = [load_booster(tag, 1, p) for p in pools1]
    m2 = [load_booster(tag, 2, p) for p in ("P0", "P1")] if has_s2 else []
    T = (0.76, 0.76) if tier0 else load_thresholds(tag)
    kind = "p2" if has_s2 else "p1"
    print(f"[predict] tag {tag}: stage-1 models {pools1}, stage-2 {'yes' if has_s2 else 'no'}, xfeats {C.USE_XFEATS}, thresholds {T}")
    s1c = pl.read_parquet(io.norm_path("test", 1), columns=["country"])["country"]
    n_s2, unseen = _n_s2(), unseen_countries()
    print(f"[predict] rescue {C.FR_RESCUE} for {sorted(unseen)}; thresholds there {C.T_UNSEEN or T}; per-source caps {C.CAPS}")
    preds, cs_, cq_, mon = [], [], [], []
    for c in countries("test"):
        predict_part(m1, "test", c, 1, tag, "p1")
        if has_s2:
            predict_part(m2, "test", c, 2, tag, "p2")
        k = load_cols("test", c, ["q_row", "s1_row"])
        p = load_pred(tag, kind, "test", c)
        blk = pl.scan_parquet(cand_path("test", c))
        n_blk = blk.select(pl.len()).collect().item()
        assert len(p) == k.height == n_blk, f"{c}: {len(p):,} scores, {k.height:,} feature rows, {n_blk:,} candidates"
        q, s = k["q_row"].to_numpy(), k["s1_row"].to_numpy()
        top = argmax_rows(q, s, p)
        Tc = C.T_UNSEEN if (c in unseen and C.T_UNSEEN) else T
        if C.FR_RESCUE and c in unseen:
            top = rescue_top(top, q, s, c, Tc)
        pred = decide(assign(top), *Tc, n_s2=n_s2)
        ck = np.sort(_key(s, q))  # matches must be scored candidate pairs
        pk = _key(pred["s1_row"], pred["q_row"])
        pos = np.minimum(np.searchsorted(ck, pk), len(ck) - 1)
        assert (ck[pos] == pk).all(), f"{c}: matched pairs outside the candidate set"
        assert (np.diff(ck) > 0).all(), f"{c}: duplicate candidate pairs"
        n_s1 = int((s1c == c).sum())
        mon.append(dict(country=c, n_s1=n_s1, links_per_s1=pred.height / n_s1,
                        empty_share=1 - pred["s1_row"].n_unique() / n_s1,
                        median_top1_blk=blk.filter(pl.col("blk_rank") == 1).select(pl.col("blk_score").median()).collect().item(),
                        share_queries_linked=pred.height / max(top.height, 1)))
        preds.append(pred)
        cs_.append(s)
        cq_.append(q)
        del k, p, top, ck, pk, pos
    pred = pl.concat(preds)
    with pl.Config(float_precision=4):
        print(pl.DataFrame(mon))
    cands = not tier0 and C.WRITE_CANDIDATES
    write_outputs(pred, np.concatenate(cs_) if cands else None, np.concatenate(cq_) if cands else None)
    del cs_, cq_
    rescore_links(pred)
    validate(check_ids)


def blend_and_write(tags, check_ids=False):
    """Average the final p2 of several runs (each already cross-fitted and scored on test), tune (T1, T2) on the averaged
    pooled OOF, and write the blended test decision. The blend lives under tag 'blend-<tags>' (thresholds only)."""
    from . import model as M
    from .decide import save_thresholds, tune_thresholds

    name = "blend-" + "+".join(tags)
    orig = M.load_pred

    def avg(tag, kind, part, c):
        if tag != name:
            return orig(tag, kind, part, c)
        return np.mean([orig(t, "p2", part, c) for t in tags], axis=0).astype(np.float32)

    M.load_pred = avg
    for t in tags:
        M.evaluate(t, "p2", T=load_thresholds(t), decomp=False, label=f"{t} alone")
    _, top, truth, uni = M.evaluate(name, "p2", T=(0.76, 0.76), decomp=False, label=f"{name} pre-tune")
    (f, T), _ = tune_thresholds(top, truth, uni)
    del top
    save_thresholds(*T, extra=dict(kind="p2", F=f, tags=list(tags)), tag=name)
    M.evaluate(name, "p2", T=T, label=f"{name} tuned")
    print(f"[blend] {name}: OOF F={f:.5f} at {T}")
    s1c = pl.read_parquet(io.norm_path("test", 1), columns=["country"])["country"]
    n_s2, unseen = _n_s2(), unseen_countries()
    print(f"[blend] rescue {C.FR_RESCUE} for {sorted(unseen)}; thresholds there {C.T_UNSEEN or T}; per-source caps {C.CAPS}")
    preds, cs_, cq_ = [], [], []
    for c in countries("test"):
        k = load_cols("test", c, ["q_row", "s1_row"])
        q, s = k["q_row"].to_numpy(), k["s1_row"].to_numpy()
        Tc = T
        if c in unseen and C.BLEND_TAGS_UNSEEN:  # other members (and their own OOF-tuned thresholds) for unseen countries
            ut = C.BLEND_TAGS_UNSEEN
            p = np.mean([orig(t, "p2", "test", c) for t in ut], axis=0).astype(np.float32)
            Tc = load_thresholds("blend-" + "+".join(ut)) if len(ut) > 1 else load_thresholds(ut[0])
            print(f"[blend] test {c}: members {ut}, thresholds {Tc}")
        else:
            p = avg(name, "p2", "test", c)
        top = argmax_rows(q, s, p)
        Tc = C.T_UNSEEN if (c in unseen and C.T_UNSEEN) else Tc
        if C.FR_RESCUE and c in unseen:
            top = rescue_top(top, q, s, c, Tc)
        pred = decide(assign(top), *Tc, n_s2=n_s2)
        n_s1 = int((s1c == c).sum())
        print(f"[blend] test {c} T={Tc}: links/S1 {pred.height / n_s1:.4f}, empty share {1 - pred['s1_row'].n_unique() / n_s1:.4f}")
        preds.append(pred)
        if C.WRITE_CANDIDATES:
            cs_.append(s)
            cq_.append(q)
    M.load_pred = orig
    pred = pl.concat(preds)
    write_outputs(pred, np.concatenate(cs_) if cs_ else None, np.concatenate(cq_) if cq_ else None)
    validate(check_ids)


def write_outputs(pred: pl.DataFrame, cand_s1=None, cand_q=None):
    """matching_results.tsv always; candidate_pairs.tsv when candidate keys are given (skipped for tier 0)."""
    s1_ids = pl.read_parquet(io.norm_path("test", 1), columns=["entity_id"])["entity_id"]
    q_ids = io.load_norm("test", "q", columns=["entity_id"])["entity_id"]
    raw_order = io.read_source("test", 1)["entity_id"]
    assert raw_order.equals(s1_ids), "normalized S1 order differs from test_source1.tsv"
    n = raw_order.len()
    del raw_order
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pl.DataFrame({"s1_id": s1_ids.gather(pred["s1_row"]), "cand_id": q_ids.gather(pred["q_row"])})
    m = io.write_id_lists(C.OUTPUT_DIR / "matching_results.tsv", MATCH_HEADER, s1_ids, pairs)
    if n != N_TEST_S1:
        print(f"[write] WARNING: test S1 has {n:,} rows, not the official {N_TEST_S1:,} (smoke/mini data?)")
    assert m.height == n and m["source1_entity_id"].n_unique() == n
    print(f"[write] matching_results.tsv {m.height:,} rows, {pred.height:,} links")
    if cand_s1 is not None:
        k = write_candidates(C.OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, q_ids, cand_s1, cand_q)
        print(f"[write] candidate_pairs.tsv {n:,} rows, {k:,} pairs")


def write_candidates(path, s1_ids: pl.Series, q_ids: pl.Series, s1_row, q_row, chunk=50_000) -> int:
    """One row per S1 in file order; ids of each list sorted as strings (as write_id_lists does)."""
    t0 = time.time()
    order = q_ids.arg_sort().to_numpy().astype(np.int64)  # string order of the query ids
    qrank = np.empty(len(order), np.int64)
    qrank[order] = np.arange(len(order))
    key = (np.asarray(s1_row, np.int64) << 32) | qrank[q_row]
    del qrank
    key.sort()
    assert (np.diff(key) > 0).all(), "duplicate candidate pairs"
    s_sorted = (key >> 32).astype(np.int32)
    q_sorted = order[key & 0xFFFFFFFF].astype(np.int32)
    del key, order
    n = s1_ids.len()
    bounds = np.searchsorted(s_sorted, np.arange(0, n + chunk, chunk))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(CAND_HEADER) + "\n")
        for i, a in enumerate(range(0, n, chunk)):
            b = min(a + chunk, n)
            lo, hi = bounds[i], bounds[i + 1]
            agg = (pl.DataFrame({"s": s_sorted[lo:hi], "id": q_ids.gather(q_sorted[lo:hi])})
                   .group_by("s", maintain_order=True).agg(pl.col("id").str.join(",")))
            block = (pl.DataFrame({"s": np.arange(a, b, dtype=np.int32), "s1": s1_ids.slice(a, b - a)})
                     .join(agg, on="s", how="left").sort("s"))
            block.select("s1", pl.col("id").fill_null("")).write_csv(fh, separator="\t", include_header=False, quote_style="never")
    print(f"[write] candidate file in {time.time() - t0:.0f}s")
    return len(s_sorted)


def check_candidates(cand_file, match_file, s1_ids) -> None:
    """The validator's candidate_pairs.tsv rules, streamed: header, one row per S1 in order, S2-/S3- ids only,
    no duplicates within a list, and every matched id present among that S1's candidates."""
    t0 = time.time()
    n_ids = 0
    with open(cand_file, encoding="utf-8") as fc, open(match_file, encoding="utf-8") as fm:
        assert fc.readline() == "\t".join(CAND_HEADER) + "\n", "candidate header"
        assert fm.readline() == "\t".join(MATCH_HEADER) + "\n", "matching header"
        i = -1
        for i, (lc, lm) in enumerate(zip(fc, fm)):
            s1, tab, rest = lc.rstrip("\n").partition("\t")
            s1m, tabm, restm = lm.rstrip("\n").partition("\t")
            assert tab and tabm and s1 == s1m == s1_ids[i], f"row {i + 2}: {s1!r} / {s1m!r} / {s1_ids[i]!r}"
            ids = rest.split(",") if rest else []
            st = set(ids)
            assert len(st) == len(ids), f"duplicate candidate id for {s1}"
            assert all(x.startswith(("S2-", "S3-")) for x in ids), f"bad candidate id prefix for {s1}"
            if restm:
                assert set(restm.split(",")) <= st, f"{s1}: match outside its candidates"
            n_ids += len(ids)
        assert fc.readline() == "" and fm.readline() == "", "extra rows"
        assert i + 1 == len(s1_ids), f"{i + 1} rows for {len(s1_ids)} S1"
    print(f"[check] candidate_pairs.tsv OK: {len(s1_ids):,} rows, {n_ids:,} ids, matches within candidates ({time.time() - t0:.0f}s)")


def rescore_links(pred: pl.DataFrame, n=1000):
    from rapidfuzz import fuzz

    samp = pred.sample(min(n, pred.height), seed=C.SEED)
    a = io.read_source("test", 1)[samp["s1_row"].to_numpy()]
    b = io.read_queries_raw("test")[samp["q_row"].to_numpy()]
    ts = [fuzz.token_set_ratio(x.lower(), y.lower()) for x, y in zip(a["business_name"], b["business_name"])]
    print(f"[rescore] {len(ts)} random links: median raw-name token_set {np.median(ts):.1f} (want >= 80)")
    for i in range(8):
        print(f"   {a['business_name'][i]!r} | {a['business_address'][i]!r}  <->  {b['business_name'][i]!r} | {b['business_address'][i]!r}")


def validate(check_ids=False):
    here = Path(__file__).resolve().parent.parent
    match_file, cand_file = C.OUTPUT_DIR / "matching_results.tsv", C.OUTPUT_DIR / "candidate_pairs.tsv"
    cmd = [sys.executable, str(here / "validate_submission.py"), "--matching", str(match_file),
           "--candidate", str(C.OUTPUT_DIR / "_candidates_checked_separately"), "--test-dir", str(C.DATA_DIR / "test")]
    if check_ids:
        cmd.append("--check-ids")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, "validator FAILED"
    if cand_file.exists():
        s1_ids = io.read_source("test", 1)["entity_id"].to_list()
        check_candidates(cand_file, match_file, s1_ids)


def package(team: str, doc="docs/Documentation_template.md"):
    root = Path(__file__).resolve().parents[2]  # code/business_entity_resolution
    out = Path(f"{team}_submission.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(C.OUTPUT_DIR / f, f"output/{f}")
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                z.write(p, f"code/business_entity_resolution/{p.relative_to(root)}")
        z.write(doc, "Documentation_template.md")
    print(f"[package] {out} {out.stat().st_size / 2**20:.1f} MB")
    with zipfile.ZipFile(out) as z:
        for n in z.namelist()[:40]:
            print("  ", n)
