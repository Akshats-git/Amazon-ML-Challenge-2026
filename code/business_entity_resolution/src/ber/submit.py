"""Section 7.10 / Step 13-14: test inference, output files, asserts, label-free monitors, validator, packaging."""
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C
from . import io
from .blocking import cand_path
from .decide import decide, load_thresholds
from .model import _model_path, _oof, build_ctx2, load_booster, predict_stage
from .pools import countries

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")
N_TEST_S1 = 1_732_544


def predict_and_write(tier0=False, tag="main", check_ids=False):
    has_s2 = all(_model_path(tag, 2, p).exists() for p in ("P0", "P1")) and not tier0
    pools1 = ("P0",) if tier0 else ("P0", "P1")
    m1 = [load_booster(tag, 1, p) for p in pools1]
    m2 = [load_booster(tag, 2, p) for p in ("P0", "P1")] if has_s2 else []
    T = (0.76, 0.76) if tier0 else load_thresholds()
    kind = "p2" if has_s2 else "p1"
    print(f"[predict] stage-1 models {pools1}, stage-2 {'yes' if has_s2 else 'no'}, thresholds {T}")
    preds, cands, mon = [], [], []
    for c in countries("test"):
        predict_stage(m1, "test", c, 1, tag, "p1")
        if has_s2:
            build_ctx2("test", c, tag)
            predict_stage(m2, "test", c, 2, tag, "p2")
        df = pl.read_parquet(_oof(tag, kind, "test", c)).select("q_row", "s1_row", pl.col(kind).alias("p"))
        blk = pl.read_parquet(cand_path("test", c), columns=["q_row", "s1_row", "blk_score", "blk_rank"])
        assert df.height == blk.height, f"{c}: scored pairs {df.height:,} != candidate pairs {blk.height:,}"
        pred = decide(df, *T)
        preds.append(pred)
        cands.append(blk.select("s1_row", "q_row"))
        mon.append((c, pred, blk, df))
    pred, cand = pl.concat(preds), pl.concat(cands)
    stray = pred.join(cand, on=["s1_row", "q_row"], how="anti")
    assert stray.height == 0, f"{stray.height} matched pairs are not in the candidate set"
    write_outputs(pred, cand)
    monitors(mon)
    validate(check_ids)


def write_outputs(pred: pl.DataFrame, cand: pl.DataFrame):
    s1_ids = pl.read_parquet(io.norm_path("test", 1), columns=["entity_id"])["entity_id"]
    q_ids = io.load_norm("test", "q", columns=["entity_id"])["entity_id"]
    raw_order = io.read_source("test", 1)["entity_id"]
    assert raw_order.equals(s1_ids), "normalized S1 order differs from test_source1.tsv"

    def to_ids(pairs):
        return pl.DataFrame({"s1_id": s1_ids.gather(pairs["s1_row"]), "cand_id": q_ids.gather(pairs["q_row"])})

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m = io.write_id_lists(C.OUTPUT_DIR / "matching_results.tsv", MATCH_HEADER, s1_ids, to_ids(pred))
    k = io.write_id_lists(C.OUTPUT_DIR / "candidate_pairs.tsv", CAND_HEADER, s1_ids, to_ids(cand))
    n = raw_order.len()
    if n != N_TEST_S1:
        print(f"[write] WARNING: test S1 has {n:,} rows, not the official {N_TEST_S1:,} (smoke/mini data?)")
    assert m.height == n and k.height == n
    assert m["source1_entity_id"].n_unique() == n
    print(f"[write] matching_results.tsv {m.height:,} rows, {pred.height:,} links; candidate_pairs.tsv {cand.height:,} pairs")


def monitors(mon):
    """Label-free sanity monitors per country (Step 13)."""
    s1c = pl.read_parquet(io.norm_path("test", 1), columns=["country"]).with_row_index("s1_row")
    rows = []
    for c, pred, blk, df in mon:
        n_s1 = int((s1c["country"] == c).sum())
        n_q = blk["q_row"].n_unique()
        linked_s1 = pred["s1_row"].n_unique()
        top1 = blk.filter(pl.col("blk_rank") == 1)["blk_score"].median()
        rows.append(dict(country=c, n_s1=n_s1, links_per_s1=pred.height / n_s1, empty_share=1 - linked_s1 / n_s1,
                         median_top1_blk=top1, share_queries_linked=pred.height / max(n_q, 1)))
    rep = pl.DataFrame(rows)
    with pl.Config(float_precision=4):
        print(rep)
    rescore_links(pl.concat([m[1] for m in mon]))
    return rep


def rescore_links(pred: pl.DataFrame, n=1000):
    from rapidfuzz import fuzz

    s1 = io.read_source("test", 1)
    q = io.read_queries_raw("test")
    samp = pred.sample(min(n, pred.height), seed=C.SEED)
    a = s1[samp["s1_row"].to_numpy()]
    b = q[samp["q_row"].to_numpy()]
    ts = [fuzz.token_set_ratio(x.lower(), y.lower()) for x, y in zip(a["business_name"], b["business_name"])]
    print(f"[rescore] {len(ts)} random links: median raw-name token_set {np.median(ts):.1f} (want >= 80)")
    for i in range(8):
        print(f"   {a['business_name'][i]!r} | {a['business_address'][i]!r}  <->  {b['business_name'][i]!r} | {b['business_address'][i]!r}")


def validate(check_ids=False):
    here = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(here / "validate_submission.py"), "--matching", str(C.OUTPUT_DIR / "matching_results.tsv"),
           "--candidate", str(C.OUTPUT_DIR / "candidate_pairs.tsv"), "--test-dir", str(C.DATA_DIR / "test")]
    if check_ids:
        cmd.append("--check-ids")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, "validator FAILED"


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
