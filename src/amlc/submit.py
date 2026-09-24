"""Write the two official output files and run the official validator on them.

    run_dir = write_outputs("tfidf_lgb_v1", test_s1_ids, match_pairs, cand_pairs)
    -> submissions/<MMDD_HHMM>_tfidf_lgb_v1/matching_results.tsv   (upload this to the portal)
       submissions/<MMDD_HHMM>_tfidf_lgb_v1/candidate_pairs.tsv    (goes in the final zip)

Format rules enforced here: TAB separated, exact headers, one row per test S1 entity (empty
list for singletons), comma-joined ids without quoting, no duplicate ids, only S2-/S3- ids,
matches a subset of candidates.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from .config import CAND_COL, GT_MATCH_COL, GT_S1_COL, ROOT, SUBMISSIONS, RAW

VALIDATOR = ROOT / "scripts" / "validate_submission.py"


def _to_lists(pairs: pl.DataFrame, s1_ids, list_col: str) -> pl.DataFrame:
    grouped = (
        pairs.filter(pl.col("cand_id").str.contains(r"^S[23]-"))
        .unique(["s1_id", "cand_id"])
        .sort("s1_id", "cand_id")
        .group_by("s1_id", maintain_order=True)
        .agg(pl.col("cand_id").str.join(","))
    )
    base = pl.DataFrame({GT_S1_COL: pl.Series(s1_ids, dtype=pl.Utf8)}).unique(maintain_order=True)
    return (
        base.join(grouped.rename({"s1_id": GT_S1_COL, "cand_id": list_col}), on=GT_S1_COL, how="left")
        .with_columns(pl.col(list_col).fill_null(""))
    )


def write_outputs(name: str, s1_ids, matches: pl.DataFrame, candidates: pl.DataFrame,
                  validate: bool = True, test_dir: Path = RAW / "test") -> Path:
    """matches / candidates: pair frames (s1_id, cand_id). Matches are forced into the candidate set."""
    run_dir = SUBMISSIONS / f"{datetime.now():%m%d_%H%M}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = pl.concat([candidates.select("s1_id", "cand_id"), matches.select("s1_id", "cand_id")])
    for fname, pairs, col in [("matching_results.tsv", matches, GT_MATCH_COL),
                              ("candidate_pairs.tsv", candidates, CAND_COL)]:
        _to_lists(pairs, s1_ids, col).write_csv(run_dir / fname, separator="\t", quote_style="never")
    n = matches.height
    print(f"wrote {run_dir}  ({len(set(s1_ids)):,} S1 rows, {n:,} matched links, {candidates.height:,} candidate links)")
    if validate:
        run_validator(run_dir, test_dir)
    return run_dir


def run_validator(run_dir: Path, test_dir: Path = RAW / "test", check_ids: bool = False) -> bool:
    cmd = [sys.executable, str(VALIDATOR), "--matching", str(run_dir / "matching_results.tsv"),
           "--candidate", str(run_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir)]
    if check_ids:
        cmd.append("--check-ids")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-3000:], res.stderr[-2000:])
    return res.returncode == 0
