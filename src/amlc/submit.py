"""Write + validate submission files against the official sample.

    write_submission(pred_df, "lgb_tfidf_v1", sample=RAW / "sample_test_out.csv", id_col="sample_id")

The #1 way to lose a leaderboard slot is a malformed CSV (wrong ids, order, header,
NaNs, dtype). This refuses to write anything that doesn't match the sample exactly.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import SUBMISSIONS


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Raise on any mismatch; return `sub` reordered to the sample's row/column order."""
    errors = []
    if list(sub.columns) != list(sample.columns):
        errors.append(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
    if len(sub) != len(sample):
        errors.append(f"rows {len(sub)} != sample {len(sample)}")
    if sub[id_col].duplicated().any():
        errors.append(f"{sub[id_col].duplicated().sum()} duplicated ids")
    missing = set(sample[id_col]) - set(sub[id_col])
    extra = set(sub[id_col]) - set(sample[id_col])
    if missing:
        errors.append(f"{len(missing)} ids missing, e.g. {list(missing)[:5]}")
    if extra:
        errors.append(f"{len(extra)} unexpected ids, e.g. {list(extra)[:5]}")
    pred_cols = [c for c in sub.columns if c != id_col]
    nans = sub[pred_cols].isna().sum()
    if nans.any():
        errors.append(f"NaNs in predictions: {nans[nans > 0].to_dict()}")
    if errors:
        raise ValueError("Submission invalid:\n  - " + "\n  - ".join(errors))
    return sample[[id_col]].merge(sub, on=id_col, how="left")[list(sample.columns)]


def write_submission(sub: pd.DataFrame, name: str, sample, id_col: str, **to_csv_kw) -> Path:
    sample = pd.read_csv(sample) if not isinstance(sample, pd.DataFrame) else sample
    sub = check_submission(sub, sample, id_col)
    path = SUBMISSIONS / f"{datetime.now():%m%d_%H%M}_{name}.csv"
    sub.to_csv(path, index=False, **to_csv_kw)
    print(f"wrote {path} ({len(sub)} rows)")
    print(sub.head())
    return path
