"""Small helpers used everywhere: seeding, timing, memory, experiment log."""
import csv
import gc
import os
import pickle
import random
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

from .config import EXPERIMENT_LOG, SEED


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.perf_counter() - t0:.1f}s")


def reduce_mem(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns in place. RAM on the laptop is tight (13 GB)."""
    before = df.memory_usage(deep=True).sum() / 2**20
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["floating"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    if verbose:
        after = df.memory_usage(deep=True).sum() / 2**20
        print(f"memory {before:.1f} MB -> {after:.1f} MB")
    return df


def save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def log_experiment(name: str, cv_score: float, notes: str = "", lb_score: float | None = None, **extra) -> None:
    """Append one row to experiments.csv - the source of truth for the approach doc."""
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": name,
        "cv": round(float(cv_score), 5),
        "lb": "" if lb_score is None else lb_score,
        "notes": notes,
        **extra,
    }
    new = not EXPERIMENT_LOG.exists()
    with open(EXPERIMENT_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"logged {name}: cv={row['cv']}")
