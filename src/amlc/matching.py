"""Pair classifier + the decision rule that turns probabilities into match lists.

Decision rule (uses the dataset's structure): each S2/S3 record belongs to at most one S1
entity, so a query keeps only its highest-probability S1, and only if that probability
clears `threshold`. The threshold is tuned on out-of-fold predictions for macro F_0.5.
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from .blocking import to_id_pairs
from .config import N_FOLDS, SEED
from .features import FEATURES
from .metrics import fbeta_macro

LGB_PARAMS = dict(objective="binary", n_estimators=2000, learning_rate=0.05, num_leaves=127,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  verbose=-1, n_jobs=-1, random_state=SEED)


def label_pairs(pairs: pl.DataFrame, s1n: pl.DataFrame, qn: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    ids = to_id_pairs(pairs, s1n, qn)
    y = ids.join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_id", "cand_id"], how="left")["y"].fill_null(0)
    return pairs.with_columns(y=y)


def train_oof(df: pl.DataFrame, n_folds: int = N_FOLDS, params: dict | None = None, features=FEATURES):
    """GroupKFold by S1 entity (all candidates of one S1 stay in one fold).
    Returns (oof_proba, models)."""
    X = df.select(features).to_numpy()
    y = df["y"].to_numpy()
    groups = df["si"].to_numpy()
    p = {**LGB_PARAMS, **(params or {})}
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        m = lgb.LGBMClassifier(**p)
        m.fit(X[tr], y[tr], eval_X=X[va], eval_y=y[va],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        models.append(m)
        print(f"  fold {k}: best_iter={m.best_iteration_} logloss={m.best_score_['valid_0']['binary_logloss']:.4f}")
    return oof, models


def predict(models, df: pl.DataFrame, features=FEATURES) -> np.ndarray:
    X = df.select(features).to_numpy()
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0).astype(np.float32)


def decide(pairs: pl.DataFrame, proba_col: str = "proba", threshold: float = 0.5) -> pl.DataFrame:
    """Keep, for each query, its single best S1 if proba >= threshold."""
    best = pairs.filter(pl.col(proba_col) == pl.col(proba_col).max().over("qi"))
    return best.unique("qi", keep="first").filter(pl.col(proba_col) >= threshold)


def tune_threshold(pairs: pl.DataFrame, s1n, qn, truth, s1_ids, proba_col: str = "proba",
                   grid=np.arange(0.20, 0.96, 0.05)) -> tuple[float, float]:
    best = (0.5, -1.0)
    for t in grid:
        f = fbeta_macro(to_id_pairs(decide(pairs, proba_col, t), s1n, qn), truth, s1_ids)
        print(f"  threshold {t:.2f}: F0.5={f:.5f}")
        if f > best[1]:
            best = (float(t), f)
    print(f"best threshold {best[0]:.2f} -> F0.5 {best[1]:.5f}")
    return best
