"""Metrics, including the ones used in previous Amazon ML Challenges.

Swap in the official metric as soon as the problem statement is out and
ALWAYS reproduce it exactly (read the definition, check edge cases like zeros).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent: 0 is perfect, 200 is worst. (AMLC 2025, price prediction.)

    Rows where both values are 0 count as 0 error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    err = np.divide(np.abs(y_pred - y_true), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(100 * err.mean())


def mape_score(y_true, y_pred) -> float:
    """AMLC 2023 style: score = max(0, 100 * (1 - MAPE)). Higher is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    return float(max(0.0, 100 * (1 - mape)))


def entity_f1(y_true, y_pred) -> float:
    """AMLC 2024 style F1 for string extraction, where "" means 'no prediction'.

    TP: pred != "" and gt != "" and pred == gt
    FP: pred != "" and (gt == "" or pred != gt)
    FN: pred == "" and gt != ""
    """
    tp = fp = fn = 0
    for gt, out in zip(y_true, y_pred, strict=True):
        gt = "" if gt is None or (isinstance(gt, float) and np.isnan(gt)) else str(gt).strip()
        out = "" if out is None or (isinstance(out, float) and np.isnan(out)) else str(out).strip()
        if out and gt and out == gt:
            tp += 1
        elif out:
            fp += 1
        elif gt:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rmsle(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def micro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="micro"))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


# name -> (function, greater_is_better)
METRICS = {
    "smape": (smape, False),
    "mape_score": (mape_score, True),
    "entity_f1": (entity_f1, True),
    "rmse": (rmse, False),
    "rmsle": (rmsle, False),
    "mae": (mae, False),
    "macro_f1": (macro_f1, True),
    "micro_f1": (micro_f1, True),
    "accuracy": (accuracy, True),
    "auc": (auc, True),
}


def get_metric(name: str):
    return METRICS[name]
