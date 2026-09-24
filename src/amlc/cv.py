"""Fold creation + a generic out-of-fold GBDT trainer (LightGBM / XGBoost / CatBoost).

Typical use:
    folds = make_folds(train, target="price", task="regression")
    res = train_gbdt_cv(X, y, X_test, folds, model="lgb", task="regression",
                        metric="smape", target_transform="log1p")
    res["oof"], res["test"], res["score"]
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from .config import N_FOLDS, SEED
from .metrics import get_metric


def make_folds(
    df: pd.DataFrame,
    target: str | None = None,
    task: str = "regression",
    group: str | None = None,
    n_splits: int = N_FOLDS,
    n_bins: int = 10,
    seed: int = SEED,
) -> np.ndarray:
    """Return an int array of fold ids (0..n_splits-1), one per row.

    - group given           -> (Stratified)GroupKFold, so near-duplicates never straddle folds
    - classification target -> StratifiedKFold
    - regression target     -> StratifiedKFold on quantile bins of the target
    - no target             -> KFold
    """
    folds = np.full(len(df), -1, dtype=int)
    y = None
    if target is not None:
        y = df[target].to_numpy()
        if task == "regression":
            y = pd.qcut(pd.Series(y).rank(method="first"), q=n_bins, labels=False).to_numpy()

    if group is not None:
        groups = df[group].to_numpy()
        splitter = (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if y is not None
            else GroupKFold(n_splits=n_splits)
        )
        splits = splitter.split(df, y, groups)
    elif y is not None:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df, y)
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df)

    for k, (_, va_idx) in enumerate(splits):
        folds[va_idx] = k
    return folds


_TRANSFORMS = {
    None: (lambda y: y, lambda y: y),
    "log1p": (np.log1p, np.expm1),
    "sqrt": (np.sqrt, np.square),
}


def _default_params(model: str, task: str, n_classes: int, use_gpu: bool) -> dict:
    if model == "lgb":
        obj = {"regression": "regression", "binary": "binary", "multiclass": "multiclass"}[task]
        p = dict(objective=obj, n_estimators=5000, learning_rate=0.05, num_leaves=63,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
        if task == "multiclass":
            p["num_class"] = n_classes
        return p
    if model == "xgb":
        obj = {"regression": "reg:squarederror", "binary": "binary:logistic",
               "multiclass": "multi:softprob"}[task]
        return dict(objective=obj, n_estimators=5000, learning_rate=0.05, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    device="cuda" if use_gpu else "cpu", n_jobs=-1)
    if model == "cat":
        loss = {"regression": "RMSE", "binary": "Logloss", "multiclass": "MultiClass"}[task]
        return dict(loss_function=loss, iterations=5000, learning_rate=0.05, depth=8,
                    task_type="GPU" if use_gpu else "CPU", verbose=0)
    raise ValueError(f"unknown model {model!r}")


def _fit_predict(model, task, params, X_tr, y_tr, X_va, y_va, X_te, es, cat_features):
    if model == "lgb":
        import lightgbm as lgb

        est = (lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier)(**params)
        est.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
                callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)],
                categorical_feature=cat_features if cat_features else "auto")
    elif model == "xgb":
        import xgboost as xgb

        est = (xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier)(
            **params, early_stopping_rounds=es)
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        import catboost as cb

        est = (cb.CatBoostRegressor if task == "regression" else cb.CatBoostClassifier)(**params)
        est.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=es,
                cat_features=cat_features)

    if task == "regression":
        return est, est.predict(X_va), est.predict(X_te)
    if task == "binary":
        return est, est.predict_proba(X_va)[:, 1], est.predict_proba(X_te)[:, 1]
    return est, est.predict_proba(X_va), est.predict_proba(X_te)


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def train_gbdt_cv(
    X,
    y,
    X_test,
    folds: np.ndarray,
    model: str = "lgb",
    task: str = "regression",
    metric: str = "rmse",
    params: dict | None = None,
    target_transform: str | None = None,
    early_stopping: int = 200,
    cat_features: list | None = None,
    use_gpu: bool = False,
) -> dict:
    """K-fold training. Returns dict(oof, test, score, fold_scores, models).

    For binary/multiclass tasks oof/test are probabilities; the metric is computed on
    argmax / 0.5-threshold labels unless the metric is 'auc'.
    """
    y = np.asarray(y)
    fwd, inv = _TRANSFORMS[target_transform]
    n_classes = int(len(np.unique(y))) if task == "multiclass" else 1
    p = {**_default_params(model, task, n_classes, use_gpu), **(params or {})}
    if model == "lgb":
        p.setdefault("random_state", SEED)
    elif model == "xgb":
        p.setdefault("random_state", SEED)
    else:
        p.setdefault("random_seed", SEED)
    metric_fn, _ = get_metric(metric)

    n_folds = int(folds.max()) + 1
    oof = np.zeros((len(y), n_classes)) if task == "multiclass" else np.zeros(len(y))
    test = np.zeros((len(X_test), n_classes)) if task == "multiclass" else np.zeros(len(X_test))
    models, fold_scores = [], []

    def _score(y_true, pred):
        if task == "regression" or metric == "auc":
            return metric_fn(y_true, pred)
        labels = pred.argmax(1) if task == "multiclass" else (pred > 0.5).astype(int)
        return metric_fn(y_true, labels)

    for k in range(n_folds):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        y_tr = fwd(y[tr]) if task == "regression" else y[tr]
        y_va = fwd(y[va]) if task == "regression" else y[va]
        est, p_va, p_te = _fit_predict(model, task, p, _take(X, tr), y_tr, _take(X, va), y_va,
                                       X_test, early_stopping, cat_features)
        if task == "regression":
            p_va, p_te = inv(p_va), inv(p_te)
        oof[va] = p_va
        test += p_te / n_folds
        s = _score(y[va], p_va)
        fold_scores.append(s)
        models.append(est)
        print(f"  fold {k}: {metric}={s:.5f}")

    score = _score(y, oof)
    print(f"{model} CV {metric}: {score:.5f} (folds {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test, score=score, fold_scores=fold_scores, models=models)
