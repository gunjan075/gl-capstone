"""Regression metrics and evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from . import config


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }


def evaluate_train_test(pipe, X_train, y_train, X_test, y_test) -> dict:
    """Compute train and test metrics plus the train/test RMSE gap."""
    train = regression_metrics(y_train, pipe.predict(X_train))
    test = regression_metrics(y_test, pipe.predict(X_test))
    row = {f"train_{k}": v for k, v in train.items()}
    row.update({f"test_{k}": v for k, v in test.items()})
    row["train_test_RMSE_gap"] = test["RMSE"] - train["RMSE"]
    return row


def cross_val_rmse(pipe, X, y, cv: int = config.CV_FOLDS, n_jobs: int = -1) -> tuple[float, float]:
    """Mean and std of cross-validated RMSE (lower is better).

    Use ``n_jobs=1`` for GPU-trained models so parallel folds don't contend for
    the single GPU.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(
        pipe, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=n_jobs
    )
    return float(-scores.mean()), float(scores.std())
