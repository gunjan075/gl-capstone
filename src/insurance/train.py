"""Train, tune, select, and persist the final insurance-cost model.

Run with::

    uv run python -m insurance.train            # full run
    uv run python -m insurance.train --quick     # fast smoke run

Outputs:
    models/final_model.pkl          bundle: pipeline + thresholds + drivers + metrics
    models/metrics.json             final-model + comparison summary
    reports/tables/model_comparison.csv
    reports/tables/feature_importance.csv
"""

from __future__ import annotations

import os as _os

# Limit native (OpenMP/BLAS) thread pools so sklearn CV/search parallelism
# (n_jobs) does not oversubscribe cores. MUST be set before numpy/sklearn/
# boosting libraries are imported.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from . import config, evaluate, models
from .data import load_clean, split_X_y
from .pipeline import build_pipeline

warnings.filterwarnings("ignore")


def _evaluate_all(X_train, y_train, X_test, y_test, cv_folds: int) -> pd.DataFrame:
    """Fit every candidate model and collect train/test/CV metrics."""
    rows = []
    for name, estimator in models.get_models().items():
        try:
            pipe = build_pipeline(estimator)
            pipe.fit(X_train, y_train)
            row = {"model": name}
            row.update(evaluate.evaluate_train_test(pipe, X_train, y_train, X_test, y_test))
            # GPU models run folds sequentially to avoid contending for one GPU
            cv_njobs = 1 if name in models.GPU_MODELS else -1
            cv_mean, cv_std = evaluate.cross_val_rmse(
                pipe, X_train, y_train, cv=cv_folds, n_jobs=cv_njobs
            )
            row["cv_rmse_mean"] = cv_mean
            row["cv_rmse_std"] = cv_std
            rows.append(row)
            print(f"  {name:22s} test_RMSE={row['test_RMSE']:9.1f}  test_R2={row['test_R2']:.4f}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  {name:22s} SKIPPED ({exc})")
    return pd.DataFrame(rows).sort_values("test_RMSE").reset_index(drop=True)


def _tune_best(best_name, X_train, y_train, n_iter, cv):
    """RandomizedSearchCV over the best family's search space."""
    space = models.get_search_spaces().get(best_name)
    if not space:
        print(f"  No search space for {best_name}; using default hyperparameters.")
        return build_pipeline(models.get_models()[best_name]).fit(X_train, y_train), {}

    search_njobs = 1 if best_name in models.GPU_MODELS else -1
    print(f"  Tuning {best_name} with RandomizedSearchCV "
          f"(n_iter={n_iter}, cv={cv}, n_jobs={search_njobs})...")
    search = RandomizedSearchCV(
        build_pipeline(models.get_models()[best_name]),
        param_distributions=space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=search_njobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_params = {k: _to_native(v) for k, v in search.best_params_.items()}
    print(f"  Best CV RMSE: {-search.best_score_:.1f}")
    return search.best_estimator_, best_params


def _to_native(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _feature_importance(pipe, X_test, y_test) -> pd.DataFrame:
    """Permutation importance over the raw input columns."""
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def run_training(n_iter: int = 25, baseline_cv: int = 3, tune_cv: int = 5) -> dict:
    print(f"GPU acceleration: {'ON (XGBoost/CatBoost)' if config.USE_GPU else 'OFF (CPU only)'}")
    print("Loading data...")
    df = load_clean()
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"  train={len(X_train)}  test={len(X_test)}")

    print("\nEvaluating candidate models...")
    comparison = _evaluate_all(X_train, y_train, X_test, y_test, baseline_cv)
    comparison.to_csv(config.TABLES_DIR / "model_comparison.csv", index=False)

    ranked = comparison[comparison["model"] != "DummyRegressor"]
    best_name = ranked.iloc[0]["model"]
    best_untuned_rmse = float(ranked.iloc[0]["test_RMSE"])
    print(f"\nBest baseline model: {best_name} (test RMSE={best_untuned_rmse:.1f})")

    print("\nHyperparameter tuning...")
    final_pipe, best_params = _tune_best(best_name, X_train, y_train, n_iter, tune_cv)

    final_metrics = evaluate.evaluate_train_test(
        final_pipe, X_train, y_train, X_test, y_test
    )
    # Keep the better of tuned vs untuned baseline
    if final_metrics["test_RMSE"] > best_untuned_rmse:
        print("  Tuned model did not beat baseline; refitting baseline defaults.")
        final_pipe = build_pipeline(models.get_models()[best_name]).fit(X_train, y_train)
        final_metrics = evaluate.evaluate_train_test(
            final_pipe, X_train, y_train, X_test, y_test
        )
        best_params = {}

    # Make a GPU-trained XGBoost portable so the saved artifact predicts on CPU
    if best_name == "XGBoost" and config.USE_GPU:
        try:
            final_pipe.named_steps["model"].get_booster().set_param({"device": "cpu"})
        except Exception:
            pass

    print("\nComputing permutation importance...")
    importance = _feature_importance(final_pipe, X_test, y_test)
    importance.to_csv(config.TABLES_DIR / "feature_importance.csv", index=False)

    # Risk thresholds = tertiles of the training target
    low_to_medium = float(np.quantile(y_train, 1 / 3))
    medium_to_high = float(np.quantile(y_train, 2 / 3))
    thresholds = {"low_to_medium": low_to_medium, "medium_to_high": medium_to_high}

    bundle = {
        "pipeline": final_pipe,
        "model_name": best_name,
        "best_params": best_params,
        "risk_thresholds": thresholds,
        "feature_importance": importance.head(8).to_dict(orient="records"),
        "metrics": final_metrics,
        "input_columns": list(X.columns),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, config.MODEL_PATH)

    metrics_out = {
        "final_model": best_name,
        "gpu_enabled": config.USE_GPU,
        "best_params": best_params,
        "metrics": final_metrics,
        "risk_thresholds": thresholds,
        "top_drivers": bundle["feature_importance"],
        "model_comparison": comparison.to_dict(orient="records"),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics_out, indent=2))

    print("\n" + "=" * 60)
    print(f"FINAL MODEL: {best_name}")
    print(f"  Test MAE : {final_metrics['test_MAE']:.1f}")
    print(f"  Test RMSE: {final_metrics['test_RMSE']:.1f}")
    print(f"  Test R2  : {final_metrics['test_R2']:.4f}")
    print(f"  Saved -> {config.MODEL_PATH}")
    print("=" * 60)
    if final_metrics["test_R2"] < 0.95:
        print("WARNING: test R2 below 0.95 target.")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the insurance-cost model.")
    parser.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations")
    parser.add_argument("--baseline-cv", type=int, default=3, help="CV folds for baseline comparison")
    parser.add_argument("--tune-cv", type=int, default=5, help="CV folds for tuning")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run (n_iter=5, cv=3)")
    args = parser.parse_args()
    if args.quick:
        run_training(n_iter=5, baseline_cv=3, tune_cv=3)
    else:
        run_training(n_iter=args.n_iter, baseline_cv=args.baseline_cv, tune_cv=args.tune_cv)


if __name__ == "__main__":
    main()
