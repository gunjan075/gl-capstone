"""Model registry and hyperparameter search spaces.

Gradient-boosting libraries (XGBoost / LightGBM / CatBoost) are imported
defensively: if a library is unavailable in the environment it is skipped
rather than crashing the run, matching the plan's graceful-degradation policy.
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from . import config

RS = config.RANDOM_STATE

# Models that train on the GPU when USE_GPU is set. These must NOT be run under
# parallel CV/search (n_jobs>1) because they would contend for the single GPU;
# see ``insurance.train`` which runs them sequentially.
GPU_MODELS: set[str] = set()

# Optional boosting libraries -------------------------------------------------
HAS_XGB = HAS_LGBM = HAS_CATBOOST = False
try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover - environment dependent
    pass
try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except Exception:  # pragma: no cover
    pass
try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except Exception:  # pragma: no cover
    pass


def get_models() -> dict:
    """Return the full set of candidate estimators (untuned defaults).

    Each model is single-threaded (``n_jobs=1``); parallelism is applied at the
    cross-validation / search level instead, which avoids thread oversubscription.
    Boosting models use the GPU when :data:`insurance.config.USE_GPU` is set.
    """
    GPU_MODELS.clear()
    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RS),
        "Lasso": Lasso(random_state=RS, max_iter=5000),
        "ElasticNet": ElasticNet(random_state=RS, max_iter=5000),
        "DecisionTree": DecisionTreeRegressor(random_state=RS),
        "RandomForest": RandomForestRegressor(random_state=RS, n_jobs=1),
        "ExtraTrees": ExtraTreesRegressor(random_state=RS, n_jobs=1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RS),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RS),
    }
    if HAS_XGB:
        xgb_kwargs = dict(random_state=RS, tree_method="hist", verbosity=0, n_jobs=1)
        if config.USE_GPU:
            xgb_kwargs["device"] = "cuda"
            GPU_MODELS.add("XGBoost")
        models["XGBoost"] = XGBRegressor(**xgb_kwargs)
    if HAS_LGBM:
        # LightGBM GPU build needs OpenCL (unavailable here) -> stays on CPU.
        models["LightGBM"] = LGBMRegressor(random_state=RS, n_jobs=1, verbose=-1)
    if HAS_CATBOOST:
        cat_kwargs = dict(random_state=RS, verbose=0, allow_writing_files=False, thread_count=1)
        if config.USE_GPU:
            cat_kwargs.update(task_type="GPU", devices="0")
            cat_kwargs.pop("thread_count")
            GPU_MODELS.add("CatBoost")
        models["CatBoost"] = CatBoostRegressor(**cat_kwargs)
    return models


def get_search_spaces() -> dict:
    """RandomizedSearchCV parameter distributions, keyed by model name.

    Keys are prefixed with ``model__`` to address the estimator step inside the
    full pipeline returned by :func:`insurance.pipeline.build_pipeline`.
    """
    spaces = {
        "Ridge": {"model__alpha": loguniform(1e-2, 1e2)},
        "Lasso": {"model__alpha": loguniform(1e-2, 1e2)},
        "ElasticNet": {
            "model__alpha": loguniform(1e-2, 1e2),
            "model__l1_ratio": uniform(0.05, 0.9),
        },
        "DecisionTree": {
            "model__max_depth": randint(3, 25),
            "model__min_samples_leaf": randint(5, 80),
            "model__max_features": uniform(0.4, 0.6),
        },
        "RandomForest": {
            "model__n_estimators": randint(200, 600),
            "model__max_depth": randint(5, 30),
            "model__min_samples_leaf": randint(2, 40),
            "model__max_features": uniform(0.3, 0.7),
        },
        "ExtraTrees": {
            "model__n_estimators": randint(200, 600),
            "model__max_depth": randint(5, 30),
            "model__min_samples_leaf": randint(2, 40),
            "model__max_features": uniform(0.3, 0.7),
        },
        "GradientBoosting": {
            "model__n_estimators": randint(150, 500),
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__max_depth": randint(2, 5),
            "model__subsample": uniform(0.6, 0.4),
        },
        "HistGradientBoosting": {
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__max_iter": randint(150, 600),
            "model__max_leaf_nodes": randint(8, 64),
            "model__min_samples_leaf": randint(15, 80),
            "model__max_bins": randint(64, 255),
            "model__l2_regularization": loguniform(1e-3, 1.0),
        },
    }
    if HAS_XGB:
        spaces["XGBoost"] = {
            "model__n_estimators": randint(200, 700),
            "model__max_depth": randint(3, 9),
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__reg_alpha": loguniform(1e-3, 1.0),
            "model__reg_lambda": loguniform(1e-3, 10.0),
        }
    if HAS_LGBM:
        spaces["LightGBM"] = {
            "model__n_estimators": randint(200, 800),
            "model__num_leaves": randint(15, 80),
            "model__max_depth": randint(3, 12),
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_samples": randint(10, 80),
            "model__reg_alpha": loguniform(1e-3, 1.0),
            "model__reg_lambda": loguniform(1e-3, 10.0),
        }
    if HAS_CATBOOST:
        spaces["CatBoost"] = {
            "model__iterations": randint(200, 500),
            "model__depth": randint(4, 10),
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__l2_leaf_reg": loguniform(1.0, 10.0),
        }
    return spaces
