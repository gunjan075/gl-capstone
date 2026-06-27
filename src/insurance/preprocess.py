"""Preprocessing ColumnTransformer (impute + scale + encode).

Operates on the output of :class:`insurance.features.FeatureEngineer`. Scaling
is applied to all numeric columns: it is required by the linear/regularized
models and harmless for the tree/boosting models, so a single shared
preprocessor keeps the comparison fair and the deployment artifact simple.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    nominal_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("nom", nominal_pipe, config.NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Keep feature names through the pipeline (consistent fit/predict, friendlier SHAP)
    preprocessor.set_output(transform="pandas")
    return preprocessor
