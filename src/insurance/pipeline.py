"""Assemble the full estimator pipeline: features -> preprocess -> model."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .features import FeatureEngineer
from .preprocess import build_preprocessor


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Wrap an estimator in the shared feature + preprocessing pipeline.

    The returned pipeline accepts the *cleaned raw* DataFrame (output of
    :func:`insurance.data.clean_schema`) so that all fitted transformations are
    applied identically at training and inference time.
    """
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
