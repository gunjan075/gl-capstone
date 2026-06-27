"""Inference utilities: load the trained artifact and score applicants.

The training step persists a *bundle* (dict) rather than a bare estimator so
the serving layers also get the risk thresholds and the global feature-driver
ranking without recomputing anything.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from . import config


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    """Load (and cache) the trained model bundle from disk."""
    return joblib.load(path or config.MODEL_PATH)


def risk_category(cost: float, thresholds: dict | None = None) -> str:
    """Map a predicted cost to a Low / Medium / High band."""
    thresholds = thresholds or config.DEFAULT_RISK_THRESHOLDS
    if cost < thresholds["low_to_medium"]:
        return "Low"
    if cost < thresholds["medium_to_high"]:
        return "Medium"
    return "High"


def _to_frame(applicants) -> pd.DataFrame:
    if isinstance(applicants, pd.DataFrame):
        return applicants
    if isinstance(applicants, dict):
        return pd.DataFrame([applicants])
    return pd.DataFrame(list(applicants))


def predict(applicants, bundle: dict | None = None) -> list[dict]:
    """Predict insurance cost + risk category for one or more applicants.

    ``applicants`` may be a dict, a list of dicts, or a DataFrame using the
    cleaned input column names (see :class:`insurance.schemas.InsuranceApplicant`).
    """
    bundle = bundle or load_bundle()
    pipeline = bundle["pipeline"]
    thresholds = bundle.get("risk_thresholds", config.DEFAULT_RISK_THRESHOLDS)

    frame = _to_frame(applicants)
    preds = pipeline.predict(frame)
    return [
        {
            "predicted_cost": round(float(p), 2),
            "risk_category": risk_category(float(p), thresholds),
        }
        for p in preds
    ]
