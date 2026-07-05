"""Feature engineering as a scikit-learn compatible transformer.

All derived features and the data-driven imputations (BMI group medians,
admission-recency reference year) are learned on the training fold only, then
applied identically at predict time. This is the single source of truth for
feature creation - the EDA notebook, training, API, and Streamlit app all use it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


def _bmi_category(bmi: pd.Series) -> pd.Series:
    bins = [-np.inf, 18.5, 25, 30, np.inf]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
    return pd.cut(bmi, bins=bins, labels=labels).astype(object)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create engineered features and apply learned imputations.

    Learned on ``fit`` (training data only):
      * ``bmi`` median by (``age_band``, ``Gender``) with a global-median fallback.
      * the reference year used for ``years_since_last_admitted``.

    Created features: ``age_band``, ``bmi_category``, ``cholesterol_midpoint``,
    ``was_admitted_before``, ``years_since_last_admitted``,
    ``any_major_disease_history``, ``weight_bmi_interaction``, ``steps_per_age``.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        age_band = pd.cut(X["age"], bins=config.AGE_BANDS, labels=config.AGE_BAND_LABELS)
        tmp = X.assign(_age_band=age_band)
        self.bmi_group_median_ = (
            tmp.groupby(["_age_band", "Gender"], observed=True)["bmi"].median()
        )
        self.bmi_global_median_ = float(X["bmi"].median())

        if "Year_last_admitted" in X.columns:
            max_year = X["Year_last_admitted"].max()
            self.reference_year_ = int(max_year) if pd.notna(max_year) else 2024
        else:
            self.reference_year_ = 2024
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if config.ID_COL in X.columns:
            X = X.drop(columns=[config.ID_COL])

        # Age band (used for grouped BMI imputation and as a categorical feature)
        X["age_band"] = pd.cut(
            X["age"], bins=config.AGE_BANDS, labels=config.AGE_BAND_LABELS
        ).astype(object)

        # BMI: grouped-median imputation then plausibility capping (vectorized)
        group_fill = pd.Series(
            [self.bmi_group_median_.get(k, np.nan) for k in zip(X["age_band"], X["Gender"])],
            index=X.index,
        )
        X["bmi"] = (
            X["bmi"].fillna(group_fill).fillna(self.bmi_global_median_).astype(float)
        )
        X["bmi"] = X["bmi"].clip(lower=config.BMI_MIN, upper=config.BMI_MAX)
        X["bmi_category"] = _bmi_category(X["bmi"])

        # Cholesterol range -> numeric midpoint
        X["cholesterol_midpoint"] = (
            X["cholesterol_level"].map(config.CHOLESTEROL_MIDPOINTS).astype(float)
        )

        # Admission history / recency from Year_last_admitted
        if "Year_last_admitted" in X.columns:
            year = pd.to_numeric(X["Year_last_admitted"], errors="coerce")
            X["was_admitted_before"] = year.notna().astype(int)
            X["years_since_last_admitted"] = (
                (self.reference_year_ - year).clip(lower=0).fillna(0).astype(float)
            )
        else:
            X["was_admitted_before"] = 0
            X["years_since_last_admitted"] = 0.0

        # Combined disease history
        X["any_major_disease_history"] = (
            (X["heart_disease_history"] == 1) | (X["other_major_disease_history"] == 1)
        ).astype(int)

        # Interactions / ratios
        X["weight_bmi_interaction"] = X["weight"] * X["bmi"]
        X["steps_per_age"] = X["daily_avg_steps"] / X["age"].replace(0, np.nan)
        X["steps_per_age"] = X["steps_per_age"].fillna(X["daily_avg_steps"])

        # Drop columns superseded by engineered versions
        X = X.drop(columns=["cholesterol_level"], errors="ignore")
        X = X.drop(columns=["Year_last_admitted"], errors="ignore")

        return X[config.NUMERIC_FEATURES + config.NOMINAL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(config.NUMERIC_FEATURES + config.NOMINAL_FEATURES)
