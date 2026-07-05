from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUMN_RENAMES = {
    "regular_checkup_lasy_year": "regular_checkup_last_year",
    "heart_decs_history": "heart_disease_history",
    "other_major_decs_history": "other_major_disease_history",
}

BASE_NUMERIC_FEATURES = [
    "years_of_insurance_with_us",
    "regular_checkup_last_year",
    "adventure_sports",
    "visited_doctor_last_1_year",
    "daily_avg_steps",
    "age",
    "heart_disease_history",
    "other_major_disease_history",
    "avg_glucose_level",
    "bmi",
    "weight",
    "weight_change_in_last_one_year",
    "fat_percentage",
    "cholesterol_midpoint",
    "was_admitted_before",
    "admission_year_missing_flag",
    "bmi_missing_flag",
    "years_since_last_admitted",
    "any_major_disease_history",
    "weight_bmi_interaction",
    "steps_per_age",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + [
    "smoking_risk_score",
    "alcohol_risk_score",
    "exercise_risk_score",
    "medical_risk_score",
    "lifestyle_risk_score",
    "preventive_care_score",
    "obesity_risk_flag",
    "high_glucose_flag",
    "high_cholesterol_flag",
    "sedentary_flag",
    "current_smoker_flag",
    "daily_alcohol_flag",
    "age_bmi_interaction",
    "bmi_glucose_interaction",
    "bmi_exercise_risk_interaction",
    "age_medical_risk_interaction",
    "smoking_alcohol_interaction",
    "steps_exercise_interaction",
    "fat_bmi_interaction",
    "doctor_visit_intensity",
    "admission_recency_risk",
    "cholesterol_age_interaction",
]

BASE_CATEGORICAL_FEATURES = [
    "Occupation",
    "Gender",
    "smoking_status",
    "Location",
    "covered_by_any_other_company",
    "Alcohol",
    "exercise",
    "age_band",
    "bmi_category",
]

CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + [
    "risk_profile_segment",
    "smoking_alcohol_segment",
    "age_bmi_segment",
    "admission_status",
]

RAW_MODEL_COLUMNS = [
    "years_of_insurance_with_us",
    "regular_checkup_last_year",
    "adventure_sports",
    "Occupation",
    "visited_doctor_last_1_year",
    "cholesterol_level",
    "daily_avg_steps",
    "age",
    "heart_disease_history",
    "other_major_disease_history",
    "Gender",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "Year_last_admitted",
    "Location",
    "weight",
    "covered_by_any_other_company",
    "Alcohol",
    "exercise",
    "weight_change_in_last_one_year",
    "fat_percentage",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent project column names and categorical typos fixed."""
    cleaned = df.copy()
    cleaned = cleaned.rename(columns=COLUMN_RENAMES)
    if "Occupation" in cleaned.columns:
        cleaned["Occupation"] = cleaned["Occupation"].replace({"Salried": "Salaried"})
    return cleaned


def make_age_band(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "66+"],
        right=True,
    ).astype("object").fillna("Unknown")


def make_bmi_category(bmi: pd.Series) -> pd.Series:
    return pd.cut(
        bmi,
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,
    ).astype("object").fillna("Unknown")


def cholesterol_to_midpoint(value: object) -> float:
    if pd.isna(value):
        return np.nan
    matches = re.findall(r"\d+(?:\.\d+)?", str(value))
    if len(matches) >= 2:
        return (float(matches[0]) + float(matches[1])) / 2.0
    if len(matches) == 1:
        return float(matches[0])
    return np.nan


def _category_score(series: pd.Series, mapping: dict[str, float], default: float = 0.0) -> pd.Series:
    normalized = series.astype("object").fillna("Unknown").astype(str).str.strip().str.lower()
    lowered_mapping = {str(key).strip().lower(): value for key, value in mapping.items()}
    return normalized.map(lowered_mapping).fillna(default).astype(float)


class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering fitted only on training data inside sklearn pipelines."""

    def __init__(self, cap_bmi: bool = True):
        self.cap_bmi = cap_bmi

    def fit(self, X: pd.DataFrame, y: object = None) -> "InsuranceFeatureEngineer":
        frame = clean_column_names(pd.DataFrame(X))
        frame["age_band"] = make_age_band(frame["age"])
        self.global_bmi_median_ = float(frame["bmi"].median())
        grouped = frame.groupby(["age_band", "Gender"], observed=False)["bmi"].median()
        self.bmi_group_medians_ = grouped.dropna().to_dict()
        admitted_year = pd.to_numeric(frame["Year_last_admitted"], errors="coerce")
        self.reference_year_ = int(admitted_year.max() + 1) if admitted_year.notna().any() else 2019
        self.bmi_cap_upper_ = float(frame["bmi"].quantile(0.995)) if self.cap_bmi else np.inf
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = clean_column_names(pd.DataFrame(X))
        frame["age_band"] = make_age_band(frame["age"])
        frame["cholesterol_midpoint"] = frame["cholesterol_level"].map(cholesterol_to_midpoint)

        bmi = pd.to_numeric(frame["bmi"], errors="coerce").copy()
        missing_bmi = bmi.isna()
        frame["bmi_missing_flag"] = missing_bmi.astype(int)
        if missing_bmi.any():
            fallback = frame.loc[missing_bmi, ["age_band", "Gender"]].apply(
                lambda row: self.bmi_group_medians_.get(
                    (row["age_band"], row["Gender"]), self.global_bmi_median_
                ),
                axis=1,
            )
            bmi.loc[missing_bmi] = fallback.astype(float)
        if self.cap_bmi:
            bmi = bmi.clip(lower=10, upper=self.bmi_cap_upper_)
        frame["bmi"] = bmi
        frame["bmi_category"] = make_bmi_category(frame["bmi"])

        admitted_year = pd.to_numeric(frame["Year_last_admitted"], errors="coerce")
        frame["was_admitted_before"] = admitted_year.notna().astype(int)
        frame["admission_year_missing_flag"] = admitted_year.isna().astype(int)
        frame["years_since_last_admitted"] = np.where(
            admitted_year.notna(), self.reference_year_ - admitted_year, 0
        )
        frame["any_major_disease_history"] = (
            (frame["heart_disease_history"].astype(int) == 1)
            | (frame["other_major_disease_history"].astype(int) == 1)
        ).astype(int)
        frame["weight_bmi_interaction"] = frame["weight"] * frame["bmi"]
        age = pd.to_numeric(frame["age"], errors="coerce")
        weight = pd.to_numeric(frame["weight"], errors="coerce")
        glucose = pd.to_numeric(frame["avg_glucose_level"], errors="coerce")
        steps = pd.to_numeric(frame["daily_avg_steps"], errors="coerce")
        fat_percentage = pd.to_numeric(frame["fat_percentage"], errors="coerce")
        regular_checkups = pd.to_numeric(frame["regular_checkup_last_year"], errors="coerce").fillna(0)
        doctor_visits = pd.to_numeric(frame["visited_doctor_last_1_year"], errors="coerce").fillna(0)
        heart_history = pd.to_numeric(frame["heart_disease_history"], errors="coerce").fillna(0)
        other_major_history = pd.to_numeric(frame["other_major_disease_history"], errors="coerce").fillna(0)
        adventure_sports = pd.to_numeric(frame["adventure_sports"], errors="coerce").fillna(0)

        smoking_score = _category_score(
            frame["smoking_status"],
            {"never smoked": 0.0, "formerly smoked": 1.0, "smokes": 2.0, "Unknown": 0.75},
            default=0.75,
        )
        alcohol_score = _category_score(
            frame["Alcohol"],
            {"No": 0.0, "Rare": 0.5, "Daily": 2.0},
            default=0.5,
        )
        exercise_risk = _category_score(
            frame["exercise"],
            {"Extreme": 0.0, "Moderate": 0.5, "No": 2.0},
            default=0.5,
        )

        frame["steps_per_age"] = steps / age.replace(0, np.nan)
        frame["smoking_risk_score"] = smoking_score
        frame["alcohol_risk_score"] = alcohol_score
        frame["exercise_risk_score"] = exercise_risk
        frame["obesity_risk_flag"] = (frame["bmi"] >= 30).astype(int)
        frame["high_glucose_flag"] = (glucose >= 140).astype(int)
        frame["high_cholesterol_flag"] = (frame["cholesterol_midpoint"] >= 200).astype(int)
        frame["sedentary_flag"] = ((steps < 5000) | (exercise_risk >= 2)).astype(int)
        frame["current_smoker_flag"] = (smoking_score >= 2).astype(int)
        frame["daily_alcohol_flag"] = (alcohol_score >= 2).astype(int)

        frame["medical_risk_score"] = (
            heart_history * 2.0
            + other_major_history * 2.0
            + frame["obesity_risk_flag"] * 1.2
            + frame["high_glucose_flag"] * 1.0
            + frame["high_cholesterol_flag"] * 0.8
            + frame["was_admitted_before"] * 0.7
        )
        frame["lifestyle_risk_score"] = (
            smoking_score
            + alcohol_score
            + exercise_risk
            + frame["sedentary_flag"] * 0.8
            + adventure_sports * 0.6
        )
        frame["preventive_care_score"] = regular_checkups.clip(0, 5) - (doctor_visits.clip(0, 12) / 6.0)
        frame["age_bmi_interaction"] = age * frame["bmi"]
        frame["bmi_glucose_interaction"] = frame["bmi"] * glucose
        frame["bmi_exercise_risk_interaction"] = frame["bmi"] * exercise_risk
        frame["age_medical_risk_interaction"] = age * frame["medical_risk_score"]
        frame["smoking_alcohol_interaction"] = smoking_score * alcohol_score
        frame["steps_exercise_interaction"] = steps * (2.0 - exercise_risk)
        frame["fat_bmi_interaction"] = fat_percentage * frame["bmi"]
        frame["doctor_visit_intensity"] = doctor_visits / age.replace(0, np.nan)
        frame["admission_recency_risk"] = frame["was_admitted_before"] / (
            pd.to_numeric(frame["years_since_last_admitted"], errors="coerce").fillna(0) + 1.0
        )
        frame["cholesterol_age_interaction"] = frame["cholesterol_midpoint"] * age

        frame["risk_profile_segment"] = np.select(
            [
                (frame["medical_risk_score"] >= 4) & (frame["lifestyle_risk_score"] >= 3),
                frame["medical_risk_score"] >= 4,
                frame["lifestyle_risk_score"] >= 3,
            ],
            ["High medical and lifestyle", "High medical", "High lifestyle"],
            default="Lower combined risk",
        )
        frame["smoking_alcohol_segment"] = (
            frame["smoking_status"].astype("object").fillna("Unknown").astype(str)
            + " | "
            + frame["Alcohol"].astype("object").fillna("Unknown").astype(str)
        )
        frame["age_bmi_segment"] = frame["age_band"].astype(str) + " | " + frame["bmi_category"].astype(str)
        frame["admission_status"] = np.where(frame["was_admitted_before"] == 1, "Previously admitted", "No known admission")

        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype("object").fillna("Unknown")

        return frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def make_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(model: object, feature_set: str = "enhanced") -> Pipeline:
    if feature_set == "base":
        numeric_features = BASE_NUMERIC_FEATURES
        categorical_features = BASE_CATEGORICAL_FEATURES
    elif feature_set == "enhanced":
        numeric_features = NUMERIC_FEATURES
        categorical_features = CATEGORICAL_FEATURES
    else:
        raise ValueError("feature_set must be either 'base' or 'enhanced'")
    return Pipeline(
        steps=[
            ("features", InsuranceFeatureEngineer()),
            ("preprocess", make_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr)
    return float(np.mean(np.where(denominator == 0, 0, 2 * np.abs(y_pred_arr - y_true_arr) / denominator)) * 100)


def evaluate_regression(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(y_true_arr), 1.0)
    return {
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "R2": float(r2_score(y_true_arr, y_pred_arr)),
        "MAPE": float(np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100),
        "SMAPE": smape(y_true_arr, y_pred_arr),
    }


def get_transformed_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    preprocess = fitted_pipeline.named_steps["preprocess"]
    return list(preprocess.get_feature_names_out())
