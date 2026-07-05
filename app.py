from __future__ import annotations

import json
from pathlib import Path

import json
import joblib
import pandas as pd
import streamlit as st

# Required so pickle can resolve the custom sklearn transformer.
from insurance_modeling import InsuranceFeatureEngineer  # noqa: F401


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "models" / "final_model.pkl"
METADATA_PATH = ROOT / "outputs" / "models" / "model_metadata.json"
SCHEMA_PATH = ROOT / "outputs" / "models" / "app_schema.json"
CALIBRATOR_PATH = ROOT / "outputs" / "models" / "prediction_calibrator.pkl"
IMPORTANCE_PATH = ROOT / "outputs" / "models" / "feature_importance.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_calibrator():
    if CALIBRATOR_PATH.exists():
        return joblib.load(CALIBRATOR_PATH)
    return None


@st.cache_data
def load_json(path: Path, default: dict):
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return default


@st.cache_data
def load_importance():
    if IMPORTANCE_PATH.exists():
        return pd.read_csv(IMPORTANCE_PATH)
    return pd.DataFrame(columns=["feature", "importance_mean"])


def numeric_cfg(schema: dict, name: str, fallback_min: float, fallback_max: float, fallback_default: float) -> tuple[float, float, float]:
    cfg = schema.get("numeric_ranges", {}).get(name, {})
    return (
        float(cfg.get("min", fallback_min)),
        float(cfg.get("max", fallback_max)),
        float(cfg.get("median", fallback_default)),
    )


def categorical_options(schema: dict, name: str, fallback: list[str]) -> list[str]:
    values = schema.get("categorical_options", {}).get(name, fallback)
    return [str(value) for value in values if str(value) != "nan"] or fallback


def nearest_quote_band(value: float, levels: list[int]) -> float:
    if not levels:
        return value
    return float(min(levels, key=lambda level: abs(float(level) - value)))


def calibrated_value(raw_prediction: float, calibrator_bundle: dict | None) -> float | None:
    if not calibrator_bundle or "calibrator" not in calibrator_bundle:
        return None
    return float(calibrator_bundle["calibrator"].predict([raw_prediction])[0])


st.set_page_config(page_title="Insurance Price Prediction", page_icon="IP", layout="wide")
st.title("Insurance Price Prediction")
st.caption("Decision-support estimator for health, lifestyle, habit, and demographic inputs.")

model = load_model()
metadata = load_json(METADATA_PATH, {})
schema = load_json(SCHEMA_PATH, {})
calibrator_bundle = load_calibrator()
importance = load_importance()
thresholds = metadata.get("risk_thresholds", {"low_medium": 18000, "medium_high": 35000})
target_grid = schema.get("target_grid", metadata.get("target_grid", {}))
valid_levels = target_grid.get("valid_target_levels", [])
deployment_variant = metadata.get("deployment_variant", {})
quote_variant = deployment_variant.get("quote_band", "rounded_to_nearest_price_band")
risk_variant = deployment_variant.get("risk_category", "raw_continuous")

with st.form("prediction_form"):
    left, middle, right = st.columns(3)
    with left:
        age_min, age_max, age_default = numeric_cfg(schema, "age", 16, 74, 45)
        age = st.slider("Age", int(age_min), int(age_max), int(age_default))
        gender = st.selectbox("Gender", categorical_options(schema, "Gender", ["Female", "Male"]))
        occupation = st.selectbox("Occupation", categorical_options(schema, "Occupation", ["Business", "Salaried", "Student"]))
        location = st.selectbox("Location", categorical_options(schema, "Location", ["Delhi", "Mumbai", "Bangalore"]))
        yi_min, yi_max, yi_default = numeric_cfg(schema, "years_of_insurance_with_us", 0, 8, 4)
        years_insured = st.slider("Years with insurer", int(yi_min), int(yi_max), int(yi_default))
        other_coverage = st.selectbox("Covered by another company", categorical_options(schema, "covered_by_any_other_company", ["N", "Y"]))
    with middle:
        weight_min, weight_max, weight_default = numeric_cfg(schema, "weight", 52, 110, 72)
        weight = st.slider("Weight", int(weight_min), int(weight_max), int(weight_default))
        bmi_unknown = st.checkbox("BMI unknown")
        bmi_min, bmi_max, bmi_default = numeric_cfg(schema, "bmi", 12.0, 60.0, 30.5)
        bmi = st.slider("BMI", float(bmi_min), float(bmi_max), float(bmi_default), step=0.1, disabled=bmi_unknown)
        fat_min, fat_max, fat_default = numeric_cfg(schema, "fat_percentage", 10, 45, 29)
        fat_percentage = st.slider("Fat percentage", int(fat_min), int(fat_max), int(fat_default))
        glucose_min, glucose_max, glucose_default = numeric_cfg(schema, "avg_glucose_level", 55, 280, 168)
        avg_glucose_level = st.slider("Average glucose level", int(glucose_min), int(glucose_max), int(glucose_default))
        cholesterol_level = st.selectbox("Cholesterol level", categorical_options(schema, "cholesterol_level", ["125 to 150", "150 to 175", "175 to 200", "200 to 225", "225 to 250"]))
        wc_min, wc_max, wc_default = numeric_cfg(schema, "weight_change_in_last_one_year", 0, 6, 2)
        weight_change = st.slider("Weight change in last year", int(wc_min), int(wc_max), int(wc_default))
    with right:
        steps_min, steps_max, steps_default = numeric_cfg(schema, "daily_avg_steps", 2000, 12000, 5200)
        daily_steps = st.slider("Daily average steps", int(steps_min), int(steps_max), int(steps_default), step=100)
        exercise = st.selectbox("Exercise", categorical_options(schema, "exercise", ["No", "Moderate", "Extreme"]))
        alcohol = st.selectbox("Alcohol", categorical_options(schema, "Alcohol", ["No", "Rare", "Daily"]))
        smoking_status = st.selectbox("Smoking status", categorical_options(schema, "smoking_status", ["never smoked", "formerly smoked", "smokes", "Unknown"]))
        adventure_sports = st.selectbox("Adventure sports", [0, 1])
        rc_min, rc_max, rc_default = numeric_cfg(schema, "regular_checkup_last_year", 0, 5, 1)
        regular_checkup = st.slider("Regular checkups last year", int(rc_min), int(rc_max), int(rc_default))

    c1, c2, c3 = st.columns(3)
    with c1:
        vd_min, vd_max, vd_default = numeric_cfg(schema, "visited_doctor_last_1_year", 0, 12, 3)
        visited_doctor = st.slider("Doctor visits last year", int(vd_min), int(vd_max), int(vd_default))
    with c2:
        heart_history = st.selectbox("Heart disease history", [0, 1])
    with c3:
        other_major_history = st.selectbox("Other major disease history", [0, 1])

    admitted_before = st.checkbox("Customer was admitted before")
    year_last_admitted = None
    if admitted_before:
        yla_min, yla_max, yla_default = numeric_cfg(schema, "Year_last_admitted", 1990, 2018, 2010)
        year_last_admitted = st.slider("Year last admitted", int(yla_min), int(yla_max), int(yla_default))

    submitted = st.form_submit_button("Predict insurance cost")

if submitted:
    input_row = pd.DataFrame(
        [
            {
                "years_of_insurance_with_us": years_insured,
                "regular_checkup_last_year": regular_checkup,
                "adventure_sports": adventure_sports,
                "Occupation": occupation,
                "visited_doctor_last_1_year": visited_doctor,
                "cholesterol_level": cholesterol_level,
                "daily_avg_steps": daily_steps,
                "age": age,
                "heart_disease_history": heart_history,
                "other_major_disease_history": other_major_history,
                "Gender": gender,
                "avg_glucose_level": avg_glucose_level,
                "bmi": None if bmi_unknown else bmi,
                "smoking_status": smoking_status,
                "Year_last_admitted": year_last_admitted,
                "Location": location,
                "weight": weight,
                "covered_by_any_other_company": other_coverage,
                "Alcohol": alcohol,
                "exercise": exercise,
                "weight_change_in_last_one_year": weight_change,
                "fat_percentage": fat_percentage,
            }
        ]
    )
    raw_prediction = float(model.predict(input_row)[0])
    calibrated_prediction = calibrated_value(raw_prediction, calibrator_bundle)
    quote_source = raw_prediction
    if quote_variant == "calibrated_then_rounded" and calibrated_prediction is not None:
        quote_source = calibrated_prediction
    quote_band = nearest_quote_band(quote_source, valid_levels)
    risk_source = raw_prediction
    if risk_variant == "calibrated_continuous" and calibrated_prediction is not None:
        risk_source = calibrated_prediction

    if risk_source < thresholds["low_medium"]:
        risk = "Low"
    elif risk_source < thresholds["medium_high"]:
        risk = "Medium"
    else:
        risk = "High"

    m1, m2, m3 = st.columns(3)
    m1.metric("Raw predicted cost", f"INR {raw_prediction:,.0f}")
    if calibrated_prediction is not None:
        m2.metric("Calibrated predicted cost", f"INR {calibrated_prediction:,.0f}")
    else:
        m2.metric("Calibrated predicted cost", "Not available")
    m3.metric("Nearest quote band", f"INR {quote_band:,.0f}")
    st.info(f"Risk category: {risk}")
    st.caption(
        "Calibration improved some percentage/band metrics but did not improve MAE/RMSE; "
        "quote-band output uses the selected deployment variant from model metadata."
    )
    if target_grid:
        step = target_grid.get("target_grid_step")
        step_text = f"{step:,.0f}" if isinstance(step, (int, float)) else "unknown"
        st.caption(
            f"Detected pricing grid: {target_grid.get('target_unique_count')} valid bands, "
            f"step INR {step_text}."
        )
    if not importance.empty:
        st.subheader("Top model drivers")
        st.dataframe(importance.head(8), use_container_width=True, hide_index=True)

st.warning(
    "Disclaimer: This model supports pricing analysis and triage. It should not be used as the final underwriting decision without policy, compliance, and human review."
)
