"""Central configuration: paths, column groups, and constants.

Column groups describe the schema *after* feature engineering
(:class:`insurance.features.FeatureEngineer`), which is the input the
preprocessing :class:`~sklearn.compose.ColumnTransformer` operates on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "insurance.csv"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MODEL_PATH = MODELS_DIR / "final_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

for _d in (MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Core constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def gpu_available() -> bool:
    """True if an NVIDIA GPU is usable (used to enable GPU boosting)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


# GPU for XGBoost/CatBoost. Override with INSURANCE_USE_GPU=0/1.
_gpu_env = os.environ.get("INSURANCE_USE_GPU")
USE_GPU = (_gpu_env == "1") if _gpu_env is not None else gpu_available()

TARGET = "insurance_cost"
ID_COL = "applicant_id"

# Schema cleanup (problem-statement typos)
COLUMN_RENAMES = {
    "regular_checkup_lasy_year": "regular_checkup_last_year",
    "heart_decs_history": "heart_disease_history",
    "other_major_decs_history": "other_major_disease_history",
}
OCCUPATION_FIXES = {"Salried": "Salaried"}

# Cholesterol ranges -> numeric midpoints (informed ordinal encoding)
CHOLESTEROL_MIDPOINTS = {
    "125 to 150": 137.5,
    "150 to 175": 162.5,
    "175 to 200": 187.5,
    "200 to 225": 212.5,
    "225 to 250": 237.5,
}

# Plausibility caps for BMI (extreme/unrealistic values only)
BMI_MIN, BMI_MAX = 12.0, 60.0

# Labels avoid characters XGBoost rejects in feature names ([ ] <)
AGE_BANDS = [0, 30, 40, 50, 60, 200]
AGE_BAND_LABELS = ["0-29", "30-39", "40-49", "50-59", "60+"]

# --------------------------------------------------------------------------- #
# Feature groups (post feature-engineering)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "years_of_insurance_with_us",
    "regular_checkup_last_year",
    "visited_doctor_last_1_year",
    "daily_avg_steps",
    "age",
    "avg_glucose_level",
    "bmi",
    "weight",
    "weight_change_in_last_one_year",
    "fat_percentage",
    "adventure_sports",
    "heart_disease_history",
    "other_major_disease_history",
    # engineered
    "cholesterol_midpoint",
    "years_since_last_admitted",
    "was_admitted_before",
    "any_major_disease_history",
    "weight_bmi_interaction",
    "steps_per_age",
]

NOMINAL_FEATURES = [
    "Occupation",
    "Gender",
    "smoking_status",
    "Location",
    "covered_by_any_other_company",
    "Alcohol",
    "exercise",
    # engineered
    "age_band",
    "bmi_category",
]

# Default risk-category fallbacks (overwritten with training tertiles at train time)
DEFAULT_RISK_THRESHOLDS = {"low_to_medium": 19410.0, "medium_to_high": 33718.0}
