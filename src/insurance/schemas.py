"""Pydantic I/O schemas and form field metadata.

``FIELD_SPECS`` is the single source of truth for the applicant input fields:
the FastAPI ``/schema`` endpoint serves it so the React form can render the
correct inputs/dropdowns, and the Streamlit app uses it too.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Category options (from the cleaned dataset)
# --------------------------------------------------------------------------- #
OCCUPATIONS = ["Student", "Business", "Salaried"]
CHOLESTEROL_LEVELS = [
    "125 to 150",
    "150 to 175",
    "175 to 200",
    "200 to 225",
    "225 to 250",
]
GENDERS = ["Male", "Female"]
SMOKING_STATUSES = ["never smoked", "formerly smoked", "smokes", "Unknown"]
LOCATIONS = [
    "Bangalore", "Jaipur", "Bhubaneswar", "Mangalore", "Delhi", "Ahmedabad",
    "Guwahati", "Chennai", "Kanpur", "Nagpur", "Mumbai", "Lucknow", "Pune",
    "Kolkata", "Surat",
]
YES_NO = ["N", "Y"]
ALCOHOL_LEVELS = ["No", "Rare", "Daily"]
EXERCISE_LEVELS = ["No", "Moderate", "Extreme"]

# Field metadata that drives the UI (React + Streamlit) and /schema endpoint.
FIELD_SPECS = [
    {"name": "age", "label": "Age", "kind": "number", "min": 16, "max": 100, "step": 1, "default": 40},
    {"name": "Gender", "label": "Gender", "kind": "select", "options": GENDERS, "default": "Male"},
    {"name": "Occupation", "label": "Occupation", "kind": "select", "options": OCCUPATIONS, "default": "Salaried"},
    {"name": "Location", "label": "Location", "kind": "select", "options": LOCATIONS, "default": "Bangalore"},
    {"name": "weight", "label": "Weight (kg)", "kind": "number", "min": 30, "max": 130, "step": 1, "default": 72},
    {"name": "bmi", "label": "BMI", "kind": "number", "min": 10, "max": 60, "step": 0.1, "default": 30.0, "optional": True},
    {"name": "fat_percentage", "label": "Fat percentage", "kind": "number", "min": 5, "max": 60, "step": 1, "default": 28},
    {"name": "weight_change_in_last_one_year", "label": "Weight change last year", "kind": "number", "min": 0, "max": 10, "step": 1, "default": 2},
    {"name": "avg_glucose_level", "label": "Avg glucose level", "kind": "number", "min": 50, "max": 320, "step": 1, "default": 130},
    {"name": "cholesterol_level", "label": "Cholesterol level", "kind": "select", "options": CHOLESTEROL_LEVELS, "default": "150 to 175"},
    {"name": "daily_avg_steps", "label": "Daily avg steps", "kind": "number", "min": 1000, "max": 12000, "step": 50, "default": 5000},
    {"name": "exercise", "label": "Exercise", "kind": "select", "options": EXERCISE_LEVELS, "default": "Moderate"},
    {"name": "smoking_status", "label": "Smoking status", "kind": "select", "options": SMOKING_STATUSES, "default": "never smoked"},
    {"name": "Alcohol", "label": "Alcohol", "kind": "select", "options": ALCOHOL_LEVELS, "default": "Rare"},
    {"name": "adventure_sports", "label": "Adventure sports", "kind": "binary", "options": [0, 1], "default": 0},
    {"name": "heart_disease_history", "label": "Heart disease history", "kind": "binary", "options": [0, 1], "default": 0},
    {"name": "other_major_disease_history", "label": "Other major disease history", "kind": "binary", "options": [0, 1], "default": 0},
    {"name": "regular_checkup_last_year", "label": "Regular checkups last year", "kind": "number", "min": 0, "max": 12, "step": 1, "default": 1},
    {"name": "visited_doctor_last_1_year", "label": "Doctor visits last year", "kind": "number", "min": 0, "max": 12, "step": 1, "default": 3},
    {"name": "years_of_insurance_with_us", "label": "Years insured with us", "kind": "number", "min": 0, "max": 20, "step": 1, "default": 3},
    {"name": "covered_by_any_other_company", "label": "Covered by another insurer", "kind": "select", "options": YES_NO, "default": "N"},
    {"name": "Year_last_admitted", "label": "Year last admitted (blank = never)", "kind": "number", "min": 1980, "max": 2025, "step": 1, "default": None, "optional": True},
]


class InsuranceApplicant(BaseModel):
    """One applicant's features (cleaned schema, no id / target)."""

    years_of_insurance_with_us: int = 3
    regular_checkup_last_year: int = 1
    adventure_sports: int = 0
    Occupation: str = "Salaried"
    visited_doctor_last_1_year: int = 3
    cholesterol_level: str = "150 to 175"
    daily_avg_steps: int = 5000
    age: int = 40
    heart_disease_history: int = 0
    other_major_disease_history: int = 0
    Gender: str = "Male"
    avg_glucose_level: int = 130
    bmi: Optional[float] = 30.0
    smoking_status: str = "never smoked"
    Year_last_admitted: Optional[float] = None
    Location: str = "Bangalore"
    weight: int = 72
    covered_by_any_other_company: str = "N"
    Alcohol: str = "Rare"
    exercise: str = "Moderate"
    weight_change_in_last_one_year: int = 2
    fat_percentage: int = 28

    def to_frame(self) -> pd.DataFrame:
        """Return a single-row DataFrame with the model's expected columns."""
        return pd.DataFrame([self.model_dump()])


class Driver(BaseModel):
    feature: str
    importance: float


class PredictionResponse(BaseModel):
    predicted_cost: float = Field(..., description="Estimated insurance cost")
    risk_category: str = Field(..., description="Low / Medium / High")
    currency: str = "INR"
    top_drivers: list[Driver] = []
