"""Tests for prediction utilities and schemas (artifact-aware)."""

import pytest

from insurance import config
from insurance.predict import predict, risk_category
from insurance.schemas import FIELD_SPECS, InsuranceApplicant


def test_risk_category_bands():
    t = {"low_to_medium": 100.0, "medium_to_high": 200.0}
    assert risk_category(50, t) == "Low"
    assert risk_category(150, t) == "Medium"
    assert risk_category(250, t) == "High"


def test_applicant_to_frame_columns():
    frame = InsuranceApplicant().to_frame()
    assert len(frame) == 1
    # cleaned schema columns must be present
    for col in ["Occupation", "cholesterol_level", "bmi", "Year_last_admitted"]:
        assert col in frame.columns


def test_field_specs_cover_model_inputs():
    spec_names = {f["name"] for f in FIELD_SPECS}
    model_fields = set(InsuranceApplicant().model_dump().keys())
    assert spec_names == model_fields


@pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="model artifact not trained yet"
)
def test_predict_smoke():
    results = predict(InsuranceApplicant().to_frame())
    assert len(results) == 1
    r = results[0]
    assert r["predicted_cost"] > 0
    assert r["risk_category"] in {"Low", "Medium", "High"}
