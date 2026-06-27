"""Tests for the feature-engineering transformer."""

from insurance import config, data
from insurance.features import FeatureEngineer


def _fitted():
    df = data.load_clean()
    X, _ = data.split_X_y(df)
    fe = FeatureEngineer().fit(X)
    return fe, X


def test_output_columns_match_config():
    fe, X = _fitted()
    out = fe.transform(X)
    assert list(out.columns) == config.NUMERIC_FEATURES + config.NOMINAL_FEATURES
    assert config.ID_COL not in out.columns


def test_no_missing_bmi_after_imputation():
    fe, X = _fitted()
    out = fe.transform(X)
    assert out["bmi"].isna().sum() == 0
    assert out["bmi"].between(config.BMI_MIN, config.BMI_MAX).all()


def test_engineered_features_present():
    fe, X = _fitted()
    out = fe.transform(X)
    for col in [
        "cholesterol_midpoint",
        "was_admitted_before",
        "years_since_last_admitted",
        "any_major_disease_history",
        "weight_bmi_interaction",
        "steps_per_age",
        "age_band",
        "bmi_category",
    ]:
        assert col in out.columns
    assert set(out["was_admitted_before"].unique()) <= {0, 1}
