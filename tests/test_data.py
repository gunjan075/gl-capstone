"""Tests for schema cleanup and data loading."""

import pandas as pd

from insurance import config, data


def test_clean_schema_renames_typo_columns():
    raw = pd.DataFrame(
        {
            "regular_checkup_lasy_year": [1],
            "heart_decs_history": [0],
            "other_major_decs_history": [1],
            "Occupation": ["Salried"],
        }
    )
    clean = data.clean_schema(raw)
    assert "regular_checkup_last_year" in clean.columns
    assert "heart_disease_history" in clean.columns
    assert "other_major_disease_history" in clean.columns
    assert "regular_checkup_lasy_year" not in clean.columns


def test_clean_schema_fixes_occupation_typo():
    clean = data.clean_schema(pd.DataFrame({"Occupation": ["Salried", "Student"]}))
    assert set(clean["Occupation"]) == {"Salaried", "Student"}


def test_load_clean_shape_and_target():
    df = data.load_clean()
    assert df.shape == (25000, 24)
    assert config.TARGET in df.columns
    assert "Salried" not in df["Occupation"].unique()


def test_split_x_y():
    df = data.load_clean()
    X, y = data.split_X_y(df)
    assert config.TARGET not in X.columns
    assert len(X) == len(y) == 25000
