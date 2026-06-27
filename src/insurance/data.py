"""Data loading and schema cleanup.

Keeps raw I/O and schema correction separate from feature engineering so the
same clean DataFrame feeds both the EDA notebooks and the production pipeline.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Load the raw insurance CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the known schema/category issues from the problem statement.

    - Rename the typo columns (``regular_checkup_lasy_year`` etc.).
    - Fix the ``Occupation`` category typo (``Salried`` -> ``Salaried``).
    - Strip stray whitespace from object columns.

    The transformation is non-destructive (operates on a copy) and idempotent.
    """
    df = df.copy()
    df = df.rename(columns=config.COLUMN_RENAMES)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    if "Occupation" in df.columns:
        df["Occupation"] = df["Occupation"].replace(config.OCCUPATION_FIXES)

    return df


def load_clean(path=config.DATA_PATH) -> pd.DataFrame:
    """Load and clean the dataset in one call."""
    return clean_schema(load_raw(path))


def split_X_y(df: pd.DataFrame):
    """Split a clean DataFrame into features ``X`` and target ``y``.

    ``applicant_id`` is retained in ``X`` here; the feature-engineering step
    drops it so it never reaches the model.
    """
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])
    return X, y
