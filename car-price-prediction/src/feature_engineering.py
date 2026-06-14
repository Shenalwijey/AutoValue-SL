"""
feature_engineering.py
-----------------------
Builds the scikit-learn preprocessing pipeline (ColumnTransformer)
used for both training and inference.

Pipeline design:
  - Numeric features  → SimpleImputer (median) → StandardScaler
  - Binary features   → SimpleImputer (0)       → PassThrough
  - Categorical feats → SimpleImputer (most_frequent) → OrdinalEncoder
                        (OrdinalEncoder handles unknown categories gracefully
                         and is faster than OneHot for high-cardinality cols
                         like brand/town when used with tree models.
                         For linear models we use OneHotEncoder via a flag.)
"""

import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.data_preprocessing import (
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_preprocessor(encoder: str = "ordinal") -> ColumnTransformer:
    """
    Build and return a ColumnTransformer preprocessing pipeline.

    Parameters
    ----------
    encoder : str
        'ordinal'  – OrdinalEncoder (best for tree-based models)
        'onehot'   – OneHotEncoder  (best for linear models)

    Returns
    -------
    ColumnTransformer
    """
    # --- Numeric pipeline ---
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # --- Binary pipeline (already 0/1, just impute) ---
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    # --- Categorical pipeline ---
    if encoder == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", cat_encoder),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      NUMERIC_FEATURES),
            ("bin", binary_pipeline,       BINARY_FEATURES),
            ("cat", categorical_pipeline,  CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def save_preprocessor(preprocessor, path: str = "models/preprocessor.pkl"):
    """Persist fitted preprocessor to disk."""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[INFO] Preprocessor saved → {path}")


def load_preprocessor(path: str = "models/preprocessor.pkl"):
    """Load a previously saved preprocessor."""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return joblib.load(path)
