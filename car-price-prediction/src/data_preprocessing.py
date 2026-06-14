"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, and preparation steps for the
Car Price Prediction project.

Dataset: car_price_dataset.csv (9,788 Sri Lankan vehicle listings)
Target : Price (in LKR Lakhs)
"""

from datetime import datetime
import json
import os

import numpy as np
import pandas as pd

CURRENT_YEAR = datetime.now().year
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and drop the unnamed index column if present."""
    df = pd.read_csv(filepath, index_col=0)
    df = df.reset_index(drop=True)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. INITIAL INSPECTION
# ─────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> dict:
    """Return a summary dictionary of key dataset statistics."""
    summary = {
        "shape"          : df.shape,
        "columns"        : list(df.columns),
        "dtypes"         : df.dtypes.astype(str).to_dict(),
        "missing_values" : df.isnull().sum().to_dict(),
        "duplicates"     : int(df.duplicated().sum()),
        "numeric_cols"   : list(df.select_dtypes(include=np.number).columns),
        "categorical_cols": list(df.select_dtypes(include="object").columns),
        "target_stats"   : df["Price"].describe().to_dict(),
    }
    return summary


# ─────────────────────────────────────────────
# 3. CLEAN COLUMN NAMES
# ─────────────────────────────────────────────

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names: strip spaces, replace special chars."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[\s\(\)/]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.rstrip("_")
    )
    # Manual renames for clarity
    rename_map = {
        "Engine_cc"      : "engine_cc",   # after regex: "Engine (cc)" → "Engine_cc"
        "Millage_KM"     : "mileage_km",  # "Millage(KM)" → "Millage_KM"
        "Fuel_Type"      : "fuel_type",
        "YOM"            : "year",
        "Gear"           : "gear",
        "Brand"          : "brand",
        "Model"          : "model",
        "Town"           : "town",
        "Date"           : "listing_date",
        "Leasing"        : "leasing",
        "Condition"      : "condition",
        "AIR_CONDITION"  : "air_condition",
        "POWER_STEERING" : "power_steering",
        "POWER_MIRROR"   : "power_mirror",
        "POWER_WINDOW"   : "power_window",
        "Price"          : "price",
    }
    df = df.rename(columns=rename_map)
    print(f"[INFO] Columns after cleaning: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 4. DATA CLEANING
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates, handle outliers, fix obvious data-quality issues.
    """
    df = df.copy()

    # 4a. Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows")

    # 4b. Drop rows where price is 0 or negative
    df = df[df["price"] > 0]

    # 4c. Remove extreme price outliers (top 1% – cars > ~300 lakhs)
    p99 = df["price"].quantile(0.99)
    df = df[df["price"] <= p99]
    print(f"[INFO] Price cap at 99th percentile: {p99:.1f} lakhs "
          f"→ {len(df)} rows remaining")

    # 4d. Year sanity: keep 1990-2024 (tiny handful of 1956 outliers)
    df = df[(df["year"] >= 1990) & (df["year"] <= CURRENT_YEAR)]

    # 4e. Engine CC sanity: 500–5000
    df = df[(df["engine_cc"] >= 500) & (df["engine_cc"] <= 5000)]

    # 4f. Mileage sanity: 0–750000
    df = df[(df["mileage_km"] >= 0) & (df["mileage_km"] <= 750000)]

    print(f"[INFO] Shape after cleaning: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns to help the model
    capture non-linear relationships.
    """
    df = df.copy()

    # 5a. Car age based on the current calendar year
    df["car_age"] = CURRENT_YEAR - df["year"]

    # 5b. Mileage per year (wear rate)
    df["mileage_per_year"] = df["mileage_km"] / (df["car_age"].clip(lower=1))

    # 5c. Binary encoding of feature amenities
    df["has_ac"]     = (df["air_condition"]   == "Available").astype(int)
    df["has_ps"]     = (df["power_steering"]  == "Available").astype(int)
    df["has_pm"]     = (df["power_mirror"]    == "Available").astype(int)
    df["has_pw"]     = (df["power_window"]    == "Available").astype(int)
    df["amenity_score"] = df[["has_ac","has_ps","has_pm","has_pw"]].sum(axis=1)

    # 5d. Leasing binary flag
    df["is_leasing"] = (df["leasing"] == "Ongoing Lease").astype(int)

    # 5e. Is new condition
    df["is_new"] = (df["condition"] == "NEW").astype(int)

    # 5f. Is automatic
    df["is_auto"] = (df["gear"] == "Automatic").astype(int)

    # 5g. Log-transform skewed numerics for tree models benefit less,
    #     but linear models need this; we add log columns as extra features
    df["log_mileage"]    = np.log1p(df["mileage_km"])
    df["log_engine_cc"]  = np.log1p(df["engine_cc"])
    df["log_car_age"]    = np.log1p(df["car_age"])

    # 5h. Extract listing year/month from date
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df["listing_year"]  = df["listing_date"].dt.year.fillna(CURRENT_YEAR).astype(int)
    df["listing_month"] = df["listing_date"].dt.month.fillna(1).astype(int)

    # 5i. Luxury brand flag
    luxury_brands = {"BMW","MERCEDES-BENZ","AUDI","PORSCHE","LEXUS","VOLVO","JAGUAR","LAND ROVER"}
    df["is_luxury"] = df["brand"].isin(luxury_brands).astype(int)

    print(f"[INFO] Shape after feature engineering: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 6. SELECT FINAL FEATURE SET
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    "year", "engine_cc", "mileage_km", "car_age",
    "mileage_per_year", "amenity_score",
    "log_mileage", "log_engine_cc", "log_car_age",
    "listing_year", "listing_month",
]

BINARY_FEATURES = [
    "has_ac", "has_ps", "has_pm", "has_pw",
    "is_leasing", "is_new", "is_auto", "is_luxury",
]

CATEGORICAL_FEATURES = [
    "brand", "fuel_type", "town",
]

TARGET = "price"

# Columns to DROP (leakage / redundant raw sources)
DROP_COLS = [
    "model",           # too high cardinality; brand already captured
    "listing_date",    # raw date replaced by year/month features
    "gear",            # replaced by is_auto
    "leasing",         # replaced by is_leasing
    "condition",       # replaced by is_new
    "air_condition",   # replaced by has_ac
    "power_steering",  # replaced by has_ps
    "power_mirror",    # replaced by has_pm
    "power_window",    # replaced by has_pw
]


def select_features(df: pd.DataFrame) -> tuple:
    """
    Return (X, y) DataFrames with only selected features.
    Also saves the feature list to models/selected_features.json.
    """
    feature_cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    # Keep only columns that actually exist (safety)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Save feature info
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "selected_features.json"), "w") as f:
        json.dump({
            "numeric"     : NUMERIC_FEATURES,
            "binary"      : BINARY_FEATURES,
            "categorical" : CATEGORICAL_FEATURES,
            "all_features": feature_cols,
            "target"      : TARGET,
        }, f, indent=2)

    print(f"[INFO] Feature set: {len(feature_cols)} features, {len(y)} samples")
    return X, y
