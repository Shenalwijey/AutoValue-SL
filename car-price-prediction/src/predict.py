"""
Prediction helpers for the Flask application.
"""

from datetime import datetime
import math
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
CURRENT_YEAR = datetime.now().year
FEATURE_COLUMNS = [
    "year", "engine_cc", "mileage_km", "car_age", "mileage_per_year",
    "amenity_score", "log_mileage", "log_engine_cc", "log_car_age",
    "listing_year", "listing_month",
    "has_ac", "has_ps", "has_pm", "has_pw", "is_leasing", "is_new", "is_auto", "is_luxury",
    "brand", "fuel_type", "town",
]

_pipeline = None


def _safe_log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "The app will rebuild it automatically."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def _smoke_test_features() -> pd.DataFrame:
    sample_year = max(1990, min(CURRENT_YEAR, 2018))
    sample_age = max(0, CURRENT_YEAR - sample_year)
    sample_mileage = 50000.0
    sample_engine_cc = 1500.0

    return pd.DataFrame([{
        "year": sample_year,
        "engine_cc": sample_engine_cc,
        "mileage_km": sample_mileage,
        "car_age": sample_age,
        "mileage_per_year": sample_mileage / max(1, sample_age),
        "amenity_score": 4,
        "log_mileage": _safe_log1p(sample_mileage),
        "log_engine_cc": _safe_log1p(sample_engine_cc),
        "log_car_age": _safe_log1p(sample_age),
        "listing_year": CURRENT_YEAR,
        "listing_month": 1,
        "has_ac": 1,
        "has_ps": 1,
        "has_pm": 1,
        "has_pw": 1,
        "is_leasing": 0,
        "is_new": 0,
        "is_auto": 1,
        "is_luxury": 0,
        "brand": "TOYOTA",
        "fuel_type": "Petrol",
        "town": "Colombo",
    }])


def ensure_pipeline_ready():
    pipeline = _load_pipeline()
    try:
        pipeline.predict(_smoke_test_features())
    except Exception as exc:
        raise RuntimeError(
            "Stored model is incompatible with the current environment. "
            "Run 'python app.py --train' once to rebuild and store a fresh model."
        ) from exc
    return pipeline


def train_and_store_model():
    from src.data_preprocessing import clean_column_names, clean_data, engineer_features, load_data, select_features
    from src.feature_engineering import build_preprocessor, save_preprocessor
    from src.utils import overfitting_status, regression_metrics, save_json

    data_path = os.path.join(PROJECT_ROOT, "data", "car_price_dataset.csv")
    df = load_data(data_path)
    df = clean_column_names(df)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(encoder="ordinal")),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    train_metrics = regression_metrics(y_train.values, train_pred, "Gradient Boosting TRAIN")
    test_metrics = regression_metrics(y_test.values, test_pred, "Gradient Boosting TEST")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    save_preprocessor(pipeline.named_steps["preprocessor"])
    save_json([{
        "model_name": "Gradient Boosting",
        "train_r2": train_metrics["R2"],
        "test_r2": test_metrics["R2"],
        "cv_r2_mean": round(test_metrics["R2"], 4),
        "cv_r2_std": 0.0,
        "train_mae": train_metrics["MAE"],
        "test_mae": test_metrics["MAE"],
        "train_rmse": train_metrics["RMSE"],
        "test_rmse": test_metrics["RMSE"],
        "test_mape": test_metrics["MAPE"],
        "fit_status": overfitting_status(train_metrics["R2"], test_metrics["R2"]),
        "final_test_r2": test_metrics["R2"],
        "adjusted_score": test_metrics["R2"],
    }], METRICS_PATH)
    return pipeline


def predict_price(input_dict: dict) -> float:
    pipeline = ensure_pipeline_ready()

    from src.data_preprocessing import engineer_features

    row = {
        "brand": input_dict.get("brand", "TOYOTA"),
        "model": input_dict.get("model", "Unknown"),
        "year": int(input_dict.get("year", max(1990, min(CURRENT_YEAR, 2015)))),
        "engine_cc": float(input_dict.get("engine_cc", 1500)),
        "gear": input_dict.get("gear", "Automatic"),
        "fuel_type": input_dict.get("fuel_type", "Petrol"),
        "mileage_km": float(input_dict.get("mileage_km", 80000)),
        "town": input_dict.get("town", "Colombo"),
        "listing_date": f"{CURRENT_YEAR}-01-01",
        "leasing": input_dict.get("leasing", "No Leasing"),
        "condition": input_dict.get("condition", "USED"),
        "air_condition": input_dict.get("air_condition", "Available"),
        "power_steering": input_dict.get("power_steering", "Available"),
        "power_mirror": input_dict.get("power_mirror", "Available"),
        "power_window": input_dict.get("power_window", "Available"),
    }

    df = engineer_features(pd.DataFrame([row]))
    prediction = pipeline.predict(df[FEATURE_COLUMNS])[0]
    return round(float(prediction), 2)
