"""
Flask web application for AutoValue SL.
"""

from datetime import datetime
import json
import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "car_price_dataset.csv")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
CURRENT_YEAR = datetime.now().year

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


MODEL_OPTIONS = []
BRAND_MODEL_MAP = {}
try:
    if os.path.exists(DATASET_PATH):
        df_data = pd.read_csv(DATASET_PATH)
        brand_col = next((c for c in df_data.columns if c.strip().lower() == "brand"), None)
        model_col = next((c for c in df_data.columns if c.strip().lower() == "model"), None)
        if model_col:
            MODEL_OPTIONS = sorted(df_data[model_col].dropna().astype(str).str.upper().unique().tolist())
        if brand_col and model_col:
            pairs = (
                df_data[[brand_col, model_col]]
                .dropna()
                .astype(str)
            )
            for brand, group in pairs.groupby(brand_col):
                normalized_brand = brand.strip().upper()
                models = sorted({model.strip().upper() for model in group[model_col] if model.strip()})
                if models:
                    BRAND_MODEL_MAP[normalized_brand] = models
except Exception:
    MODEL_OPTIONS = []
    BRAND_MODEL_MAP = {}


BRANDS = sorted([
    "AUDI", "BMW", "CHEVROLET", "CHRYSLER", "CITROEN", "DAEWOO", "DAIHATSU",
    "DODGE", "FIAT", "FORD", "HONDA", "HYUNDAI", "ISUZU", "JAGUAR", "JEEP",
    "KIA", "LADA", "LAND ROVER", "LEXUS", "MAHINDRA", "MARUTI", "MAZDA",
    "MERCEDES-BENZ", "MICRO", "MITSUBISHI", "NISSAN", "OPEL", "PERODUA",
    "PEUGEOT", "PORSCHE", "PROTON", "RENAULT", "SEAT", "SKODA", "SUBARU",
    "SUZUKI", "TATA", "TOYOTA", "VOLKSWAGEN", "VOLVO",
])

TOWNS = sorted([
    "Ampara", "Anuradhapura", "Badulla", "Batticaloa", "Colombo", "Dehiwala-Mount-Lavinia",
    "Galle", "Gampaha", "Hambantota", "Jaffna", "Kalutara", "Kandy", "Kegalle",
    "Kilinochchi", "Kurunegala", "Mannar", "Matale", "Matara", "Monaragala",
    "Mullaitivu", "Negombo", "Nuwara Eliya", "Polonnaruwa", "Puttalam",
    "Ratnapura", "Trincomalee", "Vavuniya",
])

FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric"]
GEAR_TYPES = ["Automatic", "Manual"]
CONDITIONS = ["USED", "NEW"]
LEASING = ["No Leasing", "Ongoing Lease"]
AMENITY_OPTS = ["Available", "Not Available"]


def _parse_form_input(form_data):
    return {
        "brand": form_data.get("brand", "TOYOTA"),
        "model": form_data.get("model", "Unknown"),
        "year": int(form_data.get("year", max(1990, min(CURRENT_YEAR, 2015)))),
        "engine_cc": float(form_data.get("engine_cc", 1500)),
        "gear": form_data.get("gear", "Automatic"),
        "fuel_type": form_data.get("fuel_type", "Petrol"),
        "mileage_km": float(form_data.get("mileage_km", 80000)),
        "town": form_data.get("town", "Colombo"),
        "leasing": form_data.get("leasing", "No Leasing"),
        "condition": form_data.get("condition", "USED"),
        "air_condition": form_data.get("air_condition", "Available"),
        "power_steering": form_data.get("power_steering", "Available"),
        "power_mirror": form_data.get("power_mirror", "Available"),
        "power_window": form_data.get("power_window", "Available"),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "predict.html",
            brands=BRANDS,
            towns=TOWNS,
            models=MODEL_OPTIONS,
            brand_model_map=BRAND_MODEL_MAP,
            fuel_types=FUEL_TYPES,
            gear_types=GEAR_TYPES,
            conditions=CONDITIONS,
            leasing_opts=LEASING,
            amenity_opts=AMENITY_OPTS,
            current_year=CURRENT_YEAR,
        )

    try:
        from src.predict import predict_price

        input_data = _parse_form_input(request.form)
        predicted_price = predict_price(input_data)
        low = round(predicted_price * 0.90, 2)
        high = round(predicted_price * 1.10, 2)

        return render_template(
            "result.html",
            price=predicted_price,
            low=low,
            high=high,
            input_data=input_data,
        )
    except FileNotFoundError as exc:
        return render_template("result.html", error=str(exc), price=None)
    except Exception as exc:
        return render_template("result.html", error=f"Prediction error: {exc}", price=None)


@app.route("/performance")
def performance():
    metrics = []
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as handle:
            metrics = json.load(handle)
    return render_template("performance.html", metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        from src.predict import predict_price

        data = request.get_json(force=True) or {}
        price = predict_price(data)
        return jsonify({"success": True, "predicted_price_lakhs": price})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
