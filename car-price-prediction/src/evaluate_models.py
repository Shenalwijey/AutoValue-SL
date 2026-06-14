"""
evaluate_models.py
------------------
Generates all evaluation charts and the overfitting/underfitting report.
Saves charts to static/images/ for the Flask web app.

Run:
    python -m src.evaluate_models
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score
import joblib

from src.data_preprocessing import (
    load_data, clean_column_names, clean_data,
    engineer_features, select_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.utils import load_json

# ── Plot style ──────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR  = os.path.join(PROJECT_ROOT, "static", "images")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "car_price_dataset.csv")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_JOBS       = 1

os.makedirs(IMAGES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [CHART] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# EDA CHARTS
# ─────────────────────────────────────────────

def plot_target_distribution(y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("Price Distribution (Raw)", fontweight="bold")
    axes[0].set_xlabel("Price (LKR Lakhs)")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(y), bins=50, color="#4CAF50", edgecolor="white", alpha=0.85)
    axes[1].set_title("Price Distribution (Log Scale)", fontweight="bold")
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Target Variable – Car Price", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "target_distribution.png")


def plot_correlation_heatmap(df):
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + ["price"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.4, annot_kws={"size": 8})
    ax.set_title("Correlation Matrix – Numeric Features", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def plot_feature_vs_price(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    pairs = [
        ("car_age",   "Car Age (years)"),
        ("mileage_km","Mileage (KM)"),
        ("engine_cc", "Engine CC"),
        ("year",      "Year of Manufacture"),
        ("log_mileage","Log(Mileage)"),
        ("amenity_score","Amenity Score"),
    ]
    for ax, (col, label) in zip(axes.flat, pairs):
        if col in df.columns:
            ax.scatter(df[col], df["price"], alpha=0.2, s=8, color="#1976D2")
            ax.set_xlabel(label); ax.set_ylabel("Price (Lakhs)")
            ax.set_title(f"{label} vs Price", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "feature_vs_price.png")


def plot_brand_price(df):
    top_brands = df["brand"].value_counts().head(12).index
    sub = df[df["brand"].isin(top_brands)]

    fig, ax = plt.subplots(figsize=(14, 6))
    order = sub.groupby("brand")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=sub, x="brand", y="price", order=order,
                palette="Blues_d", ax=ax, fliersize=2)
    ax.set_xlabel("Brand"); ax.set_ylabel("Price (LKR Lakhs)")
    ax.set_title("Price Distribution by Brand (Top 12)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _save(fig, "brand_price.png")


def plot_fuel_gear_price(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df, x="fuel_type", y="price",
                palette="Set2", ax=axes[0], fliersize=2)
    axes[0].set_title("Price by Fuel Type", fontweight="bold")
    axes[0].set_xlabel("Fuel Type"); axes[0].set_ylabel("Price (Lakhs)")

    sns.boxplot(data=df, x="gear", y="price",
                palette="Set3", ax=axes[1], fliersize=2)
    axes[1].set_title("Price by Gear Type", fontweight="bold")
    axes[1].set_xlabel("Gear"); axes[1].set_ylabel("Price (Lakhs)")

    fig.tight_layout()
    return _save(fig, "fuel_gear_price.png")


# ─────────────────────────────────────────────
# MODEL EVALUATION CHARTS
# ─────────────────────────────────────────────

def plot_model_comparison(metrics_list):
    df = pd.DataFrame(metrics_list)
    df = df.sort_values("test_r2", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # R² comparison
    x = range(len(df))
    w = 0.35
    axes[0].bar([i - w/2 for i in x], df["train_r2"], width=w,
                label="Train R²", color="#1976D2", alpha=0.85)
    axes[0].bar([i + w/2 for i in x], df["test_r2"], width=w,
                label="Test R²",  color="#43A047", alpha=0.85)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(df["model_name"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("Train vs Test R² (All Models)", fontweight="bold")
    axes[0].legend(); axes[0].set_ylim(0, 1.1)

    # RMSE comparison
    axes[1].barh(df["model_name"], df["test_rmse"], color="#E53935", alpha=0.85)
    axes[1].set_xlabel("Test RMSE (LKR Lakhs)")
    axes[1].set_title("Test RMSE – All Models", fontweight="bold")

    fig.tight_layout()
    return _save(fig, "model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name="Best Model"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Actual vs Predicted scatter
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=12, color="#1976D2")
    lim = max(y_test.max(), y_pred.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Price (Lakhs)")
    axes[0].set_ylabel("Predicted Price (Lakhs)")
    axes[0].set_title(f"Actual vs Predicted – {model_name}", fontweight="bold")
    axes[0].legend()

    # Residual plot
    residuals = np.array(y_test) - np.array(y_pred)
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=12, color="#7B1FA2")
    axes[1].axhline(0, color="red", lw=1.5, ls="--")
    axes[1].set_xlabel("Predicted Price (Lakhs)")
    axes[1].set_ylabel("Residual (Actual – Predicted)")
    axes[1].set_title("Residual Plot", fontweight="bold")

    fig.tight_layout()
    return _save(fig, "actual_vs_predicted.png")


def plot_residual_distribution(y_test, y_pred):
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=60, color="#FF7043", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_xlabel("Residual (LKR Lakhs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "residual_distribution.png")


def plot_feature_importance(pipeline, feature_names):
    """Works for tree-based models that expose feature_importances_."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print("  [SKIP] Model has no feature_importances_")
        return None

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        print("  [SKIP] Feature importance length mismatch")
        return None

    idx  = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([feature_names[i] for i in idx[::-1]],
            importances[idx[::-1]], color="#1565C0", alpha=0.85)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 20 Feature Importances", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "feature_importance.png")


def plot_learning_curve(pipeline, X_train, y_train, model_name="Best Model"):
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=N_JOBS
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-",
            color="#1976D2", label="Training R²")
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1),
                    alpha=0.15, color="#1976D2")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-",
            color="#43A047", label="CV Validation R²")
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1),
                    alpha=0.15, color="#43A047")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R² Score")
    ax.set_title(f"Learning Curve – {model_name}", fontweight="bold")
    ax.legend(); ax.set_ylim(0, 1.1)
    fig.tight_layout()
    return _save(fig, "learning_curve.png")


def plot_overfitting_analysis(metrics_list):
    df = pd.DataFrame(metrics_list)
    df["gap"] = df["train_r2"] - df["test_r2"]
    df = df.sort_values("gap", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#E53935" if g > 0.15 else "#43A047" for g in df["gap"]]
    ax.barh(df["model_name"], df["gap"], color=colors, alpha=0.85)
    ax.axvline(0.15, color="#E53935", ls="--", lw=1.5, label="Overfit threshold (0.15)")
    ax.axvline(0.00, color="black",   ls="-",  lw=0.8)
    ax.set_xlabel("Train R² – Test R² Gap")
    ax.set_title("Overfitting Analysis (Gap = Train R² − Test R²)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "overfitting_analysis.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n[EVALUATE] Generating all evaluation charts...")

    df = load_data(DATA_PATH)
    df = clean_column_names(df)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # EDA charts
    plot_target_distribution(y)
    plot_correlation_heatmap(df)
    plot_feature_vs_price(df)
    plot_brand_price(df)
    plot_fuel_gear_price(df)

    # Load saved model & metrics
    model_path = f"{MODELS_DIR}/best_model.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        y_pred   = pipeline.predict(X_test)

        metrics_list = load_json(os.path.join(MODELS_DIR, "model_metrics.json"))

        plot_model_comparison(metrics_list)
        plot_actual_vs_predicted(y_test.values, y_pred)
        plot_residual_distribution(y_test.values, y_pred)
        plot_overfitting_analysis(metrics_list)

        all_features = (
            [f"num_{c}" for c in ["year","engine_cc","mileage_km","car_age",
                                   "mileage_per_year","amenity_score",
                                   "log_mileage","log_engine_cc","log_car_age",
                                   "listing_year","listing_month"]]
            + [f"bin_{c}" for c in ["has_ac","has_ps","has_pm","has_pw",
                                    "is_leasing","is_new","is_auto","is_luxury"]]
            + [f"cat_{c}" for c in ["brand","fuel_type","town"]]
        )
        # Use simpler names matching feature count
        feature_names = (
            ["year","engine_cc","mileage_km","car_age","mileage_per_year",
             "amenity_score","log_mileage","log_engine_cc","log_car_age",
             "listing_year","listing_month",
             "has_ac","has_ps","has_pm","has_pw","is_leasing","is_new","is_auto","is_luxury",
             "brand","fuel_type","town"]
        )
        plot_feature_importance(pipeline, feature_names)
        plot_learning_curve(pipeline, X_train, y_train)

        print(f"\n[EVALUATE] ✅ All charts saved to {IMAGES_DIR}/")
    else:
        print(f"[WARN] No model found at {model_path}. Run train_models.py first.")

        # Still generate EDA charts
        print("[EVALUATE] EDA charts generated.")


if __name__ == "__main__":
    main()
