"""
train_models.py
---------------
Trains and evaluates 11 regression models on the car price dataset.
Saves the best model and preprocessor to disk.

Run this file directly:
    python -m src.train_models

Features:
  - 11 diverse regressors
  - 5-fold cross-validation for every model
  - Train/test comparison (overfitting detection)
  - Hyperparameter tuning for top candidates
  - Best model selection by test R²
  - Saves model_metrics.json, best_model.pkl, preprocessor.pkl
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    load_data, clean_column_names, clean_data,
    engineer_features, select_features,
)
from src.feature_engineering import build_preprocessor, save_preprocessor
from src.utils import regression_metrics, save_json, overfitting_status

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "car_price_dataset.csv")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
N_JOBS       = 1

os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_base_models():
    """Return dict of {name: estimator} for initial comparison."""
    return {
        "Linear Regression"  : LinearRegression(),
        "Ridge"              : Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso"              : Lasso(alpha=0.1, max_iter=5000, random_state=RANDOM_STATE),
        "ElasticNet"         : ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree"      : DecisionTreeRegressor(
                                   max_depth=10, min_samples_split=10,
                                   min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random Forest"      : RandomForestRegressor(
                                   n_estimators=200, max_depth=15,
                                   min_samples_split=5, min_samples_leaf=3,
                                   max_features=0.7, random_state=RANDOM_STATE,
                                   n_jobs=N_JOBS),
        "Extra Trees"        : ExtraTreesRegressor(
                                   n_estimators=200, max_depth=15,
                                   min_samples_split=5, min_samples_leaf=3,
                                   random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "Gradient Boosting"  : GradientBoostingRegressor(
                                   n_estimators=200, learning_rate=0.05,
                                   max_depth=5, subsample=0.8,
                                   random_state=RANDOM_STATE),
        "AdaBoost"           : AdaBoostRegressor(
                                   n_estimators=100, learning_rate=0.05,
                                   random_state=RANDOM_STATE),
        "KNN"                : KNeighborsRegressor(n_neighbors=7, n_jobs=N_JOBS),
        "SVR"                : SVR(C=10, gamma="scale", kernel="rbf"),
        # Note: XGBoost/LightGBM/CatBoost unavailable in this environment
        # Gradient Boosting above is the scikit-learn equivalent
    }


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train all base models, collect metrics, detect over/underfitting.
    Returns a list of result dicts.
    """
    models   = get_base_models()
    results  = []

    print("\n" + "="*70)
    print("  BASE MODEL TRAINING & EVALUATION")
    print("="*70)

    for name, estimator in models.items():
        print(f"\n▶ {name}")
        t0 = time.time()

        # Build full pipeline: preprocessor + model
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        estimator),
        ])

        # --- Cross-validation (on training data) ---
        cv_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=CV_FOLDS, scoring="r2", n_jobs=N_JOBS
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std  = float(np.std(cv_scores))

        # --- Fit on full training set ---
        pipe.fit(X_train, y_train)

        # --- Evaluate ---
        train_pred = pipe.predict(X_train)
        test_pred  = pipe.predict(X_test)

        train_metrics = regression_metrics(y_train.values, train_pred, f"{name} TRAIN")
        test_metrics  = regression_metrics(y_test.values,  test_pred,  f"{name} TEST")

        status = overfitting_status(train_metrics["R2"], test_metrics["R2"])
        elapsed = time.time() - t0

        print(f"  CV R² = {cv_mean:.4f} ± {cv_std:.4f}  |  {status}  |  {elapsed:.1f}s")

        results.append({
            "model_name"   : name,
            "train_r2"     : train_metrics["R2"],
            "test_r2"      : test_metrics["R2"],
            "cv_r2_mean"   : round(cv_mean, 4),
            "cv_r2_std"    : round(cv_std, 4),
            "train_mae"    : train_metrics["MAE"],
            "test_mae"     : test_metrics["MAE"],
            "train_rmse"   : train_metrics["RMSE"],
            "test_rmse"    : test_metrics["RMSE"],
            "test_mape"    : test_metrics["MAPE"],
            "fit_status"   : status,
            "pipeline"     : pipe,
        })

    return results


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING (top 3 models)
# ─────────────────────────────────────────────

def tune_models(results, X_train, y_train, preprocessor):
    """
    Tune the top 3 models by test R² using RandomizedSearchCV.
    Returns updated results with tuned pipelines.
    """
    # Sort by test_r2, pick top 3
    sorted_results = sorted(results, key=lambda x: x["test_r2"], reverse=True)
    top3 = sorted_results[:3]

    print("\n" + "="*70)
    print("  HYPERPARAMETER TUNING (Top 3 Models)")
    print("="*70)

    param_grids = {
        "Random Forest": {
            "model__n_estimators"   : [100, 200, 300],
            "model__max_depth"      : [10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf" : [1, 3, 5],
            "model__max_features"   : [0.5, 0.7, "sqrt"],
        },
        "Extra Trees": {
            "model__n_estimators"   : [100, 200, 300],
            "model__max_depth"      : [10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf" : [1, 3, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators"   : [100, 200, 300],
            "model__learning_rate"  : [0.02, 0.05, 0.1],
            "model__max_depth"      : [3, 4, 5, 6],
            "model__subsample"      : [0.7, 0.8, 0.9, 1.0],
            "model__min_samples_leaf": [3, 5, 10],
        },
        "Decision Tree": {
            "model__max_depth"       : [5, 8, 10, 12, 15],
            "model__min_samples_split": [5, 10, 20],
            "model__min_samples_leaf" : [3, 5, 10],
        },
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "ElasticNet": {
            "model__alpha"   : [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights"    : ["uniform", "distance"],
        },
        "SVR": {
            "model__C"     : [1, 10, 50, 100],
            "model__gamma" : ["scale", "auto"],
            "model__kernel": ["rbf", "poly"],
        },
        "AdaBoost": {
            "model__n_estimators" : [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5],
        },
    }

    tuned_results = []

    for entry in top3:
        name = entry["model_name"]
        print(f"\n▶ Tuning: {name}")

        if name not in param_grids:
            print(f"  (No tuning grid for {name}, keeping as-is)")
            tuned_results.append(entry)
            continue

        # Rebuild a fresh pipeline for tuning
        base_model = get_base_models()[name]
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        base_model),
        ])

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=20,
            cv=CV_FOLDS,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        best_cv   = search.best_score_

        print(f"  Best CV R²: {best_cv:.4f}")
        print(f"  Best params: {search.best_params_}")

        # Re-evaluate best pipe on test set (passed in as closure variable)
        entry["tuned_pipeline"] = best_pipe
        entry["tuned_cv_r2"]    = round(float(best_cv), 4)
        entry["best_params"]    = {k: str(v) for k, v in search.best_params_.items()}
        tuned_results.append(entry)

    return tuned_results


# ─────────────────────────────────────────────
# SELECT & SAVE BEST MODEL
# ─────────────────────────────────────────────

def select_best_model(results, X_test, y_test):
    """
    Choose the best model by test R² from tuned pipelines (or base pipelines).
    Also considers generalization gap: penalise models where gap > 0.15.
    """
    print("\n" + "="*70)
    print("  MODEL SELECTION")
    print("="*70)

    best_score = -999
    best_entry = None

    for entry in results:
        # Prefer tuned pipeline if available
        pipe  = entry.get("tuned_pipeline", entry["pipeline"])
        preds = pipe.predict(X_test)

        from sklearn.metrics import r2_score
        test_r2 = r2_score(y_test, preds)
        gap     = entry["train_r2"] - test_r2

        # Penalise heavy overfitters: subtract half the gap
        adjusted = test_r2 - max(0, (gap - 0.10) * 0.5)

        entry["final_test_r2"]  = round(float(test_r2), 4)
        entry["adjusted_score"] = round(float(adjusted), 4)

        print(f"  {entry['model_name']:22s}  test_R²={test_r2:.4f}  "
              f"gap={gap:.4f}  adjusted={adjusted:.4f}")

        if adjusted > best_score:
            best_score = adjusted
            best_entry = entry

    print(f"\n🏆 Best model selected: {best_entry['model_name']} "
          f"(adjusted score = {best_score:.4f})")
    return best_entry


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  CAR PRICE PREDICTION – TRAINING PIPELINE")
    print("="*70)

    # 1. Load & preprocess data
    df = load_data(DATA_PATH)
    df = clean_column_names(df)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = select_features(df)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {X_train.shape}  Test: {X_test.shape}")

    # 3. Build preprocessor (ordinal encoder – works for all models)
    preprocessor = build_preprocessor(encoder="ordinal")

    # 4. Train all base models
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)

    # 5. Tune top 3
    results = tune_models(results, X_train, y_train, preprocessor)

    # 6. Select best
    best = select_best_model(results, X_test, y_test)

    # 7. Fit chosen pipeline on FULL training data & save
    final_pipe = best.get("tuned_pipeline", best["pipeline"])
    final_pipe.fit(X_train, y_train)          # already fitted, but re-confirms

    joblib.dump(final_pipe, f"{MODELS_DIR}/best_model.pkl")
    print(f"[INFO] Best model saved → {MODELS_DIR}/best_model.pkl")

    # Also save the standalone preprocessor
    fitted_preprocessor = final_pipe.named_steps["preprocessor"]
    save_preprocessor(fitted_preprocessor)

    # 8. Save all metrics to JSON (exclude non-serialisable pipeline objects)
    metrics_out = []
    for r in results:
        rec = {k: v for k, v in r.items()
               if k not in ("pipeline", "tuned_pipeline")}
        metrics_out.append(rec)
    save_json(metrics_out, os.path.join(MODELS_DIR, "model_metrics.json"))

    print("\n✅ Training complete. All artefacts saved to /models/")
    return results, best


if __name__ == "__main__":
    main()
