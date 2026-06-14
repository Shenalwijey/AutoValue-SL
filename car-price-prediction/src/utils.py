"""
utils.py
--------
Shared utility helpers used across the project.
"""

import numpy as np
import json
import os


def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    """
    Compute MAE, MSE, RMSE, R2, MAPE for regression.

    Returns a dict and optionally prints a summary.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    # MAPE – guard against division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        "MAE"  : round(float(mae),  4),
        "MSE"  : round(float(mse),  4),
        "RMSE" : round(float(rmse), 4),
        "R2"   : round(float(r2),   4),
        "MAPE" : round(float(mape), 4),
    }
    if label:
        print(f"  [{label}] R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return metrics


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[INFO] Saved → {path}")


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def overfitting_status(train_r2: float, test_r2: float) -> str:
    """
    Return a human-readable note about over/underfitting based on R² gap.
    """
    gap = train_r2 - test_r2
    if train_r2 < 0.70:
        return "⚠️  UNDERFITTING – training R² too low; model too simple"
    elif gap > 0.15:
        return f"⚠️  OVERFITTING  – gap={gap:.2f}; model memorises training data"
    elif gap < 0:
        return "ℹ️  SUSPICIOUS   – test > train; possible data distribution issue"
    else:
        return f"✅  GOOD FIT     – gap={gap:.2f}; generalises well"
