# AutoValue-SL

# 🚗 AutoValue SL — Intelligent Car Price Prediction for Sri Lanka

> A production-quality Machine Learning web application that predicts used car prices in the Sri Lankan market using 11 regression models, engineered features, and a Flask web interface.

---

## 📊 Project Highlights

| Metric | Value |
|--------|-------|
| Raw Listings | 9,788 |
| After Cleaning | 8,914 |
| Engineered Features | 22 |
| Models Trained & Compared | 11 |
| Best Model | Gradient Boosting Regressor |
| Test R² | **0.874** |
| Train-Test Gap | 0.043 (well-generalised) |
| Deployment | Flask Web App + REST API |

---

## ⚙️ How It Works — Pipeline

```
Raw CSV (9,788 listings)
    → Data Cleaning        (remove duplicates, outliers, range violations)
    → Feature Engineering  (16 raw → 22 features, log transforms, binary flags)
    → Train/Test Split     (80% / 20%, random_state=42)
    → Preprocessing        (StandardScaler + OrdinalEncoder via ColumnTransformer)
    → Model Training       (11 models × 5-fold Cross Validation)
    → Hyperparameter Tuning (RandomizedSearchCV on top 3 models)
    → Model Selection      (Adjusted R² penalises train-test gap > 10%)
    → Flask Deployment     (Web app + JSON REST API)
```

---

## 🧠 Models Compared

| Rank | Model | Test R² | Gap | Status |
|------|-------|---------|-----|--------|
| 1 | **Gradient Boosting** | **0.874** | 0.043 | ✅ BEST |
| 2 | Random Forest | 0.871 | 0.062 | ✅ Good |
| 3 | Extra Trees | 0.856 | 0.113 | ⚠️ Overfit |
| 4 | Decision Tree | 0.846 | 0.056 | ✅ OK |
| 5 | AdaBoost | 0.698 | 0.000 | ⚠️ Underfit |
| 6 | Linear Regression | 0.680 | — | ⚠️ Underfit |
| 7 | Ridge | 0.670 | — | ⚠️ Underfit |
| 8 | KNN | 0.655 | 0.098 | ⚠️ Overfit |
| 9 | Lasso | 0.645 | — | ⚠️ Underfit |
| 10 | ElasticNet | 0.570 | — | ❌ Underfit |
| 11 | SVR | 0.490 | — | ❌ Underfit |

---

## 🏆 Best Model — Gradient Boosting Regressor

**Optimal Hyperparameters** (tuned via RandomizedSearchCV, 5-fold CV):

```
n_estimators     = 150
learning_rate    = 0.05
max_depth        = 5
subsample        = 0.8
min_samples_leaf = 3
random_state     = 42
```

**Why it won:** Highest Test R² (0.874), lowest generalisation gap (0.043) among top models, with built-in regularisation via low learning rate and subsampling.

---

## 🔧 Feature Engineering

**Derived Numeric:** `car_age`, `mileage_per_year`, `log_mileage`, `log_engine_cc`, `log_car_age`, `listing_year`, `listing_month`

**Binary Flags:** `has_ac`, `has_ps`, `has_pm`, `has_pw`, `amenity_score`, `is_leasing`, `is_new`, `is_auto`, `is_luxury`

**Categorical:** `brand` (50 unique makes), `fuel_type`, `town` — encoded via `OrdinalEncoder(handle_unknown=-1)`

**Top Feature Importances:**
1. `brand` — 35%
2. `car_age` — 12%
3. `mileage_km` — 10%
4. `log_mileage` — 8%
5. `year` — 7%

---

## 🌐 Flask Web Application

| Route | Description |
|-------|-------------|
| `/` | Home — hero section & how-it-works |
| `/predict` | Prediction form with validation |
| `/result` | Animated price counter & range gauge |
| `/performance` | Model charts & full metrics table |
| `/about` | Dataset info & tech stack |
| `/api/predict` | JSON REST API for programmatic access |

**Design:** Dark navy theme · Electric blue accents · Syne + DM Sans fonts · Fully responsive (Bootstrap 5)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AutoValue-SL.git
cd AutoValue-SL
```

### 2. Install dependencies
```bash
pip install -r car-price-prediction/requirements.txt
```

### 3. Train the model *(optional — pre-trained model included)*
```bash
python app.py --train
```

### 4. Run the web app
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python 3.11 · Flask · joblib |
| ML | Scikit-learn · Pandas · NumPy |
| Visualisation | Matplotlib · Seaborn · Plotly |
| Frontend | Bootstrap 5 · JavaScript · CSS3 |

---

## 📈 Key Findings

- 🏷️ **Brand is the #1 price driver** (~35% importance) — brand status carries a premium beyond objective metrics in the Sri Lankan market.
- 📅 **Age + Mileage account for ~22%** of importance combined.
- 📉 **Linear models are insufficient** — Linear/Ridge R² ≈ 0.67 vs Gradient Boosting 0.874, confirming non-linear pricing.
- ⚡ **Hybrid/Electric premium** — 15–25% higher prices over equivalent petrol vehicles.
- 🏙️ **Colombo urban premium** — listings in Colombo average higher than rural districts.

---

## 🔮 Future Work

- 🚀 XGBoost / LightGBM / CatBoost integration
- 🔍 SHAP explainability for per-prediction feature attribution
- 📸 Image-based condition valuation via CNN
- 📱 Mobile app (React Native) consuming the JSON API
- ☁️ Cloud deployment (AWS / Heroku)

---

## 👥 Group Members

 Shenal - [@Shenalwijey](https://github.com/Shenalwijey)  
 Tharindu - [@jananjaya2003](https://github.com/jananjaya2003)  
 Malshi <br>
 Hansika  
---

## 📄 License

This project is for academic purposes. All data collected from publicly available Sri Lankan car listing platforms.




