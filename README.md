# Taxi Demand Forecasting – Sweet Lift Airport Service

## Project Overview

This project builds a machine learning model to **forecast the number of taxi orders for the next hour** at airports for the company **Sweet Lift Taxi**.

Accurate forecasts help the company bring **enough drivers online during peak hours**, reducing missed orders and idle time.

- **Task type:** Time series regression
- **Target:** `num_orders` (number of taxi orders per hour)
- **Main metric:** Root Mean Squared Error (**RMSE**)
- **Business requirement:** RMSE on the test set **≤ 48**
- **Best result:** **RMSE ≈ 39.6** on the hold-out test set (Gradient Boosting Regressor)

---

# Dataset

- File: `/datasets/taxi.csv`
- Columns:

  - `datetime` – timestamp (originally at a finer granularity, later resampled)
  - `num_orders` – number of taxi orders during that interval

## Preprocessing

1. Parsed `datetime` as a proper datetime index.
2. Sorted the index to ensure chronological order.
3. **Resampled to hourly frequency** using `.resample('1H').sum()` to obtain `num_orders` per hour.
4. No shuffling was used at any point to avoid data leakage across time.

---

# Problem Statement & Metric

The goal is to:

> Predict the **number of taxi orders in the next hour** using past demand and time-based patterns.

Evaluation:

- **Metric:** RMSE (Root Mean Squared Error)

  - Interpreted in units of “orders per hour”.

- **Target definition:**

  - The last **10% of the time series** is held out as a **test set**.
  - RMSE is computed on this test set for the final model.

- **Success criterion:**

  - RMSE(test) **≤ 48**.

---

# Methodology

## Feature Engineering

From the hourly time series, the following features were created:

- **Calendar features** (from the datetime index):

  - `year`
  - `month`
  - `day`
  - `dayofweek`
  - `hour`

- **Lag features** (autoregressive structure):

  - `lag_1`, `lag_2`, …, `lag_k` (recent past values of `num_orders`)

- **Rolling statistics**:

  - Rolling means over recent windows (e.g. last N hours) to capture local trends and smoothing.

These features turn the time series into a **tabular supervised learning problem**, where each row represents one hour and its target is the demand in that hour.

## Train / Validation / Test Strategy

- The data was split **chronologically**:

  - First **90%** → used as **train/validation**.
  - Last **10%** → used as **final test**.

- Within the 90% train/validation segment, **`TimeSeriesSplit`** was used for cross-validation:

  - Respects temporal order (no shuffling).
  - Simulates “train on past → validate on future” multiple times.

This avoids data leakage and gives a realistic estimate of performance on future data.

---

# Baseline Models

To understand the difficulty of the problem, two naive baselines were implemented:

1. **Naive last-value forecast**

   - Prediction: demand at time _t_ = demand at time _t–1_ (`lag_1`).

2. **Moving average of the last 24 hours**

   - Prediction: mean of the previous 24 hourly values.

Both baselines achieved **RMSE ≈ 58–59** on the test set, and are clearly outperformed by the machine learning models.

---

# Machine Learning Models

The following models were trained and compared:

1. **Linear Regression**
2. **Ridge Regression**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **XGBoost Regressor** (CPU, `tree_method='hist'`)

## Hyperparameter Tuning

- Tuning used **`GridSearchCV` + `TimeSeriesSplit`** on the train/validation segment.
- Scoring metric: `neg_root_mean_squared_error` (negative RMSE), so **lower RMSE = better**.
- Compact but expressive “pro-light” grids were used for:

  - Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, etc.).
  - Gradient Boosting (`n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample`, `max_features`).
  - XGBoost (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`).

The best models from each grid search were then evaluated on the untouched test set.

---

# Results

**Test RMSE (lower is better):**

- **Gradient Boosting Regressor** → **≈ 39.56** ✅ (best model)
- **XGBoost Regressor** → ≈ 41.22
- **Random Forest Regressor** → ≈ 43.31
- **Ridge Regression** → ≈ 46.97
- **Linear Regression** → ≈ 47.19
- **Baselines (naive / moving average)** → ≈ 58–59

Key points:

- All ML models beat the baselines by a large margin.
- Gradient Boosting achieves the **lowest test RMSE (≈ 39.6)** and **comfortably satisfies** the requirement RMSE ≤ 48.
- Random Forest tends to overfit more strongly (very low training RMSE vs higher test RMSE).
- Boosting methods (Gradient Boosting and XGBoost) reach the best balance between bias and variance.

## Final Model

The **final selected model** is:

> **Gradient Boosting Regressor**, trained on the engineered time-series features and tuned via `GridSearchCV` + `TimeSeriesSplit`.

This model is recommended for deployment to forecast taxi demand one hour ahead.

---

# Business Interpretation

- The model can forecast hourly demand with an average error of about **40 orders per hour**.
- This enables Sweet Lift Taxi to:

  - **Plan driver supply** in advance for peak hours.
  - Identify hours with systematically high or low demand.
  - Reduce both missed orders (too few drivers) and idle time (too many drivers).

In a real system, the model would be integrated into a pipeline that:

1. Continuously ingests recent orders,
2. Updates features (lags, rolling statistics, calendar features),
3. Produces forecasts for the next hour (or multiple hours ahead),
4. Feeds these predictions into operational tools for driver scheduling and incentive planning.

---

# Repository Structure (suggested)

```text
.
├── data/
│   └── taxi.csv                 # original dataset
├── notebooks/
│   └── taxi_demand_forecasting.ipynb   # main analysis & modeling notebook
├── README.md
└── requirements.txt             # project dependencies
```

---

# How to Run

1. **Clone the repository**

```bash
git clone <your-repo-url>.git
cd <your-repo-name>
```

2. **Create and activate a virtual environment (optional but recommended)**

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# or
.\.venv\Scripts\activate       # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Place the dataset**

- Ensure `taxi.csv` (or symlink) is available at `/datasets/taxi.csv`
  or adjust the path in the notebook accordingly.

5. **Run the notebook**

- Open `notebooks/taxi_demand_forecasting.ipynb` in Jupyter / VS Code and run all cells.

---

# Technologies Used

- **Python** (pandas, NumPy)
- **Visualization:** matplotlib / seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Model selection:** GridSearchCV, TimeSeriesSplit
