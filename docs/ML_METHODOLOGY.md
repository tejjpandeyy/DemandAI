# DemandAI — ML Methodology & Results

This document describes how the forecasting model was built and evaluated, with an emphasis on the choices that make the reported metrics trustworthy.

## 1. Problem framing

DemandAI predicts `sales_quantity` — the daily units sold of a product at a store — as a **supervised regression** task. Each row is one product-store-day; the target is that day's demand, and the features summarize everything knowable *before* that day.

## 2. Dataset

- **Source:** M5 Forecasting – Accuracy (real Walmart daily sales).
- **Subset:** store `CA_1`, category `FOODS`, top 50 products by total units.
- **Shape after preparation:** ~92k rows spanning 2011-01-29 to 2016-05-22.
- **Schema:** `date, product_id, product_name, category, sales_quantity, price, snap_day, holiday, event_name, store_id`.

The raw wide-format daily columns (`d_1 … d_1941`) are reshaped into a tidy long table. Rows before a product's first observed price (its pre-launch period in M5) are dropped, because their zero sales do not represent real demand.

## 3. Preprocessing principles

The preprocessing layer never manufactures observations:

- **Date gaps** are detected and reported, not silently filled with synthetic zero-demand rows.
- **Price repair** uses forward fill within each product-store series (past information only); a price from the future is never copied backward.
- **Demand outliers** are *flagged* (an `outlier_flag` column) but never deleted or capped — event-driven spikes are exactly what a demand model must learn.
- **Zero-sales days** are kept: for the top-50 FOODS subset, ~12.6% of days have zero sales, and these are legitimate observations.

## 4. Feature categories

| Category | Features | Why it matters |
|---|---|---|
| **Lag** | `lag_1`, `lag_7`, `lag_14`, `lag_28` | Recent demand is the strongest predictor of near-future demand; weekly lags capture day-of-week seasonality. |
| **Rolling statistics** | `rolling_mean/std/min/max_{7,14,28}`, `rolling_median_{7,28}` | Summarize the recent level and volatility of demand. |
| **Expanding** | `expanding_mean` | A product's long-run average demand to date. |
| **Calendar** | `year, month, quarter, week_of_year, day_of_month, day_of_week, day_of_year` | Encode seasonality and trend deterministically. |
| **Weekend** | `is_weekend` | FOODS demand shifts on weekends. |
| **SNAP** | `snap_day` | U.S. food-assistance disbursement days measurably lift FOODS demand; kept distinct from marketing promotions, which this dataset does not contain. |
| **Holiday / event** | `holiday`, `has_named_event` | Different events move demand in opposite directions (e.g. Super Bowl up, Christmas down as stores close), so the raw event identity is preserved upstream. |
| **Price** | `price_change`, `price_pct_change`, `rolling_price_mean_7`, `rolling_price_std_7` | Capture markdowns and price volatility using only past prices. |

## 5. Data leakage prevention

Leakage — letting the model see information unavailable at prediction time — is the single biggest threat to honest forecasting metrics. Three safeguards:

1. **Chronological split.** Train = oldest ~70% of dates, validation = middle ~15%, test = newest ~15%. Boundaries are computed on unique dates, so no calendar day appears in two splits, and the model never trains on dates after those it is evaluated on.

2. **Shifted rolling/expanding windows.** Every rolling and expanding statistic is computed on `sales_quantity.shift(1)` within each product series. The window for day *t* therefore ends at *t−1* and cannot contain the value being predicted. (Verified by a test: a large spike on day *t* leaves day *t*'s rolling max unchanged.)

3. **Past-only price repair.** Forward fill only. Warm-up rows where a full window of history does not yet exist are left as `NaN` rather than imputed.

## 6. Why chronological (not random) splitting

A random `train_test_split` shuffles rows, so future dates leak into training and the model is evaluated on dates it effectively already saw. Reported metrics then look great and collapse in production. Chronological splitting reproduces the real task — fit on the past, predict the future — and yields honest, if more modest, numbers.

## 7. Models compared

All three were trained on the identical chronological split, with identical feature columns and the identical set of rows (warm-up `NaN` rows were dropped uniformly so every model saw the same data).

- **XGBoost** — gradient-boosted trees; the only model tuned, via `GridSearchCV` with `TimeSeriesSplit` (no shuffling), a grid capped at 20 combinations, fitted on the training split only.
- **RandomForest** — bagged trees, fixed sensible defaults.
- **HistGradientBoosting** — histogram-based gradient boosting, fixed defaults, native `NaN` handling.

## 8. Winner selection

The winner is chosen by **highest validation R²**. **The test set is never consulted during selection** — doing so would turn it into a second validation set and inflate the final reported metrics. The test set is scored exactly once, after the winner is fixed.

| Model | Validation R² | Test R² |
|---|---|---|
| XGBoost (tuned) | 0.7232 | 0.6764 |
| RandomForest | 0.7251 | 0.6844 |
| **HistGradientBoosting** | **0.7294** | **0.7041** |

HistGradientBoosting had the best validation R² and also generalized best to the untouched test set — a reassuring sign that the validation-based choice was not overfit to the validation window.

## 9. Final model performance

**HistGradientBoosting**, evaluated once on the held-out newest ~15% of dates:

| Metric | Value | Meaning |
|---|---|---|
| R² | 0.7041 | Explains ~70% of demand variance on unseen future data. |
| MAE | 4.9338 | Average absolute error ~4.9 units/day. |
| RMSE | 7.6485 | Root-mean-square error; larger than MAE because it penalizes big misses (event spikes). |

For reference, the Phase-6 single-model baseline scored R² 0.6931 / MAE 4.9151 / RMSE 7.8407. The comparison stage improved test R² and RMSE.

## 10. A note on MAPE

MAPE (mean absolute percentage error) is deliberately not used as a headline metric here: with ~12.6% zero-sales days, percentage error is undefined or explodes on those rows. MAE and RMSE are stable in the presence of zeros and are the honest choices for intermittent demand.

## 11. Honest caveats

These results are for a **single store and one category**. They should not be read as a general claim about the model's accuracy on other stores, categories, or businesses, all of which would require re-training and re-validation.
