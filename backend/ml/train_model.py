"""Baseline demand-forecasting training pipeline for DemandAI.

Reusable, pure functions (no printing) that:
    - validate the feature dataset,
    - select model features automatically (numeric, non-leaky),
    - split chronologically by DATE (never shuffled),
    - train an XGBoost regressor with fixed sensible defaults
      (no hyperparameter tuning in this phase),
    - evaluate MAE / RMSE / R^2 on validation and test,
    - produce ranked feature importance and test predictions,
    - save all artifacts with joblib / CSV.

Leakage rules enforced here:
    - Chronological split: train = oldest dates, validation = middle,
      test = newest. Split boundaries are DATES, so no calendar day can
      appear in two splits.
    - outlier_flag is EXCLUDED from features: it was derived from
      today's sales via full-series statistics (Phase 3), so using it
      would leak the target.
    - Warm-up NaNs from lag/rolling features are passed through
      unchanged; XGBoost handles missing values natively.

xgboost is imported lazily inside training so the module's validation
and split utilities work in environments without it.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET: str = "sales_quantity"

# Excluded from features, by reason:
#   spec exclusions ......... date, sales_quantity, product_name, event_name
#   non-numeric identifiers . product_id, category, store_id (no encoding
#                             in this phase)
#   audit metadata .......... is_gap_fill
#   TARGET LEAKAGE .......... outlier_flag (function of today's sales and
#                             of full-series statistics)
EXCLUDED_FEATURE_COLUMNS: tuple[str, ...] = (
    "date", TARGET, "product_name", "event_name",
    "product_id", "category", "store_id",
    "is_gap_fill", "outlier_flag",
)

# Presence of these proves Phase 5 ran on this dataset.
REQUIRED_ENGINEERED_FEATURES: tuple[str, ...] = (
    "lag_1", "lag_7", "lag_28", "rolling_mean_7",
)

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


class TrainingError(Exception):
    """Raised when the training input or configuration is unusable."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_training_input(df: pd.DataFrame) -> None:
    """Reject datasets on which baseline training would be wrong.

    Raises TrainingError for: empty input, missing/non-numeric target,
    missing date column, duplicate (product, date) rows, dates unsorted
    within a product series, and missing engineered features (Phase 5
    was not run).
    """
    if df is None or df.empty:
        raise TrainingError("Training dataset is empty.")
    if "date" not in df.columns:
        raise TrainingError("Missing 'date' column.")
    if TARGET not in df.columns:
        raise TrainingError(f"Missing target column '{TARGET}'.")
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        raise TrainingError(f"Target '{TARGET}' must be numeric.")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TrainingError(
            "date column must be datetime (load with parse_dates=['date'])."
        )

    keys = [c for c in ("product_id", "store_id") if c in df.columns]
    dup_subset = keys + ["date"] if keys else None
    dup_mask = df.duplicated(subset=dup_subset)
    if dup_mask.any():
        raise TrainingError(
            f"{int(dup_mask.sum()):,} duplicate rows detected"
            + (f" on {dup_subset}." if dup_subset else ".")
        )

    if keys:
        monotonic = df.groupby(keys, sort=False)["date"].apply(
            lambda s: s.is_monotonic_increasing
        )
        if not monotonic.all():
            raise TrainingError(
                "Dates are not sorted ascending within product series."
            )

    missing = [c for c in REQUIRED_ENGINEERED_FEATURES if c not in df.columns]
    if missing:
        raise TrainingError(
            f"Missing engineered features: {missing}. Run Phase 5 "
            "(scripts/run_feature_engineering.py) first."
        )


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame) -> list[str]:
    """All numeric columns except exclusions (see EXCLUDED_FEATURE_COLUMNS).

    Automatic by design: new engineered features are picked up without
    code changes; identifiers, the target, and leaky metadata are not.
    """
    features = [
        c for c in df.columns
        if c not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not features:
        raise TrainingError("No usable numeric feature columns found.")
    return features


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Split by DATE: train = oldest, validation = middle, test = newest.

    Boundaries are computed on the sorted list of UNIQUE dates, so every
    calendar day belongs to exactly one split (a day can never straddle
    two splits, which a row-count split could cause). No shuffling.

    Returns (train, val, test, boundaries) where boundaries maps each
    split to its inclusive date range (ISO strings).
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 \
            or train_frac + val_frac >= 1:
        raise TrainingError(
            f"Invalid split fractions: train={train_frac}, val={val_frac} "
            "(need train+val < 1)."
        )
    dates = np.sort(df["date"].unique())
    n = len(dates)
    train_end = dates[int(n * train_frac) - 1]
    val_end = dates[int(n * (train_frac + val_frac)) - 1]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    if train.empty or val.empty or test.empty:
        raise TrainingError(
            "A split is empty -- dataset has too few distinct dates for "
            f"fractions ({train_frac}, {val_frac})."
        )
    boundaries = {
        "train": f"{pd.Timestamp(dates[0]).date()} .. "
                 f"{pd.Timestamp(train_end).date()}",
        "validation": f"{val['date'].min().date()} .. "
                      f"{pd.Timestamp(val_end).date()}",
        "test": f"{test['date'].min().date()} .. "
                f"{pd.Timestamp(dates[-1]).date()}",
    }
    return train, val, test, boundaries


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R^2 as a JSON-friendly dict."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Everything produced by one baseline training run."""

    model: Any
    feature_names: list[str]
    params: dict[str, Any]
    split_sizes: dict[str, int]
    split_boundaries: dict[str, str]
    metrics: dict[str, dict[str, float]]
    importance: pd.DataFrame
    test_predictions: pd.DataFrame
    train_seconds: float
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def train_baseline(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    params: dict[str, Any] | None = None,
    model_factory: Callable[..., Any] | None = None,
) -> TrainingResult:
    """Train the XGBoost baseline on a chronological split.

    Args:
        df: Phase 5 feature dataset (datetime date column).
        train_frac / val_frac: Chronological split fractions
            (test gets the remainder, the NEWEST dates).
        params: Overrides merged onto DEFAULT_XGB_PARAMS (used by tests
            to shrink n_estimators; NOT a tuning mechanism).
        model_factory: Regressor constructor; defaults to
            xgboost.XGBRegressor (imported lazily).

    Returns:
        TrainingResult with the fitted model, metrics, ranked feature
        importance, and test predictions (date, product_id, actual,
        predicted).
    """
    validate_training_input(df)
    features = select_features(df)
    train, val, test, boundaries = chronological_split(
        df, train_frac=train_frac, val_frac=val_frac
    )

    merged_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    if model_factory is None:
        from xgboost import XGBRegressor  # lazy: keep module importable
        model_factory = XGBRegressor
    model = model_factory(**merged_params)

    start = time.perf_counter()
    model.fit(train[features], train[TARGET])
    train_seconds = round(time.perf_counter() - start, 2)

    metrics = {
        "validation": evaluate(val[TARGET].to_numpy(),
                               model.predict(val[features])),
        "test": evaluate(test[TARGET].to_numpy(),
                         model.predict(test[features])),
    }

    importance = (
        pd.DataFrame({
            "feature": features,
            "importance": np.asarray(model.feature_importances_,
                                     dtype=float),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    test_pred = pd.DataFrame({
        "date": test["date"].values,
        "product_id": (test["product_id"].values
                       if "product_id" in test.columns else ""),
        "actual": test[TARGET].values,
        "predicted": np.round(model.predict(test[features]), 3),
    })

    return TrainingResult(
        model=model,
        feature_names=features,
        params=merged_params,
        split_sizes={"train": len(train), "validation": len(val),
                     "test": len(test)},
        split_boundaries=boundaries,
        metrics=metrics,
        importance=importance,
        test_predictions=test_pred,
        train_seconds=train_seconds,
    )


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(result: TrainingResult,
                   models_dir: Path) -> dict[str, Path]:
    """Persist model bundle, feature importance, and test predictions.

    The joblib file is a BUNDLE (model + feature names + params +
    metrics + timestamp): a model saved without its feature list is a
    deployment accident waiting to happen.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": models_dir / "baseline_model.joblib",
        "importance": models_dir / "feature_importance.csv",
        "predictions": models_dir / "test_predictions.csv",
    }
    joblib.dump(
        {
            "model": result.model,
            "feature_names": result.feature_names,
            "params": result.params,
            "metrics": result.metrics,
            "target": TARGET,
            "trained_at": result.trained_at,
            "model_type": "xgboost_baseline",
        },
        paths["model"],
    )
    result.importance.to_csv(paths["importance"], index=False)
    result.test_predictions.to_csv(paths["predictions"], index=False)
    return paths
