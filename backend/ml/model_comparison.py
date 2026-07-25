"""Multi-model comparison for DemandAI (Phase 7).

Compares XGBoost (tuned), RandomForest, and HistGradientBoosting on the
SAME chronological split methodology as Phase 6 -- the split, feature
selection, validation, and metric functions are IMPORTED from
ml.train_model, not reimplemented, so the leakage-safe methodology is
reused by construction.

Tuning policy (per spec):
    - ONLY XGBoost is tuned.
    - GridSearchCV with TimeSeriesSplit (no shuffling anywhere), fitted
      on the TRAIN split only; validation and test stay untouched until
      final evaluation.
    - The grid is capped at MAX_TUNING_COMBINATIONS (20).

NaN policy:
    RandomForest cannot ingest NaN, so rows with NaN in any selected
    feature (the Phase 5 warm-up rows: first lag/rolling window of each
    series) are dropped UNIFORMLY for all models before splitting. The
    count is recorded -- a fair comparison requires identical rows.

Winner: highest validation R^2. Test metrics are reported but never
used for selection (choosing on test would turn test into a second
validation set).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit

# Reused Phase 6 utilities -- the identical split/validation/metrics code.
from ml.train_model import (
    DEFAULT_XGB_PARAMS,
    TARGET,
    TrainingError,
    chronological_split,
    evaluate,
    select_features,
    validate_training_input,
)

MAX_TUNING_COMBINATIONS: int = 20

DEFAULT_XGB_GRID: dict[str, list] = {
    "n_estimators": [150, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}  # 2 * 3 * 2 = 12 combinations <= 20

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 16,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_HGB_PARAMS: dict[str, Any] = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "random_state": 42,
}

COMPARISON_COLUMNS: tuple[str, ...] = (
    "model",
    "validation_mae", "validation_rmse", "validation_r2",
    "test_mae", "test_rmse", "test_r2",
    "training_time_seconds",
)


@dataclass
class ComparisonResult:
    """Everything produced by one model-comparison run."""

    table: pd.DataFrame
    winner_name: str
    winner_model: Any
    winner_params: dict[str, Any] | None   # tuned params iff XGBoost won
    feature_names: list[str]
    split_sizes: dict[str, int]
    split_boundaries: dict[str, str]
    dropped_nan_rows: int
    test_predictions: pd.DataFrame
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# XGBoost tuning (TimeSeriesSplit, train split only)
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list] | None = None,
    base_params: dict[str, Any] | None = None,
    n_splits: int = 3,
    model_factory: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any], int]:
    """Grid-search XGBoost with TimeSeriesSplit on the TRAIN data only.

    TimeSeriesSplit preserves temporal order inside every CV fold
    (train on earlier, validate on later); nothing is shuffled. The
    grid size is hard-capped at MAX_TUNING_COMBINATIONS.

    Returns (best_estimator, best_params, n_combinations_evaluated).
    """
    param_grid = param_grid if param_grid is not None else DEFAULT_XGB_GRID
    n_combinations = len(ParameterGrid(param_grid))
    if n_combinations > MAX_TUNING_COMBINATIONS:
        raise TrainingError(
            f"Tuning grid has {n_combinations} combinations; maximum "
            f"allowed is {MAX_TUNING_COMBINATIONS}."
        )
    if model_factory is None:
        from xgboost import XGBRegressor  # lazy: module stays importable
        model_factory = XGBRegressor

    # Base params minus anything the grid controls.
    base = {**DEFAULT_XGB_PARAMS, **(base_params or {})}
    base = {k: v for k, v in base.items() if k not in param_grid}

    search = GridSearchCV(
        estimator=model_factory(**base),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_params = {**base, **search.best_params_}
    return search.best_estimator_, best_params, n_combinations


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _drop_feature_nan_rows(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, int]:
    """Drop rows with NaN in any selected feature, uniformly for all models.

    Required because RandomForest cannot ingest NaN; applied identically
    to every model so they are compared on the SAME rows. These are the
    Phase 5 warm-up rows (insufficient history), located at the start of
    each series -- well inside the train period for realistic splits.
    """
    mask = df[features].notna().all(axis=1) & df[TARGET].notna()
    return df[mask], int((~mask).sum())


def select_winner(table: pd.DataFrame) -> str:
    """Winner = highest validation R^2 (test is never used for selection)."""
    if table.empty:
        raise TrainingError("Comparison table is empty.")
    return str(table.loc[table["validation_r2"].idxmax(), "model"])


def compare_models(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    xgb_param_grid: dict[str, list] | None = None,
    xgb_base_params: dict[str, Any] | None = None,
    rf_params: dict[str, Any] | None = None,
    hgb_params: dict[str, Any] | None = None,
    tscv_splits: int = 3,
    xgb_factory: Callable[..., Any] | None = None,
) -> ComparisonResult:
    """Train and compare XGBoost (tuned), RandomForest, and HistGB.

    Reuses Phase 6 validation, feature selection, chronological split,
    and metrics. Per-model ``training_time_seconds`` is wall time to
    produce that model (for XGBoost this INCLUDES tuning).

    Returns a ComparisonResult; the winner's predictions on the test
    split are included as (date, product_id, actual, predicted).
    """
    validate_training_input(df)
    features = select_features(df)
    data, dropped = _drop_feature_nan_rows(df, features)
    train, val, test, boundaries = chronological_split(
        data, train_frac=train_frac, val_frac=val_frac
    )
    X_train, y_train = train[features], train[TARGET]

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    xgb_best_params: dict[str, Any] | None = None

    # --- 1. XGBoost (the ONLY tuned model) ---
    start = time.perf_counter()
    xgb_model, xgb_best_params, _ = tune_xgboost(
        X_train, y_train,
        param_grid=xgb_param_grid,
        base_params=xgb_base_params,
        n_splits=tscv_splits,
        model_factory=xgb_factory,
    )
    fitted["XGBoost"] = xgb_model
    xgb_seconds = time.perf_counter() - start

    # --- 2 & 3. Fixed-parameter models (NOT tuned, per spec) ---
    untuned = {
        "RandomForest": RandomForestRegressor(
            **{**DEFAULT_RF_PARAMS, **(rf_params or {})}
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            **{**DEFAULT_HGB_PARAMS, **(hgb_params or {})}
        ),
    }
    timings = {"XGBoost": xgb_seconds}
    for name, model in untuned.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        timings[name] = time.perf_counter() - start
        fitted[name] = model

    for name, model in fitted.items():
        val_metrics = evaluate(val[TARGET].to_numpy(),
                               model.predict(val[features]))
        test_metrics = evaluate(test[TARGET].to_numpy(),
                                model.predict(test[features]))
        rows.append({
            "model": name,
            "validation_mae": val_metrics["mae"],
            "validation_rmse": val_metrics["rmse"],
            "validation_r2": val_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "training_time_seconds": round(timings[name], 2),
        })

    table = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    winner_name = select_winner(table)
    winner_model = fitted[winner_name]

    test_predictions = pd.DataFrame({
        "date": test["date"].values,
        "product_id": (test["product_id"].values
                       if "product_id" in test.columns else ""),
        "actual": test[TARGET].values,
        "predicted": winner_model.predict(test[features]).round(3),
    })

    return ComparisonResult(
        table=table,
        winner_name=winner_name,
        winner_model=winner_model,
        winner_params=xgb_best_params if winner_name == "XGBoost" else None,
        feature_names=features,
        split_sizes={"train": len(train), "validation": len(val),
                     "test": len(test)},
        split_boundaries=boundaries,
        dropped_nan_rows=dropped,
        test_predictions=test_predictions,
    )


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_comparison_artifacts(
    result: ComparisonResult, models_dir: Path
) -> dict[str, Path]:
    """Persist the comparison table, best-model bundle, and (iff XGBoost
    won) the tuned best parameters as JSON."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "comparison": models_dir / "model_comparison.csv",
        "best_model": models_dir / "best_model.joblib",
    }
    result.table.to_csv(paths["comparison"], index=False)

    winner_row = result.table.set_index("model").loc[result.winner_name]
    joblib.dump(
        {
            "model": result.winner_model,
            "model_type": result.winner_name,
            "feature_names": result.feature_names,
            "params": result.winner_params,
            "metrics": {
                "validation": {
                    "mae": float(winner_row["validation_mae"]),
                    "rmse": float(winner_row["validation_rmse"]),
                    "r2": float(winner_row["validation_r2"]),
                },
                "test": {
                    "mae": float(winner_row["test_mae"]),
                    "rmse": float(winner_row["test_rmse"]),
                    "r2": float(winner_row["test_r2"]),
                },
            },
            "target": TARGET,
            "trained_at": result.trained_at,
        },
        paths["best_model"],
    )

    if result.winner_name == "XGBoost" and result.winner_params is not None:
        paths["best_params"] = models_dir / "best_params.json"
        paths["best_params"].write_text(
            json.dumps(result.winner_params, indent=2, default=str)
        )
    return paths
