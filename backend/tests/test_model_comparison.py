"""Pytest suite for ml/model_comparison.py.

Fast configuration throughout: tiny grids and small tree counts via the
override parameters (test convenience, not tuning). The architecture
tests assert that Phase 6 utilities are reused as the SAME objects, so
split-methodology reuse is proven, not assumed.

Run (from the backend/ directory):
    pytest tests/test_model_comparison.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import build_features
from ml import model_comparison as mc
from ml import train_model as tm
from ml.model_comparison import (
    COMPARISON_COLUMNS,
    ComparisonResult,
    MAX_TUNING_COMBINATIONS,
    compare_models,
    save_comparison_artifacts,
    select_winner,
    tune_xgboost,
)
from ml.train_model import TrainingError

FAST = dict(
    xgb_param_grid={"n_estimators": [15], "max_depth": [3]},   # 1 combo
    rf_params={"n_estimators": 10, "n_jobs": 1},
    hgb_params={"max_iter": 15},
    tscv_splits=2,
)


def featured_df(n_days: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for pid in ("A", "B"):
        dates = pd.date_range("2024-01-01", periods=n_days)
        frames.append(pd.DataFrame({
            "date": dates, "product_id": pid, "product_name": pid,
            "category": "FOODS",
            "sales_quantity": rng.poisson(10, n_days),
            "price": 2.5, "snap_day": (dates.day <= 10).astype(int),
            "holiday": 0, "event_name": "none", "store_id": "S1",
            "is_gap_fill": 0, "outlier_flag": 0,
        }))
    features, _ = build_features(pd.concat(frames, ignore_index=True))
    return features


@pytest.fixture(scope="module")
def result() -> ComparisonResult:
    """One shared comparison run (module-scoped for speed)."""
    return compare_models(featured_df(), **FAST)


# --------------------- Architecture: Phase 6 reuse ------------------------

def test_split_function_is_reused_not_reimplemented():
    assert mc.chronological_split is tm.chronological_split
    assert mc.evaluate is tm.evaluate
    assert mc.select_features is tm.select_features
    assert mc.validate_training_input is tm.validate_training_input


def test_split_reuse_is_chronological(result):
    b = result.split_boundaries
    assert set(b) == {"train", "validation", "test"}
    # Boundaries are formatted "start .. end"; order must be temporal.
    train_end = b["train"].split(" .. ")[1]
    val_start, val_end = b["validation"].split(" .. ")
    test_start = b["test"].split(" .. ")[0]
    assert train_end < val_start <= val_end < test_start


# --------------------- Comparison table -----------------------------------

def test_comparison_table_generation(result):
    table = result.table
    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert set(table["model"]) == {"XGBoost", "RandomForest",
                                   "HistGradientBoosting"}
    assert len(table) == 3


def test_training_times_nonnegative(result):
    assert (result.table["training_time_seconds"] >= 0).all()


def test_metrics_reuse_and_known_values():
    # evaluate() is Phase 6's function (identity asserted above); its
    # correctness on known values still holds through the reuse.
    perfect = mc.evaluate(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert perfect == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


# --------------------- Winner selection -----------------------------------

def test_select_winner_highest_validation_r2():
    table = pd.DataFrame({
        "model": ["A", "B", "C"],
        "validation_r2": [0.5, 0.9, 0.7],
        "test_r2": [0.99, 0.10, 0.50],   # must be ignored
    })
    assert select_winner(table) == "B"


def test_select_winner_empty_table_rejected():
    with pytest.raises(TrainingError, match="empty"):
        select_winner(pd.DataFrame(columns=["model", "validation_r2"]))


def test_result_winner_matches_table(result):
    best = result.table.loc[result.table["validation_r2"].idxmax(), "model"]
    assert result.winner_name == best
    assert result.winner_model is not None


# --------------------- Predictions ----------------------------------------

def test_prediction_length_and_columns(result):
    preds = result.test_predictions
    assert len(preds) == result.split_sizes["test"]
    assert list(preds.columns) == ["date", "product_id", "actual",
                                   "predicted"]


def test_nan_warmup_rows_dropped_uniformly(result):
    # 2 series, deepest warm-up = 28 (lag_28 / rolling_28) -> 56 rows.
    assert result.dropped_nan_rows == 56
    total = sum(result.split_sizes.values())
    assert total == 2 * 90 - 56


# --------------------- Tuning guard ---------------------------------------

def test_tuning_grid_size_capped():
    df = featured_df()
    oversized = {"n_estimators": [10, 20, 30, 40, 50],
                 "max_depth": [2, 3, 4, 5, 6]}   # 25 > 20
    with pytest.raises(TrainingError, match="maximum"):
        tune_xgboost(df[["lag_1"]].fillna(0), df["sales_quantity"],
                     param_grid=oversized)
    assert len(oversized["n_estimators"]) * len(oversized["max_depth"]) \
        > MAX_TUNING_COMBINATIONS


# --------------------- Artifacts ------------------------------------------

def test_best_model_saved_with_bundle(result, tmp_path):
    import joblib
    paths = save_comparison_artifacts(result, tmp_path)
    assert paths["best_model"].name == "best_model.joblib"
    bundle = joblib.load(paths["best_model"])
    assert bundle["model_type"] == result.winner_name
    assert bundle["feature_names"] == result.feature_names
    assert set(bundle["metrics"]) == {"validation", "test"}

    table = pd.read_csv(paths["comparison"])
    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert len(table) == 3


def test_best_params_saved_only_if_xgboost_wins(result, tmp_path):
    # Force an XGBoost win.
    forced = ComparisonResult(
        table=result.table, winner_name="XGBoost",
        winner_model=result.winner_model,
        winner_params={"n_estimators": 15, "max_depth": 3},
        feature_names=result.feature_names,
        split_sizes=result.split_sizes,
        split_boundaries=result.split_boundaries,
        dropped_nan_rows=result.dropped_nan_rows,
        test_predictions=result.test_predictions,
    )
    paths = save_comparison_artifacts(forced, tmp_path / "xgb")
    assert "best_params" in paths and paths["best_params"].exists()
    saved = json.loads(paths["best_params"].read_text())
    assert saved["max_depth"] == 3

    # Force a non-XGBoost win: file must NOT be written.
    forced_rf = ComparisonResult(
        table=result.table, winner_name="RandomForest",
        winner_model=result.winner_model, winner_params=None,
        feature_names=result.feature_names,
        split_sizes=result.split_sizes,
        split_boundaries=result.split_boundaries,
        dropped_nan_rows=result.dropped_nan_rows,
        test_predictions=result.test_predictions,
    )
    paths_rf = save_comparison_artifacts(forced_rf, tmp_path / "rf")
    assert "best_params" not in paths_rf
    assert not (tmp_path / "rf" / "best_params.json").exists()


# --------------------- Validation guards (reused) --------------------------

def test_empty_dataframe_rejected():
    with pytest.raises(TrainingError, match="empty"):
        compare_models(pd.DataFrame(), **FAST)


def test_duplicate_rows_rejected():
    df = featured_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    with pytest.raises(TrainingError, match="duplicate"):
        compare_models(df, **FAST)


def test_unsorted_dates_rejected():
    df = featured_df().sample(frac=1, random_state=1).reset_index(drop=True)
    with pytest.raises(TrainingError, match="not sorted"):
        compare_models(df, **FAST)


def test_missing_target_and_features_rejected():
    with pytest.raises(TrainingError, match="target"):
        compare_models(featured_df().drop(columns=["sales_quantity"]),
                       **FAST)
    with pytest.raises(TrainingError, match="engineered features"):
        compare_models(featured_df().drop(columns=["lag_1", "lag_7"]),
                       **FAST)
