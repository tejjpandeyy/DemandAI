"""Pytest suite for ml/train_model.py.

Uses the real Phase 5 feature pipeline to build small fixtures, then
verifies split chronology, leakage guards, training, metrics, and
artifact persistence. Training tests shrink n_estimators via the params
override for speed (this is a test convenience, not tuning).

Run (from the backend/ directory):
    pytest tests/test_training.py -v
"""

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import build_features
from ml.train_model import (
    EXCLUDED_FEATURE_COLUMNS,
    TrainingError,
    chronological_split,
    evaluate,
    save_artifacts,
    select_features,
    train_baseline,
    validate_training_input,
)

FAST_PARAMS = {"n_estimators": 30, "max_depth": 3, "n_jobs": 1}


def featured_df(n_days: int = 90, products: tuple[str, ...] = ("A", "B")
                ) -> pd.DataFrame:
    """Small featured dataset built with the REAL Phase 5 pipeline."""
    rng = np.random.default_rng(0)
    frames = []
    for pid in products:
        dates = pd.date_range("2024-01-01", periods=n_days)
        frames.append(pd.DataFrame({
            "date": dates,
            "product_id": pid,
            "product_name": pid,
            "category": "FOODS",
            "sales_quantity": rng.poisson(10, n_days),
            "price": 2.5,
            "snap_day": (dates.day <= 10).astype(int),
            "holiday": 0,
            "event_name": "none",
            "store_id": "S1",
            "is_gap_fill": 0,
            "outlier_flag": 0,
        }))
    raw = pd.concat(frames, ignore_index=True)
    features, _ = build_features(raw)
    return features


# ------------------------- Chronological split ----------------------------

def test_chronological_split_order_and_disjoint():
    df = featured_df()
    train, val, test, bounds = chronological_split(df)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    # No calendar day in two splits.
    assert not (set(train["date"]) & set(val["date"]))
    assert not (set(val["date"]) & set(test["date"]))
    assert set(bounds) == {"train", "validation", "test"}


def test_split_proportions_approximate():
    df = featured_df(n_days=100)
    train, val, test, _ = chronological_split(df, 0.70, 0.15)
    n_dates = df["date"].nunique()
    assert train["date"].nunique() == pytest.approx(0.70 * n_dates, abs=2)
    assert val["date"].nunique() == pytest.approx(0.15 * n_dates, abs=2)
    assert test["date"].nunique() == pytest.approx(0.15 * n_dates, abs=2)


def test_no_shuffle_train_is_exactly_the_oldest_dates():
    df = featured_df(n_days=100)
    train, _, test, _ = chronological_split(df)
    # Normalize both sides to pd.DatetimeIndex: pd.Timestamp and
    # np.datetime64 do not hash equally, so raw set() comparison is
    # dtype-fragile across pandas versions.
    all_dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    n_train = train["date"].nunique()
    expected_train = set(all_dates[:n_train])
    actual_train = set(pd.DatetimeIndex(train["date"].unique()))
    assert actual_train == expected_train
    assert pd.Timestamp(test["date"].min()) > pd.Timestamp(
        all_dates[int(100 * 0.85) - 1])


def test_invalid_split_fractions_rejected():
    df = featured_df(n_days=60)
    with pytest.raises(TrainingError, match="split fractions"):
        chronological_split(df, 0.8, 0.3)


# ------------------------- Feature selection ------------------------------

def test_feature_selection_excludes_leaky_and_nonnumeric():
    features = select_features(featured_df())
    for banned in EXCLUDED_FEATURE_COLUMNS:
        assert banned not in features
    # The leakage-critical exclusion, asserted explicitly:
    assert "outlier_flag" not in features
    assert "lag_1" in features and "rolling_mean_7" in features
    assert "is_weekend" in features and "price_change" in features


# ------------------------- Metrics ----------------------------------------

def test_metrics_calculation_known_values():
    perfect = evaluate(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert perfect["mae"] == 0.0
    assert perfect["rmse"] == 0.0
    assert perfect["r2"] == 1.0
    off_by_one = evaluate(np.array([1.0, 2.0, 3.0]),
                          np.array([2.0, 3.0, 4.0]))
    assert off_by_one["mae"] == pytest.approx(1.0)
    assert off_by_one["rmse"] == pytest.approx(1.0)


# ------------------------- Training ---------------------------------------

def test_model_trains_and_prediction_length():
    result = train_baseline(featured_df(), params=FAST_PARAMS)
    assert result.model is not None
    assert len(result.test_predictions) == result.split_sizes["test"]
    assert list(result.test_predictions.columns) == [
        "date", "product_id", "actual", "predicted"
    ]
    assert result.train_seconds >= 0
    assert set(result.metrics) == {"validation", "test"}
    for split_metrics in result.metrics.values():
        assert set(split_metrics) == {"mae", "rmse", "r2"}


def test_importance_ranked_and_complete():
    result = train_baseline(featured_df(), params=FAST_PARAMS)
    imp = result.importance
    assert set(imp["feature"]) == set(result.feature_names)
    assert (imp["importance"].values[:-1]
            >= imp["importance"].values[1:]).all()   # descending


# ------------------------- Artifacts --------------------------------------

def test_artifacts_saved(tmp_path):
    import joblib
    result = train_baseline(featured_df(), params=FAST_PARAMS)
    paths = save_artifacts(result, tmp_path)

    assert paths["model"].name == "baseline_model.joblib"
    assert paths["model"].exists()
    bundle = joblib.load(paths["model"])
    assert bundle["feature_names"] == result.feature_names
    assert bundle["target"] == "sales_quantity"
    assert "metrics" in bundle and "trained_at" in bundle

    assert paths["importance"].name == "feature_importance.csv"
    imp = pd.read_csv(paths["importance"])
    assert list(imp.columns) == ["feature", "importance"]
    assert len(imp) == len(result.feature_names)

    assert paths["predictions"].name == "test_predictions.csv"
    preds = pd.read_csv(paths["predictions"])
    assert list(preds.columns) == ["date", "product_id", "actual",
                                   "predicted"]
    assert len(preds) == result.split_sizes["test"]


# ------------------------- Validation guards ------------------------------

def test_empty_dataframe_rejected():
    with pytest.raises(TrainingError, match="empty"):
        validate_training_input(pd.DataFrame())


def test_missing_target_rejected():
    df = featured_df().drop(columns=["sales_quantity"])
    with pytest.raises(TrainingError, match="target"):
        validate_training_input(df)


def test_duplicate_rows_rejected():
    df = featured_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    with pytest.raises(TrainingError, match="duplicate"):
        validate_training_input(df)


def test_unsorted_dates_rejected():
    df = featured_df().sample(frac=1, random_state=1).reset_index(drop=True)
    with pytest.raises(TrainingError, match="not sorted"):
        validate_training_input(df)


def test_missing_engineered_features_rejected():
    df = featured_df().drop(columns=["lag_1", "lag_7"])
    with pytest.raises(TrainingError, match="engineered features"):
        validate_training_input(df)
