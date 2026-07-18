"""Pytest suite for ml/feature_engineering.py.

Hand-computable fixtures throughout: every assertion checks an exactly
known value. The suite's centerpiece is leakage prevention -- proving
that no feature at day t can see the sales value of day t or later.

Run (from the backend/ directory):
    pytest tests/test_feature_engineering.py -v
"""

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    FeatureEngineeringError,
    add_calendar_features,
    add_expanding_features,
    add_lag_features,
    add_price_features,
    add_rolling_features,
    build_features,
    validate_feature_input,
)


def make_series(sales, product_id="P1", start="2024-01-01",
                price=None) -> pd.DataFrame:
    """One continuous daily product series with known values."""
    n = len(sales)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n),
        "product_id": [product_id] * n,
        "sales_quantity": sales,
        "price": price if price is not None else [2.0] * n,
        "store_id": ["S1"] * n,
        "event_name": ["none"] * n,
        "holiday": [0] * n,
        "snap_day": [0] * n,
    })


# ------------------------------ Calendar ----------------------------------

def test_calendar_extraction_known_date():
    df = make_series([1], start="2024-03-15")   # Friday, Q1, day 75
    out = add_calendar_features(df).iloc[0]
    assert out["year"] == 2024
    assert out["month"] == 3
    assert out["quarter"] == 1
    assert out["day_of_month"] == 15
    assert out["day_of_week"] == 4              # Friday (0 = Monday)
    assert out["day_of_year"] == 75             # 2024 is a leap year
    assert out["week_of_year"] == 11
    assert out["is_weekend"] == 0
    assert out["is_month_start"] == 0
    assert out["is_month_end"] == 0


def test_weekend_detection():
    # 2024-01-05 = Friday, 06 = Saturday, 07 = Sunday, 08 = Monday.
    df = make_series([1, 1, 1, 1], start="2024-01-05")
    out = add_calendar_features(df)
    assert out["is_weekend"].tolist() == [0, 1, 1, 0]


def test_month_boundaries():
    df = make_series([1, 1, 1], start="2024-01-31")  # Jan 31, Feb 1, Feb 2
    out = add_calendar_features(df)
    assert out["is_month_end"].tolist() == [1, 0, 0]
    assert out["is_month_start"].tolist() == [0, 1, 0]
    assert out["month"].tolist() == [1, 2, 2]


# ------------------------------ Lags --------------------------------------

def test_lag_correctness_hand_computed():
    df = make_series([10, 20, 30, 40, 50])
    out = add_lag_features(df, lags=(1, 3))
    assert out["lag_1"].tolist()[1:] == [10, 20, 30, 40]
    assert pd.isna(out["lag_1"].iloc[0])
    assert out["lag_3"].tolist()[3:] == [10, 20]
    assert out["lag_3"].isna().sum() == 3


def test_multi_product_isolation():
    # Product B's first lag must be NaN, never product A's last value.
    df = pd.concat(
        [make_series([100, 200, 300], product_id="A"),
         make_series([7, 8, 9], product_id="B")],
        ignore_index=True,
    )
    out = add_lag_features(df, lags=(1,))
    b_rows = out[out["product_id"] == "B"]
    assert pd.isna(b_rows["lag_1"].iloc[0])
    assert b_rows["lag_1"].tolist()[1:] == [7, 8]


# ------------------------------ Rolling -----------------------------------

def test_rolling_mean_excludes_today():
    # sales = [1, 2, 3, 4, 5]; rolling_mean_3 at index t must average
    # days t-1, t-2, t-3 -- NEVER include day t.
    df = make_series([1, 2, 3, 4, 5])
    out = add_rolling_features(df, windows=(3,), median_windows=())
    assert out["rolling_mean_3"].isna().tolist()[:3] == [True, True, True]
    assert out["rolling_mean_3"].iloc[3] == pytest.approx(2.0)  # (1+2+3)/3
    assert out["rolling_mean_3"].iloc[4] == pytest.approx(3.0)  # (2+3+4)/3


def test_future_leakage_prevention_spike_invisible_today():
    # A huge spike TODAY must not appear in today's rolling stats.
    df = make_series([1, 1, 1, 1000])
    out = add_rolling_features(df, windows=(3,), median_windows=())
    assert out["rolling_mean_3"].iloc[3] == pytest.approx(1.0)
    assert out["rolling_max_3"].iloc[3] == pytest.approx(1.0)
    # The spike only becomes visible the day AFTER (here: no day after,
    # so nothing anywhere reflects 1000).
    assert (out["rolling_max_3"].dropna() < 1000).all()


def test_rolling_min_max_std_alignment():
    df = make_series([10, 20, 30, 40])
    out = add_rolling_features(df, windows=(2,), median_windows=())
    # At index 2 the window is days 0-1: [10, 20].
    assert out["rolling_min_2"].iloc[2] == 10
    assert out["rolling_max_2"].iloc[2] == 20
    assert out["rolling_std_2"].iloc[2] == pytest.approx(
        np.std([10, 20], ddof=1))


def test_rolling_median_windows():
    df = make_series(list(range(1, 11)))
    out = add_rolling_features(df, windows=(), median_windows=(3,))
    # At index 3: median of [1, 2, 3] = 2.
    assert out["rolling_median_3"].iloc[3] == pytest.approx(2.0)


# ------------------------------ Expanding ---------------------------------

def test_expanding_mean_uses_only_previous_days():
    df = make_series([10, 20, 30, 40])
    out = add_expanding_features(df)
    assert pd.isna(out["expanding_mean"].iloc[0])
    assert out["expanding_mean"].iloc[1] == pytest.approx(10.0)
    assert out["expanding_mean"].iloc[2] == pytest.approx(15.0)
    assert out["expanding_mean"].iloc[3] == pytest.approx(20.0)


# ------------------------------ Price -------------------------------------

def test_price_features_hand_computed():
    df = make_series([1, 1, 1, 1], price=[2.0, 2.0, 4.0, 4.0])
    out = add_price_features(df, window=2)
    assert pd.isna(out["price_change"].iloc[0])
    assert out["price_change"].tolist()[1:] == [0.0, 2.0, 0.0]
    assert out["price_pct_change"].iloc[2] == pytest.approx(1.0)  # 2->4
    # Rolling price mean is on SHIFTED price: at index 2 the window is
    # prices of days 0-1 = [2, 2] -> 2.0 (today's 4.0 not included).
    assert out["rolling_price_mean_2"].iloc[2] == pytest.approx(2.0)
    # At index 3: prices of days 1-2 = [2, 4] -> 3.0.
    assert out["rolling_price_mean_2"].iloc[3] == pytest.approx(3.0)


def test_price_features_isolated_per_product():
    df = pd.concat(
        [make_series([1, 1], product_id="A", price=[10.0, 10.0]),
         make_series([1, 1], product_id="B", price=[2.0, 2.0])],
        ignore_index=True,
    )
    out = add_price_features(df)
    b_first = out[out["product_id"] == "B"].iloc[0]
    assert pd.isna(b_first["price_change"])   # not 2.0 - 10.0 = -8


# ------------------------------ NaN preservation --------------------------

def test_nan_preservation_no_fill_no_drop():
    df = pd.concat(
        [make_series(list(range(40)), product_id="A"),
         make_series(list(range(40)), product_id="B")],
        ignore_index=True,
    )
    out, _ = build_features(df, lags=(1, 7), windows=(7,),
                            median_windows=(), price_window=7)
    assert len(out) == len(df)                       # nothing dropped
    assert out["lag_1"].isna().sum() == 2            # 1 per series
    assert out["lag_7"].isna().sum() == 14           # 7 per series
    # shift(1) + full 7-window -> first 7 rows NaN per series.
    assert out["rolling_mean_7"].isna().sum() == 14


# ------------------------------ Validation --------------------------------

def test_duplicate_dates_rejected():
    df = make_series([1, 2, 3])
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    with pytest.raises(FeatureEngineeringError, match="duplicate"):
        validate_feature_input(df)


def test_unsorted_dates_rejected():
    df = make_series([1, 2, 3]).iloc[[2, 0, 1]].reset_index(drop=True)
    with pytest.raises(FeatureEngineeringError, match="not sorted"):
        validate_feature_input(df)


def test_date_gaps_rejected():
    df = make_series([1, 2, 3, 4])
    df = df[df["date"] != "2024-01-03"]   # create a gap
    with pytest.raises(FeatureEngineeringError, match="continuous"):
        validate_feature_input(df)


def test_missing_required_columns_rejected():
    df = make_series([1, 2]).drop(columns=["sales_quantity"])
    with pytest.raises(FeatureEngineeringError, match="Missing required"):
        validate_feature_input(df)


def test_empty_dataframe_rejected():
    with pytest.raises(FeatureEngineeringError, match="empty"):
        validate_feature_input(pd.DataFrame())


def test_negative_and_zero_lags_rejected():
    df = make_series([1, 2, 3])
    with pytest.raises(FeatureEngineeringError, match="positive integers"):
        build_features(df, lags=(-1, 7))
    with pytest.raises(FeatureEngineeringError, match="positive integers"):
        build_features(df, lags=(0,))


# ------------------------------ End to end --------------------------------

def test_build_features_full_pipeline_and_event_flag():
    df = make_series(list(range(30)))
    df.loc[5, "event_name"] = "SuperBowl"
    out, generated = build_features(df)
    assert len(out) == 30
    assert "has_named_event" in generated
    assert out["has_named_event"].sum() == 1
    assert out["has_named_event"].iloc[5] == 1
    # Original columns intact; event_name NOT encoded.
    assert out["event_name"].iloc[5] == "SuperBowl"
    for expected in ("lag_1", "lag_28", "rolling_mean_7", "rolling_std_28",
                     "rolling_median_7", "expanding_mean", "price_change",
                     "is_weekend", "quarter"):
        assert expected in generated
