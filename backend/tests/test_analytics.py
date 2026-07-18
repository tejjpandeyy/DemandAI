"""Pytest suite for app/services/analytics.py.

Fixtures are tiny and hand-computable so every assertion checks an
exactly known value, not a vague property.

Run (from the backend/ directory):
    pytest tests/test_analytics.py -v
"""

import pandas as pd
import pytest

from app.services import analytics


def make_df() -> pd.DataFrame:
    """Two products, two categories, 4 days (Mon 2024-01-01 .. Thu).

    Hand-computable totals:
        P1 (FOODS):     10, 0, 20, 10  -> total 40, one zero day
        P2 (HOUSEHOLD):  5, 5,  0, 10  -> total 20, one zero day
    Daily totals: 15, 5, 20, 20 -> grand total 60 over 4 dates.
    """
    return pd.DataFrame({
        "date": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 2
        ),
        "product_id": ["P1"] * 4 + ["P2"] * 4,
        "product_name": ["P1"] * 4 + ["P2"] * 4,
        "category": ["FOODS"] * 4 + ["HOUSEHOLD"] * 4,
        "sales_quantity": [10, 0, 20, 10, 5, 5, 0, 10],
        "price": [2.0, 2.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0],
        "snap_day": [1, 0, 0, 0] * 2,
        "holiday": [0, 0, 1, 0] * 2,
        "event_name": ["none", "none", "SuperBowl", "none"] * 2,
        "store_id": ["S1"] * 8,
        "is_gap_fill": [0] * 8,
        "outlier_flag": [0, 0, 1, 0, 0, 0, 0, 0],
    })


# ------------------------------ Guards ------------------------------------

def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        analytics.dataset_summary(pd.DataFrame())


def test_missing_required_column_raises():
    df = make_df().drop(columns=["product_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        analytics.daily_sales(df)


def test_gap_fill_rows_are_excluded_from_statistics():
    df = make_df()
    structural = df.iloc[[0]].copy()
    structural["is_gap_fill"] = 1
    structural["sales_quantity"] = 999   # must never be counted
    df = pd.concat([df, structural], ignore_index=True)
    summary = analytics.dataset_summary(df)
    assert summary["total_units_sold"] == 60
    assert summary["total_rows"] == 8


# ------------------------------ Summary -----------------------------------

def test_summary_calculations():
    summary = analytics.dataset_summary(make_df())
    assert summary["total_rows"] == 8
    assert summary["total_products"] == 2
    assert summary["total_units_sold"] == 60
    assert summary["date_start"] == "2024-01-01"
    assert summary["date_end"] == "2024-01-04"
    assert summary["avg_daily_demand"] == pytest.approx(15.0)  # 60 / 4
    assert summary["zero_sales_pct"] == pytest.approx(25.0)    # 2 of 8
    assert summary["outlier_flag_count"] == 1


# ------------------------------ Trends ------------------------------------

def test_daily_aggregation_and_descriptive_ma():
    daily = analytics.daily_sales(make_df(), moving_averages=(2,))
    assert daily["total_units"].tolist() == [15, 5, 20, 20]
    # Trailing MA: first value undefined, then (15+5)/2, (5+20)/2, ...
    assert pd.isna(daily["ma_2"].iloc[0])
    assert daily["ma_2"].iloc[1] == pytest.approx(10.0)
    assert daily["ma_2"].iloc[3] == pytest.approx(20.0)


def test_weekly_and_monthly_aggregation():
    weekly = analytics.weekly_sales(make_df())
    monthly = analytics.monthly_sales(make_df())
    # All four dates (Mon-Thu) fall in one ISO week and one month.
    assert len(weekly) == 1 and weekly["total_units"].iloc[0] == 60
    assert len(monthly) == 1 and monthly["total_units"].iloc[0] == 60
    assert monthly["month_start"].iloc[0] == pd.Timestamp("2024-01-01")


# ------------------------------ Products ----------------------------------

def test_top_product_ordering():
    top = analytics.top_products(make_df(), n=2)
    assert top["product_id"].tolist() == ["P1", "P2"]   # 40 > 20
    lowest = analytics.top_products(make_df(), n=1, lowest=True)
    assert lowest["product_id"].tolist() == ["P2"]


def test_zero_sales_pct_per_product():
    metrics = analytics.product_metrics(make_df()).set_index("product_id")
    assert metrics.loc["P1", "zero_sales_pct"] == pytest.approx(25.0)
    assert metrics.loc["P2", "zero_sales_pct"] == pytest.approx(25.0)
    assert metrics.loc["P1", "total_units"] == 40
    assert metrics.loc["P1", "outlier_count"] == 1


# ------------------------------ Events ------------------------------------

def test_event_baseline_calculations():
    events = analytics.event_analysis(make_df()).set_index("event")
    # holiday rows: sales 20, 0 -> avg 10; non-holiday avg = 40/6.
    holiday = events.loc["holiday"]
    assert holiday["observations"] == 2
    assert holiday["avg_units"] == pytest.approx(10.0)
    expected_pct = (10.0 / (40 / 6) - 1) * 100
    assert holiday["pct_vs_baseline"] == pytest.approx(expected_pct, rel=1e-3)
    # Named event baseline uses "none" days: avg = 40/6 as well.
    superbowl = events.loc["SuperBowl"]
    assert superbowl["event_type"] == "named_event"
    assert superbowl["observations"] == 2
    assert superbowl["baseline_avg_units"] == pytest.approx(40 / 6, rel=1e-3)


# ------------------------------ Outliers ----------------------------------

def test_outlier_counts_and_event_share():
    result = analytics.outlier_analysis(make_df())
    assert result["total_flagged"] == 1
    assert result["by_product"] == {"P1": 1}
    assert result["on_holiday_days"] == 1      # the flag sits on a holiday
    assert result["on_snap_day_days"] == 0
    assert result["by_event_name"] == {"SuperBowl": 1}
    assert result["pct_on_event_days"] == pytest.approx(100.0)


def test_outlier_analysis_requires_flag_column():
    df = make_df().drop(columns=["outlier_flag"])
    with pytest.raises(ValueError, match="outlier_flag"):
        analytics.outlier_analysis(df)


# ------------------------------ Price -------------------------------------

def test_price_analysis_counts_changes_and_guards_small_samples():
    per_product, summary = analytics.price_analysis(make_df(), min_obs=30)
    per_product = per_product.set_index("product_id")
    assert per_product.loc["P1", "n_price_changes"] == 1   # 2.0 -> 4.0
    assert per_product.loc["P2", "n_price_changes"] == 0
    # Only 4 obs each < min_obs=30 -> correlation must be refused (NaN).
    assert per_product["within_product_corr"].isna().all()
    assert summary["products_with_computable_within_corr"] == 0
    assert "confounded" in summary["note"]


# ------------------------------ Day of week -------------------------------

def test_day_of_week_ordering_and_values():
    dow = analytics.day_of_week_analysis(make_df())
    assert dow["day_of_week"].tolist() == list(analytics.DAY_ORDER)
    dow = dow.set_index("day_of_week")
    assert dow.loc["Monday", "total_units"] == 15
    assert dow.loc["Thursday", "avg_daily_total_units"] == pytest.approx(20.0)
    assert dow.loc["Sunday", "n_dates"] == 0   # no Sunday in fixture


# ------------------------------ Categories --------------------------------

def test_multi_category_support_without_fabrication():
    cats = analytics.category_analysis(make_df()).set_index("category")
    assert set(cats.index) == {"FOODS", "HOUSEHOLD"}
    assert cats.loc["FOODS", "total_units"] == 40
    assert cats.loc["FOODS", "units_share_pct"] == pytest.approx(66.67,
                                                                 rel=1e-3)
    # Single-category input yields a single row -- nothing invented.
    single = analytics.category_analysis(
        make_df().assign(category="FOODS")
    )
    assert len(single) == 1
