"""Pytest suite for ml/data_preprocessing.py.

Each test locks in one behavioral guarantee. The central invariant under
test: THE PIPELINE NEVER MANUFACTURES OBSERVATIONS -- no synthetic zero
demand for missing days, no future price copied backward, no outlier
removed or capped.

Run (from the backend/ directory):
    pytest tests/test_data_preprocessing.py -v
"""

import pandas as pd
import pytest

from ml.data_preprocessing import (
    NO_EVENT,
    PreprocessingError,
    flag_demand_outliers,
    preprocess_sales_data,
)


def make_df(**overrides) -> pd.DataFrame:
    """Build a small valid sales frame; override columns per test."""
    base = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "product_id": ["P1"] * 4,
        "product_name": ["P1"] * 4,
        "category": ["FOODS"] * 4,
        "sales_quantity": [5, 0, 7, 6],
        "price": [2.5, 2.5, 2.5, 2.5],
        "snap_day": [0, 1, 0, 0],
        "holiday": [0, 0, 1, 0],
        "event_name": [None, None, "SuperBowl", None],
        "store_id": ["S1"] * 4,
    }
    base.update(overrides)
    return pd.DataFrame(base)


GAPPY_DATES = ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"]
# Jan 3 and Jan 4 are missing -> 2 gap days.


# --------------------------- Tier 1: hard failures ------------------------

def test_missing_required_column_raises():
    df = make_df().drop(columns=["product_id"])
    with pytest.raises(PreprocessingError, match="Missing required columns"):
        preprocess_sales_data(df)


def test_negative_sales_raises():
    df = make_df(sales_quantity=[5, -2, 7, 6])
    with pytest.raises(PreprocessingError, match="negative sales_quantity"):
        preprocess_sales_data(df)


def test_conflicting_duplicates_raise():
    # Same (date, product, store) key, DIFFERENT sales values.
    df = make_df()
    conflict = df.iloc[[0]].assign(sales_quantity=[99])
    df = pd.concat([df, conflict], ignore_index=True)
    with pytest.raises(PreprocessingError, match="CONFLICTING"):
        preprocess_sales_data(df)


def test_too_many_bad_dates_raise():
    df = make_df(date=["bad", "bad", "bad", "2024-01-04"])  # 75% bad
    with pytest.raises(PreprocessingError, match="unparseable dates"):
        preprocess_sales_data(df)


def test_empty_dataframe_raises():
    with pytest.raises(PreprocessingError, match="empty"):
        preprocess_sales_data(pd.DataFrame())


def test_unresolved_leading_price_raises_in_strict_mode():
    # No past price exists for the first two rows; strict mode must refuse.
    df = make_df(price=[None, None, 2.5, 2.5])
    with pytest.raises(PreprocessingError, match="no resolvable price"):
        preprocess_sales_data(df, strict=True)


# ----------------------- Tier 2: safe deterministic cleaning --------------

def test_exact_duplicates_dropped_and_counted():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # exact copy
    clean, report = preprocess_sales_data(df)
    assert report.exact_duplicates_removed == 1
    assert len(clean) == 4


def test_few_bad_dates_dropped_and_counted():
    # Build 100 rows so 1 bad date is below the 5% hard-fail threshold.
    dates = pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d").tolist()
    dates[50] = "not-a-date"
    df = make_df(
        date=dates,
        product_id=["P1"] * 100, product_name=["P1"] * 100,
        category=["FOODS"] * 100, sales_quantity=[5] * 100,
        price=[2.5] * 100, snap_day=[0] * 100, holiday=[0] * 100,
        event_name=[None] * 100, store_id=["S1"] * 100,
    )
    clean, report = preprocess_sales_data(df)
    assert report.invalid_date_rows_dropped == 1
    # The dropped row leaves a 1-day gap: DETECTED, not filled.
    assert report.total_missing_days == 1
    assert report.gap_rows_inserted == 0
    assert len(clean) == 99


def test_invalid_price_forward_filled_from_past_only():
    df = make_df(price=[2.5, -1.0, 0.0, 3.0])
    clean, report = preprocess_sales_data(df)
    assert report.invalid_price_count == 2
    assert report.prices_imputed == 2
    assert report.unresolved_price_count == 0
    # Both invalid days inherit the LAST PAST price (2.5) -- never 3.0,
    # which lies in their future.
    assert clean["price"].tolist() == [2.5, 2.5, 2.5, 3.0]


def test_no_future_price_copied_backward():
    # Leading missing prices have no past information -> must stay missing
    # (non-strict mode), never take the future value 2.5.
    df = make_df(price=[None, None, 2.5, 3.0])
    clean, report = preprocess_sales_data(df)  # strict=False default
    assert pd.isna(clean["price"].iloc[0])
    assert pd.isna(clean["price"].iloc[1])
    assert report.unresolved_price_count == 2
    assert report.prices_imputed == 0
    # ffill still works after the first valid price appears
    assert clean["price"].iloc[2] == 2.5
    assert clean["price"].iloc[3] == 3.0


def test_missing_event_name_becomes_explicit_none():
    clean, _ = preprocess_sales_data(make_df())
    assert (clean["event_name"] == NO_EVENT).sum() == 3
    assert (clean["event_name"] == "SuperBowl").sum() == 1


# ------------------- Date gaps: detect by default, never invent -----------

def test_date_gaps_detected_and_reported_without_insertion():
    df = make_df(date=GAPPY_DATES)
    clean, report = preprocess_sales_data(df)  # default fill_date_gaps=False
    assert report.products_with_date_gaps == 1
    assert report.total_missing_days == 2
    assert report.gap_rows_inserted == 0
    assert any("missing days" in w.lower() for w in report.warnings)


def test_no_rows_inserted_by_default_row_count_unchanged():
    df = make_df(date=GAPPY_DATES)
    clean, report = preprocess_sales_data(df)
    assert len(clean) == len(df) == report.output_rows == 4
    assert (clean["is_gap_fill"] == 0).all()


def test_missing_dates_receive_no_synthetic_zero_sales():
    df = make_df(date=GAPPY_DATES)
    clean, _ = preprocess_sales_data(df)
    # The missing calendar days must simply not exist in the output.
    out_dates = set(clean["date"].dt.strftime("%Y-%m-%d"))
    assert "2024-01-03" not in out_dates
    assert "2024-01-04" not in out_dates
    # And no zero rows appeared beyond the one real observed zero.
    assert (clean["sales_quantity"] == 0).sum() == 1


def test_optin_structural_rows_leave_observations_missing():
    # Opt-in mode may create structural rows, but their observation
    # columns must be MISSING -- never fabricated zeros or event values.
    df = make_df(date=GAPPY_DATES)
    clean, report = preprocess_sales_data(df, fill_date_gaps=True)
    assert report.gap_rows_inserted == 2
    assert len(clean) == 6
    inserted = clean[clean["is_gap_fill"] == 1]
    assert inserted["sales_quantity"].isna().all()
    assert inserted["price"].isna().all()
    assert inserted["snap_day"].isna().all()
    assert inserted["holiday"].isna().all()
    # Identity attributes may be carried (time-invariant facts).
    assert (inserted["product_name"] == "P1").all()


# ------------------------- Tier 3: analysis only --------------------------

def test_real_zero_sales_observations_remain_unchanged():
    clean, report = preprocess_sales_data(make_df())
    zero_rows = clean[clean["sales_quantity"] == 0]
    assert len(zero_rows) == 1
    assert zero_rows["date"].dt.strftime("%Y-%m-%d").iloc[0] == "2024-01-02"
    assert report.zero_sales_pct == pytest.approx(0.25)


def test_outliers_flagged_never_removed_or_capped():
    # 29 normal days + 1 extreme spike.
    dates = pd.date_range("2024-01-01", periods=30).strftime("%Y-%m-%d")
    sales = [5] * 29 + [648]
    df = make_df(
        date=list(dates), product_id=["P1"] * 30, product_name=["P1"] * 30,
        category=["FOODS"] * 30, sales_quantity=sales, price=[2.5] * 30,
        snap_day=[0] * 29 + [1], holiday=[0] * 30,
        event_name=[None] * 30, store_id=["S1"] * 30,
    )
    clean, report = preprocess_sales_data(df)
    assert report.outlier_flag_count == 1
    assert report.outliers_on_event_days == 1        # spike on a snap day
    # THE core guarantee: the original value survives untouched.
    assert clean["sales_quantity"].max() == 648
    assert len(clean) == 30


def test_outlier_flagging_is_per_product():
    # 40 units is normal for P_BIG but a huge spike for P_SMALL.
    dates = list(pd.date_range("2024-01-01", periods=20).strftime("%Y-%m-%d"))
    df_big = make_df(
        date=dates, product_id=["P_BIG"] * 20, product_name=["P_BIG"] * 20,
        category=["FOODS"] * 20, sales_quantity=[38, 40, 42, 39] * 5,
        price=[2.5] * 20, snap_day=[0] * 20, holiday=[0] * 20,
        event_name=[None] * 20, store_id=["S1"] * 20,
    )
    df_small = make_df(
        date=dates, product_id=["P_SMALL"] * 20,
        product_name=["P_SMALL"] * 20, category=["FOODS"] * 20,
        sales_quantity=[1, 2, 1, 2] * 4 + [1, 2, 1, 40],  # spike at end
        price=[2.5] * 20, snap_day=[0] * 20, holiday=[0] * 20,
        event_name=[None] * 20, store_id=["S1"] * 20,
    )
    df = pd.concat([df_big, df_small], ignore_index=True)
    flags = flag_demand_outliers(df)
    flagged_products = df.loc[flags == 1, "product_id"].unique().tolist()
    assert flagged_products == ["P_SMALL"]


# ------------------------------ Invariants --------------------------------

def test_output_is_chronologically_sorted_per_product():
    df = make_df().sample(frac=1, random_state=42)  # shuffle input
    clean, _ = preprocess_sales_data(df)
    for _, group in clean.groupby(["product_id", "store_id"]):
        assert group["date"].is_monotonic_increasing


def test_report_contains_unresolved_price_count_and_serializes():
    import json
    _, report = preprocess_sales_data(make_df(price=[None, 2.5, 2.5, 2.5]))
    payload = report.to_dict()
    assert "unresolved_price_count" in payload
    assert payload["unresolved_price_count"] == 1
    json.dumps(payload)  # raises if not serializable
