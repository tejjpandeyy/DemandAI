"""Leakage-safe feature engineering for DemandAI.

Pure, reusable functions that add forecasting features to the cleaned
sales dataset. THE PIPELINE NEVER USES FUTURE INFORMATION:

    - Lags: groupby(product).shift(lag) -- strictly past rows.
    - Rolling/expanding stats on the TARGET: always shift(1) BEFORE
      rolling/expanding, so the window covers days t-1, t-2, ... and
      never includes today's sales_quantity (the value being predicted).
    - Price rolling stats: also shift(1) first -- maximally conservative,
      and convenient for recursive forecasting where future prices are
      unknown. price_change / price_pct_change use current vs previous
      price (no future information).
    - NaNs produced by lagging/rolling warm-up are left as NaN: they
      truthfully mean "not enough history existed at this point". The
      training phase decides how to handle them.

PRECONDITION (enforced, not assumed): shift-based lags are only correct
on a CONTINUOUS daily series per product -- with a date gap, shift(1)
would silently mean "previous ROW", not "previous DAY". Validation
therefore rejects gapped, duplicated, or unsorted series.

No printing here; reporting belongs to scripts/run_feature_engineering.py.
"""

import numpy as np
import pandas as pd

FEATURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "product_id",
    "sales_quantity",
)

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28)
DEFAULT_MEDIAN_WINDOWS: tuple[int, ...] = (7, 28)
NO_EVENT: str = "none"


class FeatureEngineeringError(Exception):
    """Raised when input data or configuration would produce wrong features."""


def _group_keys(df: pd.DataFrame) -> list[str]:
    """Grouping keys for per-series operations (store-aware when present)."""
    return (
        ["product_id", "store_id"] if "store_id" in df.columns
        else ["product_id"]
    )


# ---------------------------------------------------------------------------
# Validation (reject anything that would make shift-based features wrong)
# ---------------------------------------------------------------------------

def validate_feature_input(df: pd.DataFrame) -> None:
    """Reject inputs on which shift-based features would be incorrect.

    Raises FeatureEngineeringError for: empty input, missing required
    columns, non-datetime-parseable dates, duplicate dates within a
    product series, unsorted dates within a product series, and date
    gaps (shift assumes previous row == previous day).
    """
    if df is None or df.empty:
        raise FeatureEngineeringError("Input DataFrame is empty.")

    missing = [c for c in FEATURE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FeatureEngineeringError(
            f"Missing required columns: {missing}."
        )
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise FeatureEngineeringError(
            "date column must be datetime (load with parse_dates=['date'])."
        )

    keys = _group_keys(df)

    dup_mask = df.duplicated(subset=keys + ["date"])
    if dup_mask.any():
        raise FeatureEngineeringError(
            f"{int(dup_mask.sum()):,} duplicate (product, date) rows -- "
            "each product series must contain each date exactly once."
        )

    grouped_dates = df.groupby(keys, sort=False)["date"]
    monotonic = grouped_dates.apply(lambda s: s.is_monotonic_increasing)
    if not monotonic.all():
        bad = monotonic[~monotonic].index.tolist()[:3]
        raise FeatureEngineeringError(
            f"Dates are not sorted ascending within series (e.g. {bad}). "
            "Sort by product_id, date before feature engineering."
        )

    stats = grouped_dates.agg(first="min", last="max", n="count")
    expected = (stats["last"] - stats["first"]).dt.days + 1
    gaps = (expected - stats["n"]).astype(int)
    gapped = gaps[gaps > 0]
    if len(gapped):
        worst = gapped.sort_values(ascending=False).head(3).to_dict()
        raise FeatureEngineeringError(
            f"{int(gapped.sum()):,} missing days across {len(gapped)} "
            f"series (worst: {worst}). Shift-based lags require a "
            "continuous daily calendar per product; resolve gaps "
            "explicitly before feature engineering."
        )


def _validate_config(lags: tuple[int, ...],
                     windows: tuple[int, ...]) -> None:
    """Reject configurations that would produce meaningless features."""
    bad_lags = [lag for lag in lags if not isinstance(lag, (int, np.integer))
                or lag < 1]
    if bad_lags:
        raise FeatureEngineeringError(
            f"Lags must be positive integers; got {bad_lags}. A lag of 0 "
            "would leak today's target; negative lags would use the future."
        )
    bad_windows = [w for w in windows if not isinstance(w, (int, np.integer))
                   or w < 2]
    if bad_windows:
        raise FeatureEngineeringError(
            f"Rolling windows must be integers >= 2; got {bad_windows}."
        )


# ---------------------------------------------------------------------------
# 1. Calendar features (deterministic functions of the date -- no leakage)
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar features derived from the date only."""
    out = df.copy()
    date = out["date"]
    out["year"] = date.dt.year
    out["month"] = date.dt.month
    out["quarter"] = date.dt.quarter
    out["week_of_year"] = date.dt.isocalendar().week.astype(int)
    out["day_of_month"] = date.dt.day
    out["day_of_week"] = date.dt.dayofweek          # 0 = Monday
    out["day_of_year"] = date.dt.dayofyear
    out["is_weekend"] = (date.dt.dayofweek >= 5).astype(int)
    out["is_month_start"] = date.dt.is_month_start.astype(int)
    out["is_month_end"] = date.dt.is_month_end.astype(int)
    return out


# ---------------------------------------------------------------------------
# 2. Lag features (groupby + shift; strictly past rows)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame,
                     lags: tuple[int, ...] = DEFAULT_LAGS) -> pd.DataFrame:
    """Add sales lags: lag_k = sales_quantity k days earlier.

    Implementation is groupby(product).shift(k): the value comes only
    from earlier rows of the SAME product series. The first k rows of
    each series are NaN by construction and stay NaN.
    """
    out = df.copy()
    grouped = out.groupby(_group_keys(out), sort=False)["sales_quantity"]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out


# ---------------------------------------------------------------------------
# 3+4+6. Rolling / expanding statistics (shift(1) BEFORE rolling)
# ---------------------------------------------------------------------------

def _shifted_target(df: pd.DataFrame) -> pd.Series:
    """sales_quantity shifted by 1 within each series.

    THE leakage guard: every rolling/expanding statistic is computed on
    this series, so the window ends at day t-1 and can never contain
    today's target value.
    """
    return df.groupby(_group_keys(df), sort=False)["sales_quantity"].shift(1)


def _align(grouped_result: pd.Series) -> pd.Series:
    """Map a groupby-rolling result (MultiIndex) back onto the row index."""
    grouped_result.index = grouped_result.index.get_level_values(-1)
    return grouped_result


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    median_windows: tuple[int, ...] = DEFAULT_MEDIAN_WINDOWS,
) -> pd.DataFrame:
    """Add rolling mean/std/min/max (+ median for selected windows).

    All statistics are computed on the shift(1)-ed target, i.e.
    sales.shift(1).rolling(w).stat() -- the window covers days
    t-1 ... t-w and NEVER includes today. min_periods equals the window,
    so a value only appears once a FULL window of history exists;
    earlier rows stay NaN.
    """
    out = df.copy()
    keys = _group_keys(out)
    shifted = _shifted_target(out)
    grouped = (
        out[keys].assign(_shifted=shifted)
        .groupby(keys, sort=False)["_shifted"]
    )
    for window in windows:
        roll = grouped.rolling(window=window, min_periods=window)
        out[f"rolling_mean_{window}"] = _align(roll.mean())
        out[f"rolling_std_{window}"] = _align(roll.std())
        out[f"rolling_min_{window}"] = _align(roll.min())
        out[f"rolling_max_{window}"] = _align(roll.max())
    for window in median_windows:
        roll = grouped.rolling(window=window, min_periods=window)
        out[f"rolling_median_{window}"] = _align(roll.median())
    return out


def add_expanding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add expanding_mean: mean of ALL strictly-previous observations.

    Computed on the shift(1)-ed target, so at day t it is the mean of
    days 1 ... t-1 of the same product. The first row of each series is
    NaN (no history) and stays NaN.
    """
    out = df.copy()
    keys = _group_keys(out)
    grouped = (
        out[keys].assign(_shifted=_shifted_target(out))
        .groupby(keys, sort=False)["_shifted"]
    )
    out["expanding_mean"] = _align(grouped.expanding(min_periods=1).mean())
    return out


# ---------------------------------------------------------------------------
# 5. Price features (past-only)
# ---------------------------------------------------------------------------

def add_price_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Add price-derived features using no future information.

    price_change / price_pct_change compare today's price with the
    PREVIOUS day's price (diff / pct_change within the product series).
    Rolling price statistics additionally shift(1) first -- stricter
    than necessary for a known covariate, but (a) uniformly leakage-safe
    and (b) forecast-friendly: future prices beyond the last known one
    do not exist at prediction time (Phase 9).
    """
    out = df.copy()
    if "price" not in out.columns:
        return out
    keys = _group_keys(out)
    grouped_price = out.groupby(keys, sort=False)["price"]
    out["price_change"] = grouped_price.diff()
    out["price_pct_change"] = grouped_price.pct_change()

    shifted_price = grouped_price.shift(1)
    grouped_shifted = (
        out[keys].assign(_p=shifted_price).groupby(keys, sort=False)["_p"]
    )
    roll = grouped_shifted.rolling(window=window, min_periods=window)
    out[f"rolling_price_mean_{window}"] = _align(roll.mean())
    out[f"rolling_price_std_{window}"] = _align(roll.std())
    return out


# ---------------------------------------------------------------------------
# 7. Event features (kept as-is; NO encoding in this phase)
# ---------------------------------------------------------------------------

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep binary event flags; add has_named_event.

    holiday / snap_day / promotion pass through unchanged when present.
    has_named_event = 1 when event_name is a real event (not "none").
    event_name itself stays raw -- encoding is a later-phase decision.
    """
    out = df.copy()
    if "event_name" in out.columns:
        out["has_named_event"] = (out["event_name"] != NO_EVENT).astype(int)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    median_windows: tuple[int, ...] = DEFAULT_MEDIAN_WINDOWS,
    price_window: int = 7,
    validate: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full leakage-safe feature pipeline.

    Args:
        df: Cleaned sales data (Phase 3 output), datetime date column.
        lags: Lag distances in days (positive integers).
        windows: Rolling windows for mean/std/min/max.
        median_windows: Rolling windows for the median.
        price_window: Window for rolling price statistics.
        validate: Run input validation (recommended; disable only for
            pre-validated data in tight loops).

    Returns:
        (features_df, generated_feature_names). Row count is UNCHANGED
        from the input; warm-up NaNs are preserved, never filled or
        dropped.
    """
    _validate_config(lags, tuple(windows) + tuple(median_windows)
                     + (price_window,))
    if validate:
        validate_feature_input(df)

    before_cols = list(df.columns)
    out = df.reset_index(drop=True)
    out = add_calendar_features(out)
    out = add_lag_features(out, lags=lags)
    out = add_rolling_features(out, windows=windows,
                               median_windows=median_windows)
    out = add_expanding_features(out)
    out = add_price_features(out, window=price_window)
    out = add_event_features(out)

    generated = [c for c in out.columns if c not in before_cols]
    return out, generated
