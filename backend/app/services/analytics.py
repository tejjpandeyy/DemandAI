"""Analytics service layer for DemandAI.

Pure, reusable analysis functions over the cleaned sales dataset. Every
function takes a DataFrame and returns a structured pandas DataFrame or a
JSON-serializable dict, so the SAME code powers:

    - the offline EDA report (scripts/run_eda.py, Phase 4)
    - the FastAPI analytics endpoints (Phase 12)
    - the React dashboard charts (Phase 14/15)

Conventions:
    - Statistics are computed over REAL observations only: rows with
      is_gap_fill == 1 (structural rows, if present) are excluded.
    - Descriptive only. Moving averages produced here are for
      VISUALIZATION, not ML features (Phase 5 builds those separately,
      leakage-safely). Event and price analyses report ASSOCIATIONS /
      observed demand differences, never causal claims.
"""

import numpy as np
import pandas as pd

# Minimum columns any analytics function needs.
ANALYTICS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "product_id",
    "sales_quantity",
)

# Binary event-flag columns handled generically when present.
EVENT_FLAG_COLUMNS: tuple[str, ...] = ("holiday", "snap_day", "promotion")

# The explicit "no event" category written by preprocessing.
NO_EVENT: str = "none"

DAY_ORDER: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
)


# ---------------------------------------------------------------------------
# Guards and helpers
# ---------------------------------------------------------------------------

def _check_input(df: pd.DataFrame) -> None:
    """Raise ValueError on unusable input (empty / missing columns)."""
    if df is None or df.empty:
        raise ValueError("Analytics input DataFrame is empty.")
    missing = [c for c in ANALYTICS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Analytics input is missing required columns: {missing}."
        )


def _real_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Return only real observed rows, excluding structural gap rows."""
    if "is_gap_fill" in df.columns:
        return df[df["is_gap_fill"] == 0]
    return df


def _prepared(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, filter to real rows, and ensure datetime dates."""
    _check_input(df)
    out = _real_observations(df).copy()
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"])
    return out


def _daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Total units per calendar date (all products combined)."""
    return (
        df.groupby("date", as_index=False)["sales_quantity"]
        .sum()
        .rename(columns={"sales_quantity": "total_units"})
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 1. Dataset summary
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> dict:
    """High-level KPIs for the dashboard header.

    ``avg_daily_demand`` = total units sold / number of distinct dates
    (units per day across the whole assortment).
    """
    real = _prepared(df)
    n_dates = int(real["date"].nunique())
    total_units = int(real["sales_quantity"].sum())
    summary = {
        "total_rows": int(len(real)),
        "total_products": int(real["product_id"].nunique()),
        "total_units_sold": total_units,
        "date_start": real["date"].min().date().isoformat(),
        "date_end": real["date"].max().date().isoformat(),
        "n_dates": n_dates,
        "avg_daily_demand": round(total_units / n_dates, 2) if n_dates else 0.0,
        "zero_sales_pct": round(
            float((real["sales_quantity"] == 0).mean()) * 100, 2
        ),
        "outlier_flag_count": (
            int(real["outlier_flag"].sum())
            if "outlier_flag" in real.columns else 0
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# 2-4. Trends
# ---------------------------------------------------------------------------

def daily_sales(df: pd.DataFrame,
                moving_averages: tuple[int, ...] = (7, 30)) -> pd.DataFrame:
    """Daily total units, with optional trailing moving averages.

    The ``ma_*`` columns are DESCRIPTIVE smoothing for charts only.
    They are computed on the aggregate series after the fact and are
    NOT ML features -- Phase 5 builds leakage-safe per-product rolling
    features separately.
    """
    daily = _daily_totals(_prepared(df))
    for window in moving_averages:
        daily[f"ma_{window}"] = (
            daily["total_units"].rolling(window=window).mean().round(2)
        )
    return daily


def weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Total units per calendar week (weeks labelled by their start date)."""
    real = _prepared(df)
    week_start = real["date"].dt.to_period("W").dt.start_time
    return (
        real.assign(week_start=week_start)
        .groupby("week_start", as_index=False)["sales_quantity"]
        .sum()
        .rename(columns={"sales_quantity": "total_units"})
        .sort_values("week_start")
        .reset_index(drop=True)
    )


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Total units per calendar month (months labelled by their start date)."""
    real = _prepared(df)
    month_start = real["date"].dt.to_period("M").dt.start_time
    return (
        real.assign(month_start=month_start)
        .groupby("month_start", as_index=False)["sales_quantity"]
        .sum()
        .rename(columns={"sales_quantity": "total_units"})
        .sort_values("month_start")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 5. Product analysis
# ---------------------------------------------------------------------------

def product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product demand profile.

    Columns: total_units, n_days, avg_daily_units, std_daily_units
    (variability), cv (coefficient of variation = std/mean, comparable
    across products of different scale), zero_sales_pct, and
    outlier_count when the flag column exists.
    """
    real = _prepared(df)
    grouped = real.groupby("product_id")
    metrics = grouped["sales_quantity"].agg(
        total_units="sum",
        n_days="count",
        avg_daily_units="mean",
        std_daily_units="std",
    )
    metrics["cv"] = metrics["std_daily_units"] / metrics["avg_daily_units"]
    metrics["zero_sales_pct"] = (
        grouped["sales_quantity"].apply(lambda s: (s == 0).mean() * 100)
    )
    if "outlier_flag" in real.columns:
        metrics["outlier_count"] = grouped["outlier_flag"].sum()
    if "product_name" in real.columns:
        metrics.insert(0, "product_name", grouped["product_name"].first())
    return (
        metrics.round(3)
        .sort_values("total_units", ascending=False)
        .reset_index()
    )


def top_products(df: pd.DataFrame, n: int = 10,
                 lowest: bool = False) -> pd.DataFrame:
    """Top (or lowest, with ``lowest=True``) N products by total units."""
    metrics = product_metrics(df)
    if lowest:
        return metrics.nsmallest(n, "total_units").reset_index(drop=True)
    return metrics.nlargest(n, "total_units").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Event analysis (associations only -- never causal claims)
# ---------------------------------------------------------------------------

def event_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Observed demand differences on event days vs. their baseline.

    One unified table covering:
      - each binary flag present (holiday, snap_day, promotion), with
        baseline = mean sales on rows where that flag is 0;
      - each named event in event_name, with baseline = mean sales on
        "none" (no-event) days.

    ``pct_vs_baseline`` is an OBSERVED DEMAND DIFFERENCE (association).
    It is confounded by seasonality, weekday, price, and co-occurring
    events, and must not be read as a causal effect of the event.
    """
    real = _prepared(df)
    rows: list[dict] = []

    for flag in EVENT_FLAG_COLUMNS:
        if flag not in real.columns:
            continue
        on = real.loc[real[flag] == 1, "sales_quantity"]
        off = real.loc[real[flag] == 0, "sales_quantity"]
        if on.empty or off.empty:
            continue
        baseline = float(off.mean())
        rows.append({
            "event": flag,
            "event_type": "flag",
            "observations": int(len(on)),
            "avg_units": round(float(on.mean()), 3),
            "median_units": float(on.median()),
            "total_units": int(on.sum()),
            "baseline_avg_units": round(baseline, 3),
            "pct_vs_baseline": (
                round((float(on.mean()) / baseline - 1) * 100, 2)
                if baseline > 0 else np.nan
            ),
        })

    if "event_name" in real.columns:
        baseline_series = real.loc[
            real["event_name"] == NO_EVENT, "sales_quantity"
        ]
        baseline = float(baseline_series.mean()) if len(baseline_series) else np.nan
        named = real[real["event_name"] != NO_EVENT]
        for name, group in named.groupby("event_name"):
            sales = group["sales_quantity"]
            rows.append({
                "event": name,
                "event_type": "named_event",
                "observations": int(len(sales)),
                "avg_units": round(float(sales.mean()), 3),
                "median_units": float(sales.median()),
                "total_units": int(sales.sum()),
                "baseline_avg_units": round(baseline, 3),
                "pct_vs_baseline": (
                    round((float(sales.mean()) / baseline - 1) * 100, 2)
                    if baseline and baseline > 0 else np.nan
                ),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("pct_vs_baseline", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 7. Outlier analysis (read-only over the existing outlier_flag)
# ---------------------------------------------------------------------------

def outlier_analysis(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Describe where the preprocessing-stage outlier flags landed.

    Read-only: nothing is deleted or modified. "Event day" = any binary
    event flag equals 1 on that row.
    """
    real = _prepared(df)
    if "outlier_flag" not in real.columns:
        raise ValueError(
            "outlier_flag column not found -- run preprocessing "
            "(Phase 3) before outlier analysis."
        )
    flagged = real[real["outlier_flag"] == 1]
    total = int(len(flagged))

    result: dict = {
        "total_flagged": total,
        "flagged_pct_of_rows": round(total / len(real) * 100, 3),
        "by_product": (
            flagged.groupby("product_id").size()
            .sort_values(ascending=False).head(top_n).to_dict()
        ),
    }
    for flag in EVENT_FLAG_COLUMNS:
        if flag in real.columns:
            result[f"on_{flag}_days"] = int(
                (flagged[flag] == 1).sum()
            )
    if "event_name" in real.columns:
        result["by_event_name"] = (
            flagged[flagged["event_name"] != NO_EVENT]
            .groupby("event_name").size()
            .sort_values(ascending=False).to_dict()
        )

    event_cols = [c for c in EVENT_FLAG_COLUMNS if c in real.columns]
    if event_cols and total:
        on_event = int(flagged[event_cols].max(axis=1).sum())
        result["pct_on_event_days"] = round(on_event / total * 100, 2)
        # Context: how common are event days overall?
        result["pct_of_all_rows_on_event_days"] = round(
            float(real[event_cols].max(axis=1).mean()) * 100, 2
        )
    return result


# ---------------------------------------------------------------------------
# 8. Price-demand analysis (associations only)
# ---------------------------------------------------------------------------

def price_analysis(df: pd.DataFrame,
                   min_obs: int = 30) -> tuple[pd.DataFrame, dict]:
    """Per-product price profile and price-demand correlations.

    Returns (per_product_df, summary_dict).

    Correlation caveats (also embedded in the summary's ``note``):
      - The POOLED correlation across all products mostly reflects
        PRODUCT IDENTITY (cheap staples sell more units than expensive
        items), not price effects.
      - The WITHIN-PRODUCT correlation removes product identity but is
        still confounded by time, seasonality, and events.
      Neither is a causal price elasticity.

    Within-product correlation is computed only where statistically
    meaningful: at least ``min_obs`` observations and at least two
    distinct prices with non-zero variance.
    """
    real = _prepared(df)
    if "price" not in real.columns:
        raise ValueError("price column not found in dataset.")

    records: list[dict] = []
    for pid, group in real.groupby("product_id"):
        price = group["price"].dropna()
        sales = group.loc[price.index, "sales_quantity"]
        n_changes = int((price.diff().fillna(0) != 0).sum())
        corr = np.nan
        if (
            len(price) >= min_obs
            and price.nunique() >= 2
            and price.std() > 0
            and sales.std() > 0
        ):
            corr = float(price.corr(sales))
        records.append({
            "product_id": pid,
            "avg_price": round(float(price.mean()), 3),
            "min_price": float(price.min()),
            "max_price": float(price.max()),
            "n_price_changes": n_changes,
            "within_product_corr": (
                round(corr, 4) if not np.isnan(corr) else np.nan
            ),
        })
    per_product = pd.DataFrame(records).sort_values(
        "avg_price", ascending=False
    ).reset_index(drop=True)

    pooled = real[["price", "sales_quantity"]].dropna()
    pooled_corr = (
        float(pooled["price"].corr(pooled["sales_quantity"]))
        if pooled["price"].std() > 0 else np.nan
    )
    computable = per_product["within_product_corr"].notna()
    summary = {
        "pooled_correlation": round(pooled_corr, 4),
        "pooled_correlation_caveat": (
            "Pooled correlation mixes products; it mostly reflects that "
            "cheaper products sell more units (product identity), not a "
            "price effect."
        ),
        "products_with_computable_within_corr": int(computable.sum()),
        "mean_within_product_corr": (
            round(float(per_product.loc[computable,
                                        "within_product_corr"].mean()), 4)
            if computable.any() else None
        ),
        "note": (
            "Correlations are observed associations. Even within-product "
            "correlation is confounded by seasonality, events, and time "
            "trends; no causal claim about price effects is made."
        ),
    }
    return per_product, summary


# ---------------------------------------------------------------------------
# 9. Day-of-week analysis
# ---------------------------------------------------------------------------

def day_of_week_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Average and total demand per weekday, ordered Monday -> Sunday.

    ``avg_daily_total_units`` = mean of the DAILY TOTAL over all dates
    falling on that weekday (i.e. an average Tuesday's total demand).
    """
    daily = _daily_totals(_prepared(df))
    daily["day_of_week"] = pd.Categorical(
        daily["date"].dt.day_name(), categories=list(DAY_ORDER), ordered=True
    )
    out = (
        daily.groupby("day_of_week", observed=False)["total_units"]
        .agg(avg_daily_total_units="mean", total_units="sum",
             n_dates="count")
        .round(2)
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# 10. Category analysis (multi-category ready; nothing fabricated)
# ---------------------------------------------------------------------------

def category_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category demand profile.

    Works for any number of categories. The current M5 subset contains
    only FOODS, so this returns a single row for that data -- correct,
    not a limitation. Future uploads with more categories are handled
    by the same code.
    """
    real = _prepared(df)
    if "category" not in real.columns:
        raise ValueError("category column not found in dataset.")
    grouped = real.groupby("category")
    out = grouped.agg(
        n_products=("product_id", "nunique"),
        total_units=("sales_quantity", "sum"),
        avg_units_per_row=("sales_quantity", "mean"),
        zero_sales_pct=("sales_quantity", lambda s: (s == 0).mean() * 100),
    ).round(3)
    out["units_share_pct"] = (
        out["total_units"] / out["total_units"].sum() * 100
    ).round(2)
    return out.sort_values(
        "total_units", ascending=False
    ).reset_index()
