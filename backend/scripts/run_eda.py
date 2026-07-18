"""Run the full EDA suite on datasets/retail_sales_clean.csv.

Thin wrapper: all analysis logic lives in app/services/analytics.py so
the exact same code path serves the FastAPI analytics endpoints later.

Prints the key findings and writes reusable analytics outputs to
backend/datasets/analytics/ (ignored by Git under the existing
datasets/* rule).

Usage (from the backend/ directory):
    python scripts/run_eda.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services import analytics  # noqa: E402

INPUT_PATH = BACKEND_DIR / "datasets" / "retail_sales_clean.csv"
OUTPUT_DIR = BACKEND_DIR / "datasets" / "analytics"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run scripts/run_preprocessing.py first."
        )
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Summary ---
    section("1. DATASET SUMMARY")
    summary = analytics.dataset_summary(df)
    for key, value in summary.items():
        print(f"  {key:22s}: {value}")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- 2-4. Trends ---
    section("2-4. TRENDS")
    daily = analytics.daily_sales(df)
    weekly = analytics.weekly_sales(df)
    monthly = analytics.monthly_sales(df)
    daily.to_csv(OUTPUT_DIR / "daily_sales.csv", index=False)
    weekly.to_csv(OUTPUT_DIR / "weekly_sales.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_sales.csv", index=False)
    print(f"  Daily points  : {len(daily):,} "
          f"(avg {daily['total_units'].mean():,.0f} units/day)")
    print(f"  Weekly points : {len(weekly):,}")
    print(f"  Monthly points: {len(monthly):,}")
    growth = (monthly["total_units"].iloc[-13:-1].mean()
              / monthly["total_units"].iloc[1:13].mean() - 1) * 100
    print(f"  Demand level, last 12 full months vs first 12: {growth:+.1f}%")
    print("  (ma_7 / ma_30 columns are chart smoothing only -- NOT ML "
          "features)")

    # --- 5. Products ---
    section("5. PRODUCT ANALYSIS")
    metrics = analytics.product_metrics(df)
    metrics.to_csv(OUTPUT_DIR / "product_metrics.csv", index=False)
    print("  Top 5 by total units:")
    print(analytics.top_products(df, 5)[
        ["product_id", "total_units", "avg_daily_units", "cv",
         "zero_sales_pct"]].to_string(index=False))
    print("\n  Lowest 5 by total units:")
    print(analytics.top_products(df, 5, lowest=True)[
        ["product_id", "total_units", "avg_daily_units", "cv",
         "zero_sales_pct"]].to_string(index=False))

    # --- 6. Events ---
    section("6. EVENT ANALYSIS (observed associations, not causation)")
    events = analytics.event_analysis(df)
    events.to_csv(OUTPUT_DIR / "event_analysis.csv", index=False)
    named = events[events["event_type"] == "named_event"]
    print("  Binary flags:")
    print(events[events["event_type"] == "flag"].to_string(index=False))
    print("\n  Strongest positive named-event associations:")
    print(named.head(5)[["event", "observations", "avg_units",
                         "pct_vs_baseline"]].to_string(index=False))
    print("\n  Strongest negative named-event associations:")
    print(named.tail(5)[["event", "observations", "avg_units",
                         "pct_vs_baseline"]].to_string(index=False))

    # --- 7. Outliers ---
    section("7. OUTLIER ANALYSIS (flags are read-only)")
    outliers = analytics.outlier_analysis(df)
    (OUTPUT_DIR / "outlier_analysis.json").write_text(
        json.dumps(outliers, indent=2)
    )
    for key in ("total_flagged", "flagged_pct_of_rows",
                "pct_on_event_days", "pct_of_all_rows_on_event_days"):
        if key in outliers:
            print(f"  {key:30s}: {outliers[key]}")
    print(f"  top products by flags        : "
          f"{dict(list(outliers['by_product'].items())[:3])}")
    if "by_event_name" in outliers:
        print(f"  top events by flags          : "
              f"{dict(list(outliers['by_event_name'].items())[:3])}")

    # --- 8. Price ---
    section("8. PRICE-DEMAND ANALYSIS (associations only)")
    per_product, price_summary = analytics.price_analysis(df)
    per_product.to_csv(OUTPUT_DIR / "price_analysis.csv", index=False)
    for key, value in price_summary.items():
        print(f"  {key}: {value}")

    # --- 9. Day of week ---
    section("9. DAY-OF-WEEK ANALYSIS")
    dow = analytics.day_of_week_analysis(df)
    dow.to_csv(OUTPUT_DIR / "day_of_week_analysis.csv", index=False)
    print(dow.to_string(index=False))

    # --- 10. Categories ---
    section("10. CATEGORY ANALYSIS")
    categories = analytics.category_analysis(df)
    categories.to_csv(OUTPUT_DIR / "category_analysis.csv", index=False)
    print(categories.to_string(index=False))
    if len(categories) == 1:
        print("  (Single category is expected for the M5 FOODS subset; "
          "the code supports any number for future uploads.)")

    print(f"\nAll analytics outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
