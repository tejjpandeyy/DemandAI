"""Print a dataset-understanding report for datasets/retail_sales.csv.

Run after prepare_m5_subset.py. This is the Phase 2 "know your data"
step: shape, date coverage, demand statistics, sparsity (zero-sales
days), and event frequencies -- everything we need to know BEFORE
making preprocessing and modeling decisions.

Usage (from the backend/ directory):
    python scripts/inspect_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

# Allow "from ml.data_validation import ..." when run as a script.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml.data_validation import validate_sales_dataframe  # noqa: E402

DATASET_PATH = BACKEND_DIR / "datasets" / "retail_sales.csv"


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run scripts/prepare_m5_subset.py first."
        )

    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])

    section("1. VALIDATION")
    result = validate_sales_dataframe(df)
    print(result.summary())

    section("2. SHAPE AND MEMORY")
    print(f"Rows x Columns : {df.shape[0]:,} x {df.shape[1]}")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Dtypes:\n{df.dtypes.to_string()}")

    section("3. TIME COVERAGE")
    print(f"Date range     : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Total days     : {df['date'].nunique():,}")
    days_per_product = df.groupby("product_id")["date"].nunique()
    print(f"Days per product: min={days_per_product.min()}, "
          f"median={int(days_per_product.median())}, "
          f"max={days_per_product.max()}")
    print("(min << max means some products launched later -- expected in M5)")

    section("4. DEMAND STATISTICS")
    print(df["sales_quantity"].describe().round(2).to_string())
    zero_frac = float((df["sales_quantity"] == 0).mean())
    print(f"\nZero-sales days: {zero_frac:.1%} of all rows")
    print("(High zero share = intermittent demand; affects metric choice "
          "later -- MAPE breaks on zeros.)")

    section("5. PRODUCTS")
    print(f"Unique products: {df['product_id'].nunique()}")
    totals = df.groupby("product_id")["sales_quantity"].sum().sort_values()
    print("\nTop 5 products by total units:")
    print(totals.tail(5).to_string())
    print("\nBottom 5 products by total units:")
    print(totals.head(5).to_string())

    section("6. PRICE")
    print(df["price"].describe().round(2).to_string())
    price_changes = (
        df.sort_values("date")
        .groupby("product_id")["price"]
        .apply(lambda s: (s.diff().fillna(0) != 0).sum())
    )
    print(f"\nAvg price changes per product over the whole period: "
          f"{price_changes.mean():.1f}")

    section("7. EVENT FLAGS")
    # Report whichever binary event flags this dataset actually contains.
    event_flags = [c for c in ("holiday", "snap_day", "promotion")
                   if c in df.columns]
    if not event_flags:
        print("No event-flag columns present in this dataset.")
    for flag in event_flags:
        freq = df[flag].mean()
        avg_off = df.loc[df[flag] == 0, "sales_quantity"].mean()
        avg_on = df.loc[df[flag] == 1, "sales_quantity"].mean()
        line = f"{flag:10s}: {freq:.1%} of rows"
        if avg_off > 0 and freq > 0:
            lift = (avg_on / avg_off - 1) * 100
            line += f" | naive demand lift on flag days: {lift:+.1f}%"
        print(line)

    # Per-event demand: which named events matter, and in which direction?
    if "event_name" in df.columns and df["event_name"].notna().any():
        print("\nTop events by frequency (avg units/product/day vs. "
              "non-event baseline):")
        baseline = df.loc[df["event_name"].isna(), "sales_quantity"].mean()
        event_stats = (
            df.dropna(subset=["event_name"])
            .groupby("event_name")["sales_quantity"]
            .agg(days="count", avg_units="mean")
            .sort_values("days", ascending=False)
            .head(10)
        )
        event_stats["vs_baseline"] = (
            (event_stats["avg_units"] / baseline - 1) * 100
        ).map(lambda v: f"{v:+.1f}%")
        event_stats["avg_units"] = event_stats["avg_units"].round(2)
        print(event_stats.to_string())
        print(f"(Non-event baseline: {baseline:.2f} avg units. Note how "
              "different events move demand in DIFFERENT directions -- "
              "this is why we preserved event_name instead of only a "
              "binary flag.)")

    section("8. SAMPLE ROWS")
    print(df.head(8).to_string(index=False))

    print("\nDataset understanding complete. If validation shows VALID, "
          "Phase 2 is done.")


if __name__ == "__main__":
    main()
