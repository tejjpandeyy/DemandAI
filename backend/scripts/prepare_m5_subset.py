"""Convert raw M5 (Walmart) competition files into the DemandAI schema.

Reads the three raw M5 files from ``datasets/raw/``:
    - sales_train_evaluation.csv (falls back to sales_train_validation.csv)
    - calendar.csv
    - sell_prices.csv

Produces ``datasets/retail_sales.csv`` in long format with columns:
    date, product_id, product_name, category, sales_quantity,
    price, snap_day, holiday, event_name, store_id

``event_name`` preserves the raw M5 event identity (Christmas, SuperBowl,
Easter, ...) and is empty on non-event days. It is kept raw -- not encoded
-- so EDA and later feature engineering can decide how to use it.

Note on naming: M5 has no marketing-promotion flag. The SNAP disbursement
flag is a real demand driver but it is NOT a promotion, so it is stored
under its own honest name ``snap_day``. The optional ``promotion`` column
in the DemandAI schema stays reserved for datasets that genuinely have one.

Key transformations:
    1. Filter to one store and one category, keep the top-N selling items.
    2. Melt the wide day columns (d_1 ... d_1941) into one row per item-day.
    3. Join the calendar to attach real dates, events, and SNAP flags.
    4. Join weekly prices; rows with no price are dropped because a missing
       price in M5 means the product was NOT yet offered in that store
       (pre-launch period), so those zero-sales rows are not real demand.

Usage (from the backend/ directory):
    python scripts/prepare_m5_subset.py
    python scripts/prepare_m5_subset.py --store TX_2 --category HOUSEHOLD --top-n 30
"""

import argparse
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DIR_DEFAULT = BACKEND_DIR / "datasets" / "raw"
OUTPUT_DEFAULT = BACKEND_DIR / "datasets" / "retail_sales.csv"

SALES_FILE_CANDIDATES = (
    "sales_train_evaluation.csv",   # preferred: includes the final 28 days
    "sales_train_validation.csv",   # fallback: shorter version
)


def load_raw_files(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw M5 CSVs, raising a clear error if any is missing."""
    sales = None
    for name in SALES_FILE_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            print(f"Loading {candidate.name} (this file is ~120 MB, please wait)...")
            sales = pd.read_csv(candidate)
            break
    if sales is None:
        raise FileNotFoundError(
            f"No M5 sales file found in {raw_dir}. Expected one of "
            f"{SALES_FILE_CANDIDATES}. Download the M5 dataset from Kaggle "
            "(competition: m5-forecasting-accuracy) and extract it there."
        )

    calendar_path = raw_dir / "calendar.csv"
    prices_path = raw_dir / "sell_prices.csv"
    for path in (calendar_path, prices_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw file: {path}")

    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    return sales, calendar, prices


def build_subset(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    store_id: str,
    category: str,
    top_n: int,
) -> pd.DataFrame:
    """Filter, melt, and join the raw M5 data into the DemandAI schema.

    Args:
        sales: Wide-format sales frame (one row per item-store, d_* columns).
        calendar: Day-code to date/event/SNAP mapping.
        prices: Weekly item prices per store.
        store_id: M5 store to keep, e.g. "CA_1".
        category: M5 category to keep: FOODS, HOBBIES, or HOUSEHOLD.
        top_n: Number of best-selling items to keep.

    Returns:
        Long-format DataFrame matching the DemandAI schema, sorted by
        product_id then date.
    """
    subset = sales[
        (sales["store_id"] == store_id) & (sales["cat_id"] == category)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No rows for store_id={store_id!r} and category={category!r}. "
            "Valid stores look like CA_1..CA_4, TX_1..TX_3, WI_1..WI_3; "
            "valid categories are FOODS, HOBBIES, HOUSEHOLD."
        )

    day_cols = [c for c in subset.columns if c.startswith("d_")]

    # Keep the top-N items by total historical units sold.
    totals = subset[day_cols].sum(axis=1)
    subset = subset.loc[totals.nlargest(top_n).index]
    print(f"Selected top {len(subset)} items in {store_id}/{category} "
          f"across {len(day_cols)} days.")

    # Wide -> long: one row per (item, day).
    long_df = subset.melt(
        id_vars=["item_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales_quantity",
    )

    # Attach real dates, events, and the SNAP flag for this store's state.
    state = store_id.split("_")[0]
    snap_col = f"snap_{state}"
    cal_cols = ["d", "date", "wm_yr_wk", "event_name_1", snap_col]
    long_df = long_df.merge(calendar[cal_cols], on="d", how="left")

    # Attach weekly prices. A missing price = product not yet offered.
    long_df = long_df.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    rows_before = len(long_df)
    long_df = long_df.dropna(subset=["sell_price"])
    dropped = rows_before - len(long_df)
    print(f"Dropped {dropped:,} pre-launch rows (no price = item not yet "
          f"sold in this store). Kept {len(long_df):,} rows.")

    final = pd.DataFrame(
        {
            "date": pd.to_datetime(long_df["date"]),
            "product_id": long_df["item_id"],
            "product_name": long_df["item_id"],  # M5 anonymizes names
            "category": long_df["cat_id"],
            "sales_quantity": long_df["sales_quantity"].astype(int),
            "price": long_df["sell_price"].round(2),
            # SNAP disbursement day: a real demand-event flag. Deliberately
            # NOT named "promotion" -- that optional schema column is
            # reserved for genuine marketing promotions.
            "snap_day": long_df[snap_col].astype(int),
            "holiday": long_df["event_name_1"].notna().astype(int),
            # Raw event identity (e.g. Christmas, SuperBowl). NaN on
            # non-event days. Kept unencoded on purpose: encoding is a
            # feature-engineering decision, not a data-prep decision.
            "event_name": long_df["event_name_1"],
            "store_id": long_df["store_id"],
        }
    )
    return final.sort_values(["product_id", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare M5 subset for DemandAI.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--store", default="CA_1", help="M5 store id (default CA_1)")
    parser.add_argument("--category", default="FOODS",
                        choices=["FOODS", "HOBBIES", "HOUSEHOLD"])
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top-selling items to keep (default 50)")
    args = parser.parse_args()

    sales, calendar, prices = load_raw_files(args.raw_dir)
    dataset = build_subset(sales, calendar, prices,
                           store_id=args.store,
                           category=args.category,
                           top_n=args.top_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print("\n=== Dataset written ===")
    print(f"Path        : {args.output}")
    print(f"Rows        : {len(dataset):,}")
    print(f"Products    : {dataset['product_id'].nunique()}")
    print(f"Date range  : {dataset['date'].min().date()} -> "
          f"{dataset['date'].max().date()}")
    print(f"Columns     : {list(dataset.columns)}")


if __name__ == "__main__":
    main()
