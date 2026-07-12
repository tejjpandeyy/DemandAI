"""Run the preprocessing pipeline on datasets/retail_sales.csv.

Thin wrapper: all logic lives in ml/data_preprocessing.py so the exact
same code path serves the future CSV-upload API endpoint.

Usage (from the backend/ directory):
    python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml.data_preprocessing import PreprocessingError, preprocess_sales_data  # noqa: E402

INPUT_PATH = BACKEND_DIR / "datasets" / "retail_sales.csv"
OUTPUT_PATH = BACKEND_DIR / "datasets" / "retail_sales_clean.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run scripts/prepare_m5_subset.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows from {INPUT_PATH.name}")

    try:
        # Explicit safe defaults:
        #   fill_date_gaps=False -> gaps are detected and reported only;
        #                           no rows inserted, nothing fabricated.
        #   strict=False         -> unresolved leading prices are counted
        #                           and left missing, not fatal (M5 data
        #                           has none; user uploads might).
        clean_df, report = preprocess_sales_data(
            df,
            fill_date_gaps=False,
            flag_outliers=True,
            strict=False,
        )
    except PreprocessingError as exc:
        print("\nPREPROCESSING FAILED (hard validation errors):")
        for err in exc.errors:
            print(f"  - {err}")
        sys.exit(1)

    print()
    print(report.summary())

    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nClean dataset written to {OUTPUT_PATH}")
    print(f"Columns: {list(clean_df.columns)}")
    print("\nSample of flagged outlier rows (original values preserved):")
    flagged = clean_df[clean_df["outlier_flag"] == 1]
    if flagged.empty:
        print("  (none flagged)")
    else:
        cols = [c for c in ("date", "product_id", "sales_quantity",
                            "snap_day", "holiday", "event_name")
                if c in flagged.columns]
        print(flagged.nlargest(5, "sales_quantity")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
