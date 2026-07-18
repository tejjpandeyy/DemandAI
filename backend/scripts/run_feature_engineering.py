"""Run leakage-safe feature engineering on retail_sales_clean.csv.

Thin wrapper: all logic lives in ml/feature_engineering.py. Prints the
Phase 5 feature report and writes the output to
backend/datasets/features/retail_sales_features.csv (Git-ignored under
the existing datasets/* rule).

Usage (from the backend/ directory):
    python scripts/run_feature_engineering.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml.feature_engineering import (  # noqa: E402
    FeatureEngineeringError,
    build_features,
)

INPUT_PATH = BACKEND_DIR / "datasets" / "retail_sales_clean.csv"
OUTPUT_DIR = BACKEND_DIR / "datasets" / "features"
OUTPUT_PATH = OUTPUT_DIR / "retail_sales_features.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run scripts/run_preprocessing.py first."
        )
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    start = time.perf_counter()
    try:
        features, generated = build_features(df)
    except FeatureEngineeringError as exc:
        print("FEATURE ENGINEERING FAILED (validation):")
        print(f"  - {exc}")
        sys.exit(1)
    elapsed = time.perf_counter() - start

    lag_cols = sorted(
        (c for c in generated if c.startswith("lag_")),
        key=lambda c: int(c.split("_")[1]),
    )
    rolling_cols = [c for c in generated if c.startswith("rolling_")
                    or c == "expanding_mean"]

    print("=== Feature engineering report ===")
    print(f"Input rows          : {len(df):,}")
    print(f"Output rows         : {len(features):,}   "
          "(unchanged -- warm-up NaNs preserved, no rows dropped)")
    print(f"Input columns       : {len(df.columns)}")
    print(f"Output columns      : {len(features.columns)}")
    print(f"Generated features  : {len(generated)}")
    print(f"Execution time      : {elapsed:.2f} s")
    print(f"Memory usage        : "
          f"{features.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\nNaN counts -- lag features (expected: n_series * lag):")
    for col in lag_cols:
        print(f"  {col:22s}: {int(features[col].isna().sum()):,}")

    print("\nNaN counts -- rolling / expanding features:")
    for col in rolling_cols:
        print(f"  {col:22s}: {int(features[col].isna().sum()):,}")

    price_cols = [c for c in generated if "price" in c]
    if price_cols:
        print("\nNaN counts -- price features:")
        for col in price_cols:
            print(f"  {col:22s}: {int(features[col].isna().sum()):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeatures written to {OUTPUT_PATH}")
    print(f"All generated features: {generated}")


if __name__ == "__main__":
    main()
