"""Compare models on the Phase 5 feature dataset and save the best one.

Thin wrapper: all logic lives in ml/model_comparison.py (which itself
reuses ml/train_model.py utilities).

Usage (from the backend/ directory):
    python scripts/run_model_comparison.py
"""

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml.model_comparison import (  # noqa: E402
    compare_models,
    save_comparison_artifacts,
)
from ml.train_model import TrainingError  # noqa: E402

INPUT_PATH = BACKEND_DIR / "datasets" / "features" / "retail_sales_features.csv"
MODELS_DIR = BACKEND_DIR / "models"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run "
            "scripts/run_feature_engineering.py first."
        )
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    try:
        result = compare_models(df)
    except TrainingError as exc:
        print("MODEL COMPARISON FAILED (validation):")
        print(f"  - {exc}")
        sys.exit(1)

    paths = save_comparison_artifacts(result, MODELS_DIR)
    winner = result.table.set_index("model").loc[result.winner_name]

    print("=== Model comparison report ===")
    print(f"Training rows       : {result.split_sizes['train']:,}   "
          f"({result.split_boundaries['train']})")
    print(f"Validation rows     : {result.split_sizes['validation']:,}   "
          f"({result.split_boundaries['validation']})")
    print(f"Test rows           : {result.split_sizes['test']:,}   "
          f"({result.split_boundaries['test']})")
    print(f"Warm-up NaN rows dropped uniformly for all models: "
          f"{result.dropped_nan_rows:,}")
    print()
    print(result.table.to_string(index=False))
    print()
    print(f"Winner              : {result.winner_name} "
          "(highest validation R2; test never used for selection)")
    print(f"Winner Validation R2: {winner['validation_r2']}")
    print(f"Winner Test R2      : {winner['test_r2']}")
    print(f"Winner training time: {winner['training_time_seconds']} s "
          "(includes tuning for XGBoost)")
    print()
    for key, path in paths.items():
        print(f"Saved [{key:11s}]  : {path}")
    if "best_params" not in paths:
        print("best_params.json not written (XGBoost did not win).")


if __name__ == "__main__":
    main()
