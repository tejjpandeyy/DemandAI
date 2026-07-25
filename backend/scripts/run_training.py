"""Train the baseline model on the Phase 5 feature dataset.

Thin wrapper: all logic lives in ml/train_model.py. Prints the Phase 6
training report and saves artifacts to backend/models/.

Usage (from the backend/ directory):
    python scripts/run_training.py
"""

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml.train_model import TrainingError, save_artifacts, train_baseline  # noqa: E402

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
        result = train_baseline(df)
    except TrainingError as exc:
        print("TRAINING FAILED (validation):")
        print(f"  - {exc}")
        sys.exit(1)

    paths = save_artifacts(result, MODELS_DIR)

    print("=== Baseline training report ===")
    print(f"Training rows       : {result.split_sizes['train']:,}   "
          f"({result.split_boundaries['train']})")
    print(f"Validation rows     : {result.split_sizes['validation']:,}   "
          f"({result.split_boundaries['validation']})")
    print(f"Test rows           : {result.split_sizes['test']:,}   "
          f"({result.split_boundaries['test']})")
    print(f"Features used       : {len(result.feature_names)}")
    print(f"Training time       : {result.train_seconds:.2f} s")
    print()
    for split in ("validation", "test"):
        m = result.metrics[split]
        label = split.capitalize()
        print(f"{label} MAE".ljust(20) + f": {m['mae']}")
        print(f"{label} RMSE".ljust(20) + f": {m['rmse']}")
        print(f"{label} R2".ljust(20) + f": {m['r2']}")
        print()
    print("Top 10 important features:")
    top = result.importance.head(10)
    total = result.importance["importance"].sum()
    for _, row in top.iterrows():
        share = row["importance"] / total * 100 if total else 0.0
        print(f"  {row['feature']:24s} {share:5.1f}%")
    print()
    print(f"Model saved to      : {paths['model']}")
    print(f"Importance saved to : {paths['importance']}")
    print(f"Predictions saved to: {paths['predictions']}")


if __name__ == "__main__":
    main()
