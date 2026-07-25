"""Prediction service for DemandAI (Phase 8).

Framework-free (joblib / numpy / pandas only): no FastAPI imports here,
so the same service is directly unit-testable and reusable by later
phases (e.g. the Phase 9 forecasting pipeline).

Responsibilities:
    - load the Phase 7 best-model BUNDLE exactly once (load_count is
      tracked so tests can prove single loading);
    - build a model-ready feature row from a request: calendar features
      are DERIVED from the request date, direct fields (price, snap_day,
      holiday, has_named_event) are copied, engineered history features
      come from the request's features list, and anything else is NaN;
    - reject unknown feature names with a clear message;
    - predict single requests and batches (batches are built as ONE
      DataFrame and scored with ONE model call -- vectorized).

Predictions are clipped at zero: negative demand is not physically
meaningful, and tree regressors can slightly undershoot near zero.
"""

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "best_model.joblib"

REQUIRED_BUNDLE_KEYS: tuple[str, ...] = ("model", "feature_names",
                                         "model_type")

# Request fields copied directly into the feature row when the model
# uses them.
DIRECT_FIELDS: tuple[str, ...] = ("price", "snap_day", "holiday",
                                  "has_named_event")


def _calendar_values(ts: pd.Timestamp) -> dict[str, float]:
    """Deterministic calendar features derived from the request date.

    Mirrors Phase 5's add_calendar_features definitions exactly
    (day_of_week: 0 = Monday, etc.).
    """
    return {
        "year": ts.year,
        "month": ts.month,
        "quarter": ts.quarter,
        "week_of_year": int(ts.isocalendar().week),
        "day_of_month": ts.day,
        "day_of_week": ts.dayofweek,
        "day_of_year": ts.dayofyear,
        "is_weekend": int(ts.dayofweek >= 5),
        "is_month_start": int(ts.day == 1),
        "is_month_end": int(ts.day == ts.days_in_month),
    }


class PredictionError(Exception):
    """Raised for request-level problems (maps to HTTP 422 in the API)."""


class ModelNotLoadedError(Exception):
    """Raised when prediction is attempted before load_model()."""


class ModelService:
    """Holds one loaded model bundle and serves predictions from it."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model: Any = None
        self.feature_names: list[str] = []
        self.model_type: str = ""
        self.load_count: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> "ModelService":
        """Load the model bundle from disk. Called ONCE at app startup."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {self.model_path}. Run "
                "scripts/run_model_comparison.py (Phase 7) first."
            )
        bundle = joblib.load(self.model_path)
        missing = [k for k in REQUIRED_BUNDLE_KEYS if k not in bundle]
        if missing:
            raise ValueError(
                f"Model bundle at {self.model_path} is missing keys: "
                f"{missing}."
            )
        self.model = bundle["model"]
        self.feature_names = list(bundle["feature_names"])
        self.model_type = str(bundle["model_type"])
        self.load_count += 1
        return self

    def _require_loaded(self) -> None:
        if not self.loaded:
            raise ModelNotLoadedError(
                "Model is not loaded; call load_model() first."
            )

    # ------------------------------------------------------------------
    # Feature-row construction
    # ------------------------------------------------------------------

    def _build_row(self, request: dict[str, Any]) -> dict[str, float]:
        """Map one request dict onto the model's feature vector.

        Priority: derived calendar values and direct fields fill their
        slots; the request's features list overlays engineered history
        features; every remaining model feature is NaN (missing).
        Unknown feature names are rejected.
        """
        ts = pd.Timestamp(request["date"])
        calendar = _calendar_values(ts)
        row: dict[str, float] = {}
        for name in self.feature_names:
            if name in calendar:
                row[name] = calendar[name]
            elif name in DIRECT_FIELDS and name in request:
                row[name] = float(request.get(name) or 0)
            else:
                row[name] = np.nan

        known = set(self.feature_names)
        for feature in request.get("features", []) or []:
            name = feature["name"] if isinstance(feature, dict) \
                else feature.name
            value = feature["value"] if isinstance(feature, dict) \
                else feature.value
            if name not in known:
                raise PredictionError(
                    f"unknown feature name {name!r}; the model was "
                    f"trained on: {sorted(known)}"
                )
            row[name] = np.nan if value is None else float(value)
        return row

    def _frame(self, requests: list[dict[str, Any]]) -> pd.DataFrame:
        rows = [self._build_row(r) for r in requests]
        return pd.DataFrame(rows, columns=self.feature_names)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_single(self, request: dict[str, Any]) -> dict[str, Any]:
        """Predict one request. Returns predicted_sales and timing."""
        self._require_loaded()
        start = time.perf_counter()
        frame = self._frame([request])
        raw = float(self.model.predict(frame)[0])
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "product_id": request["product_id"],
            "date": request["date"],
            "predicted_sales": round(max(raw, 0.0), 3),
            "processing_time_ms": round(elapsed_ms, 2),
            "model_type": self.model_type,
        }

    def predict_batch(self, requests: list[dict[str, Any]]
                      ) -> dict[str, Any]:
        """Predict a batch with ONE vectorized model call."""
        self._require_loaded()
        if not requests:
            raise PredictionError("batch is empty")
        start = time.perf_counter()
        frame = self._frame(requests)
        raw = np.asarray(self.model.predict(frame), dtype=float)
        clipped = np.clip(raw, 0.0, None).round(3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "count": len(requests),
            "processing_time_ms": round(elapsed_ms, 2),
            "predictions": [
                {
                    "product_id": r["product_id"],
                    "date": r["date"],
                    "predicted_sales": float(p),
                }
                for r, p in zip(requests, clipped)
            ],
        }
