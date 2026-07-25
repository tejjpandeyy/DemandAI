"""Pydantic schemas for the DemandAI prediction API (Phase 8).

Validation policy:
    - date must be a real ISO calendar date (YYYY-MM-DD);
    - price must be positive (negative prices rejected);
    - engineered features are passed as a LIST of {name, value} pairs,
      because JSON objects silently collapse duplicate keys -- a list is
      the only way duplicate feature names can be detected and rejected;
    - batch requests must contain at least one record.
"""

from datetime import date as date_type
import re

from pydantic import BaseModel, Field, field_validator, model_validator

# Strict dashed ISO shape. Required because Python >= 3.11
# date.fromisoformat() also accepts the COMPACT form ("20160301"),
# which this API must reject.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FeatureValue(BaseModel):
    """One engineered feature value (e.g. lag_7 = 9.0)."""

    name: str = Field(min_length=1, description="Feature name, e.g. lag_1")
    value: float | None = Field(
        default=None,
        description="Feature value; null means 'unknown' (treated as NaN).",
    )


class PredictionRequest(BaseModel):
    """One demand-prediction request.

    Calendar features are derived server-side from ``date``. History-
    dependent features (lag_*, rolling_*, expanding_mean, price_change,
    ...) are supplied through ``features``; any omitted ones are treated
    as missing (NaN).
    """

    date: str = Field(description="Prediction date, ISO format YYYY-MM-DD")
    product_id: str = Field(min_length=1)
    price: float = Field(
        gt=0, description="Unit price; must be positive."
    )
    snap_day: int = Field(default=0, ge=0, le=1)
    holiday: int = Field(default=0, ge=0, le=1)
    has_named_event: int = Field(default=0, ge=0, le=1)
    features: list[FeatureValue] = Field(
        default_factory=list,
        description="Engineered history features as {name, value} pairs.",
    )

    @field_validator("date")
    @classmethod
    def date_must_be_valid_iso(cls, v: str) -> str:
        # 1. Shape: exactly YYYY-MM-DD (rejects "20160301", "March 1st").
        if not isinstance(v, str) or not _ISO_DATE_RE.match(v):
            raise ValueError(
                f"invalid date {v!r}; expected strict YYYY-MM-DD format "
                "(e.g. 2016-03-01)"
            )
        # 2. Calendar validity: rejects impossible dates like 2016-13-45.
        try:
            date_type.fromisoformat(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"invalid date {v!r}; not a real calendar date"
            ) from exc
        return v

    @model_validator(mode="after")
    def feature_names_must_be_unique(self) -> "PredictionRequest":
        names = [f.name for f in self.features]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate feature names not allowed: {duplicates}"
            )
        return self


class BatchPredictionRequest(BaseModel):
    """A batch of prediction requests (must not be empty)."""

    requests: list[PredictionRequest] = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    product_id: str
    date: str
    predicted_sales: float
    processing_time_ms: float
    model_type: str


class BatchPredictionItem(BaseModel):
    """One prediction inside a batch response."""

    product_id: str
    date: str
    predicted_sales: float


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction."""

    count: int
    processing_time_ms: float
    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class RootResponse(BaseModel):
    service: str
    status: str
