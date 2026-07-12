"""Dataset validation for DemandAI.

A single, reusable validator used in two places:
    1. Command-line inspection during development (Phase 2).
    2. The CSV upload API endpoint (Phase 12), so users get meaningful
       errors instead of a stack trace.

Design: validation never raises for bad DATA (only for programmer misuse);
instead it returns a ValidationResult describing everything found, split
into hard errors (dataset unusable) and warnings (usable but noteworthy).
"""

from dataclasses import dataclass, field

import pandas as pd

# Columns the ML pipeline cannot work without.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "product_id",
    "sales_quantity",
    "store_id",
)

# Columns used when present; their absence only disables related features.
# Event flags (promotion, snap_day, holiday) are handled generically by the
# feature pipeline: whichever exist are used, none is individually required.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "product_name",
    "category",
    "price",
    "promotion",   # genuine marketing promotion, if the dataset has one
    "snap_day",    # SNAP disbursement flag (M5-derived datasets)
    "holiday",
    "event_name",  # raw event identity (Christmas, SuperBowl, ...)
    "discount",
    "current_stock",
)

# Columns that are sparse BY NATURE (mostly empty is their normal state),
# so the missing-value warning would only produce noise for them.
SPARSE_BY_DESIGN: tuple[str, ...] = ("event_name",)

# Warn if more than this fraction of a column is missing.
MISSING_WARN_THRESHOLD: float = 0.05


@dataclass
class ValidationResult:
    """Outcome of validating an uploaded/loaded sales dataset."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when the dataset has no hard errors."""
        return not self.errors

    def summary(self) -> str:
        """Human-readable multi-line report."""
        lines: list[str] = []
        status = "VALID" if self.is_valid else "INVALID"
        lines.append(f"Validation status: {status}")
        for err in self.errors:
            lines.append(f"  [ERROR]   {err}")
        for warn in self.warnings:
            lines.append(f"  [WARNING] {warn}")
        if self.is_valid and not self.warnings:
            lines.append("  No issues found.")
        return "\n".join(lines)


def validate_sales_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate a raw sales DataFrame against the DemandAI schema.

    Args:
        df: DataFrame as loaded from a CSV (dates may still be strings).

    Returns:
        ValidationResult with errors (dataset unusable) and warnings.
    """
    result = ValidationResult()

    if df.empty:
        result.errors.append("Dataset is empty (0 rows).")
        return result

    # --- 1. Required columns present? ---
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.errors.append(
            f"Missing required columns: {missing}. "
            f"Required columns are: {list(REQUIRED_COLUMNS)}."
        )
        return result  # further checks depend on these columns

    # --- 2. Dates parseable? ---
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = int(parsed_dates.isna().sum())
    if n_bad_dates:
        result.errors.append(
            f"{n_bad_dates:,} rows have unparseable dates in the 'date' "
            "column (expected format like YYYY-MM-DD)."
        )

    # --- 3. Sales quantity numeric and non-negative? ---
    sales_numeric = pd.to_numeric(df["sales_quantity"], errors="coerce")
    n_non_numeric = int(sales_numeric.isna().sum() - df["sales_quantity"].isna().sum())
    if n_non_numeric > 0:
        result.errors.append(
            f"{n_non_numeric:,} rows have non-numeric values in "
            "'sales_quantity'."
        )
    n_negative = int((sales_numeric < 0).sum())
    if n_negative:
        result.errors.append(
            f"{n_negative:,} rows have negative 'sales_quantity'. Returns "
            "should be handled upstream, not stored as negative demand."
        )

    # --- 4. Price sanity (optional column) ---
    if "price" in df.columns:
        price_numeric = pd.to_numeric(df["price"], errors="coerce")
        n_bad_price = int((price_numeric <= 0).sum())
        if n_bad_price:
            result.warnings.append(
                f"{n_bad_price:,} rows have zero/negative price."
            )

    # --- 5. Duplicate (date, product, store) rows? ---
    key_cols = ["date", "product_id", "store_id"]
    n_dupes = int(df.duplicated(subset=key_cols).sum())
    if n_dupes:
        result.errors.append(
            f"{n_dupes:,} duplicate rows for the same "
            "(date, product_id, store_id). Each product-store-day must "
            "appear exactly once."
        )

    # --- 6. Missing-value report ---
    for col in df.columns:
        if col in SPARSE_BY_DESIGN:
            continue  # e.g. event_name is empty on most days by definition
        frac = float(df[col].isna().mean())
        if frac > MISSING_WARN_THRESHOLD:
            result.warnings.append(
                f"Column '{col}' is {frac:.1%} missing."
            )

    # --- 7. Optional columns absent? (informational only) ---
    absent_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if absent_optional:
        result.warnings.append(
            f"Optional columns not provided: {absent_optional}. "
            "Related features will be skipped."
        )

    # --- 8. Date-continuity check per product (gaps in the time series) ---
    if not n_bad_dates:
        tmp = df.assign(_date=parsed_dates)
        gaps = 0
        for _, group in tmp.groupby(["product_id", "store_id"], sort=False):
            expected = (group["_date"].max() - group["_date"].min()).days + 1
            gaps += expected - group["_date"].nunique()
        if gaps > 0:
            result.warnings.append(
                f"Time series contain {gaps:,} missing days in total "
                "(gaps between each product's first and last date). "
                "Preprocessing (Phase 3) will need to handle these."
            )

    return result
