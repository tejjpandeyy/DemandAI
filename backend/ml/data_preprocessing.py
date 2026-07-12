"""Data preprocessing for DemandAI.

Reusable cleaning pipeline used by BOTH offline training scripts and the
future FastAPI CSV-upload endpoint.

Core invariant: THE PIPELINE NEVER MANUFACTURES OBSERVATIONS.
    - Date gaps are DETECTED and REPORTED by default; no rows are
      inserted, no synthetic sales_quantity=0 is created, no calendar
      or event values are fabricated for days that were never observed.
    - Prices are repaired ONLY by forward fill (past information) within
      each product_id + store_id series. No backward filling: a price
      from the future is never copied into the past. Prices that cannot
      be resolved from the past are counted (unresolved_price_count) and,
      in strict mode, cause a hard failure.
    - Demand outliers are FLAGGED (outlier_flag column), never removed
      or capped; original sales_quantity is always preserved.

Three explicit tiers of data issues:

    1. HARD FAILURES  -> raise PreprocessingError (no safe automatic fix):
         - missing required columns
         - negative sales_quantity
         - conflicting duplicates (same product/store/date, different values)
         - unparseable dates above MAX_BAD_DATE_FRACTION
         - unresolved prices when strict=True
    2. SAFE DETERMINISTIC CLEANING -> fix and count in the report:
         - exact duplicate rows (identical in every column)
         - a small number of unparseable dates (dropped)
         - invalid (<= 0) or missing prices (forward-filled per product,
           past information only)
         - missing event_name normalized to the explicit "none" category
           (an expected meaning, not a data-quality error)
    3. ANALYTICAL WARNINGS -> report only, data untouched:
         - date-gap detection per product series
         - outlier flagging

Opt-in structural reindexing: ``fill_date_gaps=True`` inserts rows for
missing dates marked ``is_gap_fill=1`` whose observation columns
(sales_quantity, price, event flags, event_name) are left as MISSING
(pd.NA / NaN) -- structural continuity without fabricating values. The
default is False.

Main entry point: :func:`preprocess_sales_data`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml.data_validation import REQUIRED_COLUMNS

# Explicit category used for "no event on this day" (observed days only).
NO_EVENT: str = "none"

# If more than this fraction of rows have unparseable dates, the file is
# considered structurally broken (wrong column / wrong format) -> hard fail.
MAX_BAD_DATE_FRACTION: float = 0.05

# Per-product outlier threshold: sales > Q3 + IQR_MULTIPLIER * IQR.
# 3.0 is deliberately conservative ("far out" fences), because in demand
# data moderate spikes are usually legitimate.
IQR_MULTIPLIER: float = 3.0

# Binary event-flag columns handled generically when present.
EVENT_FLAG_COLUMNS: tuple[str, ...] = ("snap_day", "holiday", "promotion")

# Identity attributes of a product (time-invariant); safe to carry forward
# onto inserted structural rows in opt-in mode.
STATIC_COLUMNS: tuple[str, ...] = ("product_name", "category")


class PreprocessingError(Exception):
    """Raised when the dataset has problems with no safe automatic fix.

    Carries a list of human-readable error strings so the API layer can
    return them directly to the user (HTTP 422 in Phase 12).
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class PreprocessingReport:
    """Everything that happened during preprocessing, for audit/UI display."""

    input_rows: int = 0
    output_rows: int = 0
    exact_duplicates_removed: int = 0
    invalid_date_rows_dropped: int = 0
    negative_sales_rows: int = 0
    invalid_price_count: int = 0          # prices <= 0 set to missing
    missing_price_count: int = 0          # missing prices before repair
    prices_imputed: int = 0               # repaired via forward fill only
    unresolved_price_count: int = 0       # no past price available
    zero_sales_pct: float = 0.0           # over real (non-gap-fill) rows
    outlier_flag_count: int = 0
    outliers_on_event_days: int = 0
    products_with_date_gaps: int = 0
    total_missing_days: int = 0
    gap_rows_inserted: int = 0            # 0 unless fill_date_gaps=True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for the future API layer."""
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "invalid_date_rows_dropped": self.invalid_date_rows_dropped,
            "negative_sales_rows": self.negative_sales_rows,
            "invalid_price_count": self.invalid_price_count,
            "missing_price_count": self.missing_price_count,
            "prices_imputed": self.prices_imputed,
            "unresolved_price_count": self.unresolved_price_count,
            "zero_sales_pct": round(self.zero_sales_pct, 4),
            "outlier_flag_count": self.outlier_flag_count,
            "outliers_on_event_days": self.outliers_on_event_days,
            "products_with_date_gaps": self.products_with_date_gaps,
            "total_missing_days": self.total_missing_days,
            "gap_rows_inserted": self.gap_rows_inserted,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Human-readable multi-line report."""
        lines = [
            "=== Preprocessing report ===",
            f"Input rows              : {self.input_rows:,}",
            f"Output rows             : {self.output_rows:,}",
            f"Exact duplicates removed: {self.exact_duplicates_removed:,}",
            f"Invalid dates dropped   : {self.invalid_date_rows_dropped:,}",
            f"Negative sales rows     : {self.negative_sales_rows:,}",
            f"Invalid prices (<=0)    : {self.invalid_price_count:,}",
            f"Missing prices          : {self.missing_price_count:,}",
            f"Prices imputed (ffill)  : {self.prices_imputed:,}",
            f"Unresolved prices       : {self.unresolved_price_count:,}",
            f"Zero-sales share        : {self.zero_sales_pct:.1%}",
            f"Outliers FLAGGED        : {self.outlier_flag_count:,} "
            f"(of which {self.outliers_on_event_days:,} on event days)",
            f"Products with date gaps : {self.products_with_date_gaps:,}",
            f"Total missing days      : {self.total_missing_days:,} "
            "(detected, NOT filled)" if self.gap_rows_inserted == 0
            else f"Total missing days      : {self.total_missing_days:,}",
            f"Gap rows inserted       : {self.gap_rows_inserted:,}",
        ]
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 1 + 2 helpers
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame) -> None:
    """Hard-fail if any required column is absent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PreprocessingError(
            [f"Missing required columns: {missing}. "
             f"Required: {list(REQUIRED_COLUMNS)}."]
        )


def _parse_dates(df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
    """Coerce the date column; drop a few bad rows, hard-fail on many."""
    parsed = pd.to_datetime(df["date"], errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        frac = bad / len(df)
        if frac > MAX_BAD_DATE_FRACTION:
            raise PreprocessingError(
                [f"{bad:,} rows ({frac:.1%}) have unparseable dates -- "
                 "above the safe-repair threshold "
                 f"({MAX_BAD_DATE_FRACTION:.0%}). Check the date column "
                 "format (expected YYYY-MM-DD)."]
            )
        report.invalid_date_rows_dropped = bad
        report.warnings.append(f"Dropped {bad:,} rows with unparseable dates.")
    df = df.assign(date=parsed).dropna(subset=["date"])
    return df


def _check_negative_sales(df: pd.DataFrame,
                          report: PreprocessingReport) -> None:
    """Hard-fail on negative sales: their meaning (returns?) is ambiguous."""
    sales = pd.to_numeric(df["sales_quantity"], errors="coerce")
    non_numeric = int(sales.isna().sum())
    if non_numeric:
        raise PreprocessingError(
            [f"{non_numeric:,} rows have non-numeric sales_quantity."]
        )
    negative = int((sales < 0).sum())
    report.negative_sales_rows = negative
    if negative:
        raise PreprocessingError(
            [f"{negative:,} rows have negative sales_quantity. Negative "
             "values may be returns or corrections; resolve them upstream "
             "-- silently clipping or dropping would fabricate demand."]
        )


def _handle_duplicates(df: pd.DataFrame,
                       report: PreprocessingReport) -> pd.DataFrame:
    """Drop exact duplicates (safe); hard-fail on conflicting duplicates.

    Exact duplicate = identical in EVERY column (e.g. a file accidentally
    uploaded twice): dropping loses no information.
    Conflicting duplicate = same (date, product_id, store_id) key but
    different values: we cannot know which row is true, so we refuse to
    guess.
    """
    before = len(df)
    df = df.drop_duplicates()
    report.exact_duplicates_removed = before - len(df)

    key = ["date", "product_id", "store_id"]
    conflicts = df.duplicated(subset=key, keep=False)
    if conflicts.any():
        sample = (
            df.loc[conflicts, key].drop_duplicates().head(3).to_dict("records")
        )
        raise PreprocessingError(
            [f"{int(conflicts.sum()):,} rows share a (date, product_id, "
             f"store_id) key with CONFLICTING values, e.g. {sample}. "
             "Cannot determine which row is correct; fix the source data."]
        )
    return df


def _clean_prices(df: pd.DataFrame, report: PreprocessingReport,
                  strict: bool) -> pd.DataFrame:
    """Repair invalid/missing prices using PAST information only.

    Price is a slowly-changing weekly attribute, so carrying the last
    known price FORWARD is a faithful, leakage-free repair. Backward
    filling is deliberately absent: copying a future price into the past
    would inject information that did not exist at that time.

    Rows before a product's first valid price therefore stay missing.
    They are counted in ``unresolved_price_count``; with ``strict=True``
    they raise :class:`PreprocessingError` instead.
    """
    if "price" not in df.columns:
        report.warnings.append("No 'price' column; price checks skipped.")
        return df

    price = pd.to_numeric(df["price"], errors="coerce")
    invalid = int((price <= 0).sum())
    report.invalid_price_count = invalid
    price = price.mask(price <= 0)
    report.missing_price_count = int(price.isna().sum())

    if report.missing_price_count:
        # Forward fill only, within each product-store series.
        # Requires chronological order WITHIN product -- caller sorts first.
        price = df.assign(price=price).groupby(
            ["product_id", "store_id"], sort=False
        )["price"].ffill()

        unresolved = int(price.isna().sum())
        report.unresolved_price_count = unresolved
        report.prices_imputed = report.missing_price_count - unresolved

        if report.prices_imputed:
            report.warnings.append(
                f"Imputed {report.prices_imputed:,} prices via per-product "
                "forward-fill (past information only)."
            )
        if unresolved:
            msg = (
                f"{unresolved:,} rows have no resolvable price (missing "
                "before the product's first observed price; backward "
                "filling from the future is not performed)."
            )
            if strict:
                raise PreprocessingError([msg])
            report.warnings.append(msg + " Left missing.")
    return df.assign(price=price)


def _normalize_event_name(df: pd.DataFrame) -> pd.DataFrame:
    """Represent 'no event' explicitly instead of leaving NaN.

    Missing event_name on OBSERVED days is EXPECTED (most days have no
    event) -- it is a meaning, not a data-quality problem, so it gets an
    explicit category rather than being treated as missing data.
    """
    if "event_name" in df.columns:
        df = df.assign(event_name=df["event_name"].fillna(NO_EVENT))
    return df


# ---------------------------------------------------------------------------
# Date gaps: detection (default) and opt-in structural reindexing
# ---------------------------------------------------------------------------

def _detect_date_gaps(df: pd.DataFrame, report: PreprocessingReport) -> None:
    """Detect and REPORT missing days per product series. Inserts nothing.

    A gap = a calendar day between a series' first and last observed
    date that has no row. We do not know why it is missing (store
    closure? export error? out of assortment?), so we refuse to invent
    an observation for it.
    """
    grouped = df.groupby(["product_id", "store_id"])["date"]
    stats = grouped.agg(first="min", last="max", observed="nunique")
    expected = (stats["last"] - stats["first"]).dt.days + 1
    missing = (expected - stats["observed"]).astype(int)
    with_gaps = missing[missing > 0]

    report.products_with_date_gaps = int(len(with_gaps))
    report.total_missing_days = int(with_gaps.sum())
    if len(with_gaps):
        worst = with_gaps.sort_values(ascending=False).head(3)
        worst_desc = ", ".join(
            f"{idx[0]}@{idx[1]}: {n} days" for idx, n in worst.items()
        )
        report.warnings.append(
            f"Detected {report.total_missing_days:,} missing days across "
            f"{report.products_with_date_gaps} product series (worst: "
            f"{worst_desc}). No rows were inserted; downstream features "
            "must be date-aware."
        )


def _insert_structural_gap_rows(df: pd.DataFrame,
                                report: PreprocessingReport) -> pd.DataFrame:
    """OPT-IN ONLY: reindex each series to a continuous daily calendar.

    Inserted rows are marked ``is_gap_fill=1`` and their observation
    columns (sales_quantity, price, event flags, event_name) are left
    MISSING (pd.NA / NaN). Only identity attributes (product_name,
    category) are carried forward, since they are time-invariant facts
    about the product, not observations. Nothing is fabricated: a gap
    row says "this day existed, and we observed nothing".
    """
    pieces: list[pd.DataFrame] = []
    inserted_total = 0

    for (pid, sid), group in df.groupby(["product_id", "store_id"],
                                        sort=False):
        group = group.set_index("date").sort_index()
        full_index = pd.date_range(group.index.min(), group.index.max(),
                                   freq="D")
        n_insert = len(full_index) - len(group)
        if n_insert == 0:
            group["is_gap_fill"] = 0
            pieces.append(group.reset_index())
            continue

        inserted_total += n_insert
        group = group.reindex(full_index)
        group.index.name = "date"
        inserted = group["product_id"].isna()
        group["is_gap_fill"] = inserted.astype(int)
        group["product_id"] = pid
        group["store_id"] = sid
        # The reindexed range starts at the first OBSERVED date, so ffill
        # alone fully populates time-invariant identity attributes.
        for col in STATIC_COLUMNS:
            if col in group.columns:
                group[col] = group[col].ffill()
        # Observation columns stay missing on inserted rows -- deliberate.
        pieces.append(group.reset_index())

    report.gap_rows_inserted = inserted_total
    if inserted_total:
        report.warnings.append(
            f"Inserted {inserted_total:,} STRUCTURAL gap rows "
            "(is_gap_fill=1) with observation columns left missing "
            "(opt-in fill_date_gaps=True)."
        )
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Tier 3: analysis (never modifies the data)
# ---------------------------------------------------------------------------

def flag_demand_outliers(df: pd.DataFrame,
                         iqr_multiplier: float = IQR_MULTIPLIER) -> pd.Series:
    """Flag unusually high demand per product WITHOUT altering it.

    Rule: sales > Q3 + iqr_multiplier * IQR, computed per product so a
    big seller's normal day is not compared against a small seller's.
    If a product's IQR is 0 (very intermittent demand), fall back to
    mean + 4*std; if std is also 0, nothing can be called an outlier.
    Missing sales (structural gap rows in opt-in mode) are never flagged.

    Returns:
        Int series (0/1) aligned to ``df.index``.
    """
    flags = pd.Series(0, index=df.index, dtype=int)
    for _, group in df.groupby(["product_id", "store_id"], sort=False):
        sales = group["sales_quantity"].astype(float)
        q1, q3 = sales.quantile(0.25), sales.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            upper = q3 + iqr_multiplier * iqr
        else:
            std = sales.std()
            if not std or np.isnan(std):
                continue
            upper = float(sales.mean()) + 4 * std
        flags.loc[group.index[sales > upper]] = 1
    return flags


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_sales_data(
    df: pd.DataFrame,
    fill_date_gaps: bool = False,
    flag_outliers: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Clean a raw sales DataFrame into analysis-ready form.

    Never manufactures observations: by default, date gaps are detected
    and reported only, and prices are repaired exclusively by forward
    fill (past information) within each product-store series.

    Args:
        df: Raw sales data (as loaded from CSV; dates may be strings).
        fill_date_gaps: OPT-IN structural reindexing. When True, missing
            days get rows marked is_gap_fill=1 whose observation columns
            are left MISSING (never synthetic zeros). Default False:
            detect and report only, output row count unchanged.
        flag_outliers: Add an outlier_flag column (analysis only; the
            original sales_quantity is never modified).
        strict: When True, prices that cannot be resolved from past
            information raise PreprocessingError instead of being left
            missing.

    Returns:
        (clean_df, report) -- clean_df sorted by product_id, store_id,
        date, with original sales values preserved.

    Raises:
        PreprocessingError: On hard failures (see module docstring).
    """
    report = PreprocessingReport(input_rows=len(df))
    if df.empty:
        raise PreprocessingError(["Dataset is empty (0 rows)."])

    df = df.copy()

    # --- Tier 1: hard validation ---
    _check_required_columns(df)
    df = _parse_dates(df, report)
    _check_negative_sales(df, report)
    df = _handle_duplicates(df, report)

    # --- Chronological order (required by ffill and by Phase 5) ---
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(
        drop=True
    )

    # --- Tier 2: safe deterministic cleaning ---
    df = df.assign(
        sales_quantity=pd.to_numeric(df["sales_quantity"]).astype(int)
    )
    df = _clean_prices(df, report, strict=strict)
    df = _normalize_event_name(df)
    for col in EVENT_FLAG_COLUMNS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            )

    # --- Date gaps: report by default; structural reindex only on opt-in ---
    _detect_date_gaps(df, report)
    if fill_date_gaps:
        df = _insert_structural_gap_rows(df, report)
        # Nullable Int64 keeps integers while representing "unobserved".
        df["sales_quantity"] = df["sales_quantity"].astype("Int64")
        for col in EVENT_FLAG_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("Int64")
    else:
        df["is_gap_fill"] = 0

    # --- Tier 3: analytical flags and statistics ---
    if flag_outliers:
        df["outlier_flag"] = flag_demand_outliers(df)
        report.outlier_flag_count = int(df["outlier_flag"].sum())
        event_cols = [c for c in EVENT_FLAG_COLUMNS if c in df.columns]
        if event_cols and report.outlier_flag_count:
            on_event = (
                df.loc[df["outlier_flag"] == 1, event_cols]
                .fillna(0).max(axis=1)
            )
            report.outliers_on_event_days = int(on_event.sum())
    else:
        df["outlier_flag"] = 0

    real_rows = df[df["is_gap_fill"] == 0]
    report.zero_sales_pct = float((real_rows["sales_quantity"] == 0).mean())
    report.output_rows = len(df)

    # Final chronological guarantee.
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(
        drop=True
    )
    return df, report
