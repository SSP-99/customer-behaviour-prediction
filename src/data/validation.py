"""Automated schema, dtype, null-rate, and date-range validation for transactions."""

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data.loader import SNAKE_CASE_COLUMNS
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when transaction data fails schema, dtype, null-rate, or date checks."""


def validate_transactions(
    df: pd.DataFrame,
    config: AppConfig,
    stage_label: str = "raw",
    require_non_null_customer: bool = False,
) -> dict[str, Any]:
    """Run automated data quality assertions and return a structured validation report.

    Checks enforced:
    1. Non-empty DataFrame and presence of all required snake_case columns.
    2. Numeric dtypes for `quantity` and `price`, datetime dtype for `invoice_date`.
    3. Per-column null-rate thresholds (`max_null_rate` from config).
    4. Transaction dates strictly within [`min_date`, `max_date`] and no future dates.

    Args:
        df: Transactions DataFrame to validate.
        config: Application configuration dataclass.
        stage_label: Descriptive tag ('raw', 'post_cleaning', 'pre_features').
        require_non_null_customer: If True, asserts `customer_id` has zero nulls.

    Returns:
        Validation report dictionary with row count, null rates, and date bounds.

    Raises:
        DataValidationError: If any validation check fails.
    """
    rules = config.cleaning_rules

    if df is None or df.empty:
        raise DataValidationError(f"[{stage_label}] Validation failed: DataFrame is empty.")

    missing_cols = [col for col in SNAKE_CASE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"[{stage_label}] Validation failed: Missing required columns {missing_cols}"
        )

    if not pd.api.types.is_numeric_dtype(df["quantity"]):
        raise DataValidationError(
            f"[{stage_label}] Validation failed: 'quantity' must be numeric, got {df['quantity'].dtype}"
        )
    if not pd.api.types.is_numeric_dtype(df["price"]):
        raise DataValidationError(
            f"[{stage_label}] Validation failed: 'price' must be numeric, got {df['price'].dtype}"
        )
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        raise DataValidationError(
            f"[{stage_label}] Validation failed: 'invoice_date' must be datetime, got {df['invoice_date'].dtype}"
        )

    null_rates: dict[str, float] = {}
    for col in SNAKE_CASE_COLUMNS:
        rate = float(df[col].isna().mean())
        null_rates[col] = round(rate, 6)
        if col == "customer_id" and not require_non_null_customer:
            continue
        if rate > rules.max_null_rate:
            raise DataValidationError(
                f"[{stage_label}] Column '{col}' null rate {rate:.4f} exceeds max_null_rate {rules.max_null_rate:.4f}"
            )

    if require_non_null_customer and df["customer_id"].isna().any():
        raise DataValidationError(
            f"[{stage_label}] Validation failed: null customer_id values found in customer-level table."
        )

    valid_dates = df["invoice_date"].dropna()
    if valid_dates.empty:
        raise DataValidationError(
            f"[{stage_label}] Validation failed: all invoice_date values are null."
        )

    min_ts = valid_dates.min()
    max_ts = valid_dates.max()
    allowed_min = pd.Timestamp(rules.min_date)
    allowed_max = pd.Timestamp(rules.max_date)
    now_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))

    if min_ts < allowed_min:
        raise DataValidationError(
            f"[{stage_label}] Earliest invoice_date {min_ts} precedes allowed min_date {allowed_min}"
        )
    if max_ts > allowed_max:
        raise DataValidationError(
            f"[{stage_label}] Latest invoice_date {max_ts} exceeds allowed max_date {allowed_max}"
        )
    if max_ts > now_utc:
        raise DataValidationError(
            f"[{stage_label}] Future invoice_date detected: {max_ts} > {now_utc}"
        )

    report: dict[str, Any] = {
        "stage": stage_label,
        "status": "PASSED",
        "row_count": int(len(df)),
        "min_invoice_date": min_ts.isoformat(),
        "max_invoice_date": max_ts.isoformat(),
        "null_rates": null_rates,
    }
    logger.info(
        "Validation [%s] PASSED | rows=%d | range=[%s .. %s]",
        stage_label,
        len(df),
        min_ts,
        max_ts,
    )
    return report
