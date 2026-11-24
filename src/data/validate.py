"""Data validation checks for raw and cleaned retail transactions."""

import pandas as pd

from src.data.ingest import CANONICAL_COLUMNS
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when raw or cleaned data violates schema or quality constraints."""


def validate_raw_schema(df: pd.DataFrame, config: AppConfig) -> dict[str, float]:
    """Verify required columns exist, dataset is non-empty, and null rates are within limits.

    Args:
        df: Raw transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        Dictionary mapping column names to their null fraction.

    Raises:
        DataValidationError: If required columns are missing, DataFrame is empty,
            or any non-customer column exceeds `max_null_rate`.
    """
    if df.empty:
        raise DataValidationError("Raw transactions DataFrame is empty.")

    missing_cols = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    null_rates = (df[CANONICAL_COLUMNS].isna().mean()).to_dict()
    max_allowed = config.cleaning_rules.max_null_rate

    for col, rate in null_rates.items():
        logger.info("Column %-12s | null_rate = %.4f", col, rate)
        # Customer ID naturally has ~22-24% nulls in Online Retail II; enforce strict threshold on others
        if col != "Customer ID" and rate > max_allowed:
            raise DataValidationError(
                f"Column '{col}' null rate {rate:.4f} exceeds max_null_rate {max_allowed:.4f}"
            )

    logger.info("Raw schema validation passed | total_rows=%d", len(df))
    return null_rates
