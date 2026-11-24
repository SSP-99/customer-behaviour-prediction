"""Transaction wrangling and cleaning logic for Online Retail II."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_transactions(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Clean raw retail line items while preserving return signals and product-level rows.

    Cleaning steps:
    1. Parse timestamps and strip whitespace from string columns.
    2. Remove exact duplicate rows.
    3. Filter out non-product StockCodes (POST, D, M, BANK CHARGES, etc.).
    4. Remove rows with Price < min_valid_price or Quantity == 0.
    5. Flag cancellations/returns (`is_return` = True if Invoice starts with 'C' or Quantity < 0)
       rather than dropping them, so downstream features capture return behavior.
    6. Flag rows with valid `Customer ID` (`has_customer_id`) so customer-level work can filter
       while product-level co-occurrence rules use all valid product baskets.
    7. Compute `line_total` = Quantity * Price and cap extreme unit-quantity outliers via IQR rule
       for reporting while preserving exact net monetary amounts.

    Args:
        df: Raw validated transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        Cleaned transactions DataFrame with standardized schema and helper flags.
    """
    rules = config.cleaning_rules
    initial_rows = len(df)
    cleaned = df.copy()

    # 1. Standardize string columns and parse InvoiceDate
    cleaned["Invoice"] = cleaned["Invoice"].astype(str).str.strip()
    cleaned["StockCode"] = cleaned["StockCode"].astype(str).str.strip().str.upper()
    cleaned["Description"] = (
        cleaned["Description"].fillna("UNKNOWN PRODUCT").astype(str).str.strip()
    )
    cleaned["Country"] = cleaned["Country"].fillna("Unknown").astype(str).str.strip()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")

    before_date_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=["InvoiceDate", "Quantity", "Price"]).copy()
    logger.info(
        "Dropped %d rows with invalid InvoiceDate/Quantity/Price", before_date_drop - len(cleaned)
    )

    # 2. Drop exact duplicates
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    logger.info("Dropped %d exact duplicate rows", before_dedup - len(cleaned))

    # 3. Exclude non-product administrative StockCodes
    non_product_codes = {code.upper() for code in rules.non_product_stockcodes}
    before_stock = len(cleaned)
    cleaned = cleaned[~cleaned["StockCode"].isin(non_product_codes)].copy()
    logger.info("Dropped %d non-product StockCode rows", before_stock - len(cleaned))

    # 4. Enforce valid positive price and non-zero quantity
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["Price"] = pd.to_numeric(cleaned["Price"], errors="coerce")
    before_price = len(cleaned)
    cleaned = cleaned[
        (cleaned["Price"] >= rules.min_valid_price) & (cleaned["Quantity"] != 0)
    ].copy()
    logger.info(
        "Dropped %d rows with invalid price (< %.2f) or zero quantity",
        before_price - len(cleaned),
        rules.min_valid_price,
    )

    # 5. Flag cancellations and returns (do NOT drop them)
    cancel_prefix = rules.cancellation_prefix.upper()
    cleaned["is_cancellation"] = cleaned["Invoice"].str.upper().str.startswith(cancel_prefix)
    cleaned["is_return"] = cleaned["is_cancellation"] | (cleaned["Quantity"] < 0)

    # Ensure return quantities are consistently negative if marked as cancellation
    cleaned.loc[cleaned["is_cancellation"] & (cleaned["Quantity"] > 0), "Quantity"] *= -1

    # 6. Flag valid Customer IDs (keep anonymous rows for product-level association rules)
    cleaned["Customer ID"] = pd.to_numeric(cleaned["Customer ID"], errors="coerce")
    cleaned["has_customer_id"] = cleaned["Customer ID"].notna()
    cleaned["customer_id"] = cleaned["Customer ID"].fillna(-1).astype(np.int64)
    cleaned = cleaned.drop(columns=["Customer ID"])

    # 7. Compute line_total and flag extreme quantity outliers using IQR
    cleaned["line_total"] = (cleaned["Quantity"] * cleaned["Price"]).round(4)

    abs_qty = cleaned["Quantity"].abs()
    q1 = float(abs_qty.quantile(0.25))
    q3 = float(abs_qty.quantile(0.75))
    iqr = q3 - q1
    upper_fence = q3 + rules.iqr_multiplier * iqr
    cleaned["is_qty_outlier"] = abs_qty > upper_fence

    cleaned = cleaned.sort_values("InvoiceDate").reset_index(drop=True)

    logger.info(
        "Cleaning complete | initial=%d | final=%d | returns=%d (%.2f%%) | with_customer_id=%d",
        initial_rows,
        len(cleaned),
        int(cleaned["is_return"].sum()),
        100.0 * cleaned["is_return"].mean(),
        int(cleaned["has_customer_id"].sum()),
    )
    return cleaned


def save_cleaned_transactions(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist cleaned transactions DataFrame to Parquet format.

    Args:
        df: Cleaned transactions DataFrame.
        output_path: Destination Parquet file path.

    Returns:
        Resolved Path where the Parquet file was saved.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved cleaned transactions (%d rows) to %s", len(df), path)
    return path
