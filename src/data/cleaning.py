"""Pure, unit-testable transaction cleaning functions for Online Retail II."""

import numpy as np
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEDUP_SUBSET = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate line items while preserving legitimate repeated purchases.

    Rule: A row is treated as an exact duplicate only if all 8 canonical columns
    (invoice, stock_code, description, quantity, invoice_date, price, customer_id,
    and country) are identical. Legitimate repeat purchases across different invoices
    or timestamps are preserved.

    Args:
        df: Input transactions DataFrame.

    Returns:
        Tuple of (deduplicated DataFrame, count of removed duplicate rows).
    """
    before = len(df)
    deduped = df.drop_duplicates(subset=DEDUP_SUBSET, keep="first").reset_index(drop=True)
    removed = before - len(deduped)
    logger.info(
        "remove_exact_duplicates | before=%d | after=%d | removed=%d",
        before,
        len(deduped),
        removed,
    )
    return deduped, removed


def handle_bad_prices(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Remove rows with missing timestamps, zero quantity, or invalid prices.

    Price Rule (configured in config.yaml -> cleaning_rules):
    - Drops rows with null `invoice_date`, `quantity`, or `price`.
    - Drops rows where `price < min_valid_price` (0.01), eliminating zero-price
      promotional/damaged write-offs and negative bad-debt adjustments.
    - Drops rows where `quantity == 0`.

    Args:
        df: Input transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        Filtered DataFrame containing only valid priced line items.
    """
    rules = config.cleaning_rules
    before = len(df)
    cleaned = df.dropna(subset=["invoice_date", "quantity", "price"]).copy()
    cleaned = cleaned[
        (cleaned["price"] >= rules.min_valid_price) & (cleaned["quantity"] != 0)
    ].reset_index(drop=True)
    logger.info(
        "handle_bad_prices | before=%d | after=%d | removed=%d",
        before,
        len(cleaned),
        before - len(cleaned),
    )
    return cleaned


def flag_returns(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Flag cancellations and returns without dropping them, computing gross and return amounts.

    Rule:
    - `is_return` is True if `invoice` starts with `cancellation_prefix` ("C") or `quantity < 0`.
    - Cancellation rows with positive quantities are normalized to negative quantities.
    - `line_total` = signed `quantity * price`.
    - `gross_sales` = positive `line_total` for non-returns, else 0.0.
    - `returns` = absolute `line_total` for returns, else 0.0.

    Args:
        df: Input transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        DataFrame enriched with `is_return`, `line_total`, `gross_sales`, and `returns`.
    """
    prefix = config.cleaning_rules.cancellation_prefix.upper()
    out = df.copy()
    starts_with_c = out["invoice"].astype(str).str.strip().str.upper().str.startswith(prefix)
    out["is_return"] = starts_with_c | (out["quantity"] < 0)

    out.loc[starts_with_c & (out["quantity"] > 0), "quantity"] *= -1

    out["line_total"] = (out["quantity"] * out["price"]).round(4)
    out["gross_sales"] = np.where(~out["is_return"], out["line_total"], 0.0).round(4)
    out["returns"] = np.where(out["is_return"], out["line_total"].abs(), 0.0).round(4)

    logger.info(
        "flag_returns | total=%d | returns=%d (%.2f%%)",
        len(out),
        int(out["is_return"].sum()),
        100.0 * out["is_return"].mean() if len(out) else 0.0,
    )
    return out


def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize product descriptions using a StockCode -> modal description lookup table.

    Normalizes descriptions to uppercase trimmed strings, computes the most frequent
    (mode) non-empty description per `stock_code`, and maps it across all rows.

    Args:
        df: Input transactions DataFrame.

    Returns:
        DataFrame with standardized `stock_code` and `description` columns.
    """
    out = df.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.strip().str.upper()
    cleaned_desc = (
        out["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    out["description"] = cleaned_desc

    valid_desc = out[out["description"] != ""]
    if not valid_desc.empty:
        counts = valid_desc.groupby(["stock_code", "description"]).size().reset_index(name="cnt")
        top_desc = (
            counts.sort_values(["stock_code", "cnt"], ascending=[True, False])
            .drop_duplicates(subset=["stock_code"], keep="first")
            .set_index("stock_code")["description"]
        )
        out["description"] = out["stock_code"].map(top_desc.to_dict()).fillna("UNKNOWN PRODUCT")
    else:
        out["description"] = out["description"].replace("", "UNKNOWN PRODUCT")

    logger.info("standardize_descriptions | unique_stock_codes=%d", out["stock_code"].nunique())
    return out


def filter_non_product_codes(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Remove administrative and non-product StockCodes (POST, D, M, BANK CHARGES, etc.).

    Intended for product-affinity and basket co-occurrence features so postage fees
    and manual adjustments do not distort product recommendations.

    Args:
        df: Input transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        DataFrame excluding administrative non-product StockCodes.
    """
    non_product = {code.strip().upper() for code in config.cleaning_rules.non_product_stockcodes}
    before = len(df)
    codes_upper = df["stock_code"].astype(str).str.strip().str.upper()
    filtered = df[~codes_upper.isin(non_product)].reset_index(drop=True)
    logger.info(
        "filter_non_product_codes | before=%d | after=%d | removed=%d",
        before,
        len(filtered),
        before - len(filtered),
    )
    return filtered


def detect_outliers(df: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag IQR outliers, separate wholesale bulk orders from data-entry errors, and drop errors.

    Domain + IQR Rules:
    - Computes IQR upper fences on `abs(quantity)` and `price`.
    - `is_bulk_order`: True when `bulk_order_qty_threshold <= abs(quantity) <= max_valid_abs_quantity`.
    - `is_data_error`: True when `abs(quantity) > max_valid_abs_quantity` (e.g., 80,995)
      or `price > max_valid_price` (e.g., 38,970.00).
    - Removes `is_data_error` rows from the returned clean DataFrame while returning
      all IQR/domain-flagged rows in a separate inspection table.

    Args:
        df: Input transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        Tuple of (cleaned DataFrame without data errors, DataFrame of all flagged outlier rows).
    """
    rules = config.cleaning_rules
    out = df.copy()

    abs_qty = out["quantity"].abs()
    q1_qty, q3_qty = float(abs_qty.quantile(0.25)), float(abs_qty.quantile(0.75))
    upper_qty_iqr = q3_qty + rules.iqr_multiplier * (q3_qty - q1_qty)

    q1_price, q3_price = float(out["price"].quantile(0.25)), float(out["price"].quantile(0.75))
    upper_price_iqr = q3_price + rules.iqr_multiplier * (q3_price - q1_price)

    out["is_iqr_outlier"] = (abs_qty > upper_qty_iqr) | (out["price"] > upper_price_iqr)
    out["is_data_error"] = (abs_qty > rules.max_valid_abs_quantity) | (
        out["price"] > rules.max_valid_price
    )
    out["is_bulk_order"] = (abs_qty >= rules.bulk_order_qty_threshold) & (~out["is_data_error"])

    flagged_table = out[out["is_iqr_outlier"] | out["is_bulk_order"] | out["is_data_error"]].copy()
    cleaned = out[~out["is_data_error"]].reset_index(drop=True)

    logger.info(
        "detect_outliers | total=%d | bulk_orders=%d | data_errors_removed=%d | flagged_total=%d",
        len(out),
        int(cleaned["is_bulk_order"].sum()),
        int(out["is_data_error"].sum()),
        len(flagged_table),
    )
    return cleaned, flagged_table


def flag_missing_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into a valid customer-level subset and a full product-level table.

    - Product-level table retains all rows (including missing customer_id) with a boolean
      `has_customer_id` indicator.
    - Customer-level subset filters to `has_customer_id == True` and formats `customer_id`
      as a clean integer string (e.g., '13085').

    Args:
        df: Cleaned transactions DataFrame.

    Returns:
        Tuple of (customer_level_df, product_level_df).
    """
    product_df = df.copy()
    numeric_ids = pd.to_numeric(product_df["customer_id"], errors="coerce")
    product_df["has_customer_id"] = numeric_ids.notna()
    product_df["customer_id"] = numeric_ids.apply(lambda x: str(int(x)) if pd.notna(x) else None)

    customer_df = product_df[product_df["has_customer_id"]].copy().reset_index(drop=True)
    customer_df["customer_id"] = customer_df["customer_id"].astype(str)

    logger.info(
        "flag_missing_customers | product_rows=%d | customer_rows=%d | unique_customers=%d",
        len(product_df),
        len(customer_df),
        customer_df["customer_id"].nunique(),
    )
    return customer_df, product_df
