"""End-to-end data ingestion, validation, cleaning, and audit pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cleaning import (
    detect_outliers,
    filter_non_product_codes,
    flag_missing_customers,
    flag_returns,
    handle_bad_prices,
    remove_exact_duplicates,
    standardize_descriptions,
)
from src.data.loader import load_raw_data
from src.data.validation import validate_transactions
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save DataFrame to Parquet with automatic CSV fallback."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        logger.warning("Parquet save failed (%s); falling back to CSV: %s", exc, csv_path)
        df.to_csv(csv_path, index=False)
        return csv_path


def run_data_pipeline(config: AppConfig, force_reload: bool = False) -> dict[str, Any]:
    """Execute load -> validate -> clean -> validate -> save and write cleaning_audit.json.

    Args:
        config: Application configuration dataclass.
        force_reload: If True, forces re-reading raw source files instead of cached raw Parquet.

    Returns:
        The complete cleaning audit dictionary written to `config.data.cleaning_audit_path`.
    """
    audit_steps: list[dict[str, Any]] = []

    def _record_step(step_name: str, before_rows: int, after_rows: int, notes: str = "") -> None:
        entry = {
            "step": step_name,
            "rows_before": int(before_rows),
            "rows_after": int(after_rows),
            "rows_removed": int(before_rows - after_rows),
            "notes": notes,
        }
        audit_steps.append(entry)
        logger.info(
            "Audit | %-28s | before=%7d | after=%7d | diff=%+7d",
            step_name,
            before_rows,
            after_rows,
            after_rows - before_rows,
        )

    # 1. Load raw combined data
    df_raw = load_raw_data(config, force_reload=force_reload)
    _record_step(
        "load_raw_data", len(df_raw), len(df_raw), "Concatenated & standardized raw sheets"
    )

    # 2. Pre-cleaning validation
    pre_val_report = validate_transactions(df_raw, config, stage_label="pre_cleaning")

    # 3. Pure cleaning transformations in deterministic sequence
    df_dedup, dup_removed = remove_exact_duplicates(df_raw)
    _record_step(
        "remove_exact_duplicates",
        len(df_raw),
        len(df_dedup),
        f"Exact 8-col duplicates removed: {dup_removed}",
    )

    df_priced = handle_bad_prices(df_dedup, config)
    _record_step(
        "handle_bad_prices",
        len(df_dedup),
        len(df_priced),
        f"Price >= {config.cleaning_rules.min_valid_price}",
    )

    df_returns = flag_returns(df_priced, config)
    _record_step(
        "flag_returns",
        len(df_priced),
        len(df_returns),
        f"Retained {int(df_returns['is_return'].sum())} return rows with signed line_total",
    )

    df_desc = standardize_descriptions(df_returns)
    _record_step(
        "standardize_descriptions",
        len(df_returns),
        len(df_desc),
        "Mapped modal StockCode descriptions",
    )

    df_clean, flagged_outliers = detect_outliers(df_desc, config)
    _record_step(
        "detect_outliers",
        len(df_desc),
        len(df_clean),
        f"Bulk orders flagged={int(df_clean['is_bulk_order'].sum())}, data errors dropped={int(flagged_outliers['is_data_error'].sum())}",
    )

    # 4. Split customer-level table vs product-level table (with non-product codes filtered)
    customer_df, product_full_df = flag_missing_customers(df_clean)
    _record_step(
        "flag_missing_customers",
        len(df_clean),
        len(customer_df),
        f"Customer-level valid rows={len(customer_df)}; product full rows={len(product_full_df)}",
    )

    product_affinity_df = filter_non_product_codes(product_full_df, config)
    _record_step(
        "filter_non_product_codes",
        len(product_full_df),
        len(product_affinity_df),
        "Removed admin/postage codes for product-level basket table",
    )

    # 5. Post-cleaning validation on both output tables
    post_customer_val = validate_transactions(
        customer_df, config, stage_label="post_cleaning_customer", require_non_null_customer=True
    )
    post_product_val = validate_transactions(
        product_affinity_df,
        config,
        stage_label="post_cleaning_product",
        require_non_null_customer=False,
    )

    # 6. Persist outputs and cleaning audit JSON
    customer_saved = _save_dataframe(customer_df, config.data.interim_customer_path)
    product_saved = _save_dataframe(product_affinity_df, config.data.interim_product_path)
    outliers_path = Path(config.data.interim_product_path).parent / "flagged_outliers.parquet"
    _save_dataframe(flagged_outliers, outliers_path)

    audit_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "customer_table": str(customer_saved),
            "product_table": str(product_saved),
            "flagged_outliers_table": str(outliers_path),
        },
        "summary": {
            "raw_rows": int(len(df_raw)),
            "customer_rows": int(len(customer_df)),
            "unique_customers": int(customer_df["customer_id"].nunique()),
            "product_rows": int(len(product_affinity_df)),
            "unique_products": int(product_affinity_df["stock_code"].nunique()),
            "customer_return_rows": int(customer_df["is_return"].sum()),
            "bulk_order_rows": int(customer_df["is_bulk_order"].sum()),
        },
        "steps": audit_steps,
        "validation_reports": {
            "pre_cleaning": pre_val_report,
            "post_cleaning_customer": post_customer_val,
            "post_cleaning_product": post_product_val,
        },
    }

    audit_path = Path(config.data.cleaning_audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    logger.info("Saved cleaning audit report to %s", audit_path)
    return audit_payload
