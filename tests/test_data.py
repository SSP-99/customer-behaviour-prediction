"""Unit tests for data validation and pure transaction cleaning functions."""

import pandas as pd
import pytest

from src.data.cleaning import (
    detect_outliers,
    filter_non_product_codes,
    flag_missing_customers,
    flag_returns,
    handle_bad_prices,
    remove_exact_duplicates,
    standardize_descriptions,
)
from src.data.validation import DataValidationError, validate_transactions
from src.utils.config import AppConfig, load_config


@pytest.fixture()
def app_config() -> AppConfig:
    """Load default application configuration for tests."""
    return load_config("config/config.yaml")


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    """Create a realistic miniature transactions DataFrame covering edge cases."""
    return pd.DataFrame(
        {
            "invoice": ["536365", "536365", "536366", "C536367", "536368", "536369", "536370"],
            "stock_code": ["85123A", "85123A", "85123A", "85123A", "POST", "22423", "23843"],
            "description": [
                "white hanging heart t-light holder",
                "white hanging heart t-light holder",
                "WHITE HANGING HEART T-LIGHT HOLDER ",
                "WHITE HANGING HEART T-LIGHT HOLDER",
                "POSTAGE",
                "REGENCY CAKESTAND 3 TIER",
                "PAPER CRAFT , LITTLE BIRDIE",
            ],
            "quantity": [6, 6, 6, -2, 1, 150, 80995],
            "invoice_date": pd.to_datetime(
                [
                    "2010-12-01 08:26:00",
                    "2010-12-01 08:26:00",
                    "2010-12-05 10:15:00",
                    "2010-12-06 11:00:00",
                    "2010-12-07 12:00:00",
                    "2010-12-08 13:00:00",
                    "2010-12-09 14:00:00",
                ]
            ),
            "price": [2.55, 2.55, 2.55, 2.55, 18.00, 10.95, 2.08],
            "customer_id": [17850.0, 17850.0, 17850.0, 17850.0, 17850.0, 13085.0, None],
            "country": [
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
            ],
        }
    )


def test_remove_exact_duplicates_preserves_repeat_purchases(
    sample_transactions: pd.DataFrame,
) -> None:
    """Exact 8-column duplicates are dropped while legitimate repeat purchases stay."""
    deduped, removed_count = remove_exact_duplicates(sample_transactions)
    assert removed_count == 1
    assert len(deduped) == len(sample_transactions) - 1
    assert {"536365", "536366"}.issubset(set(deduped["invoice"]))


def test_flag_returns_retains_rows_and_computes_signed_totals(
    sample_transactions: pd.DataFrame, app_config: AppConfig
) -> None:
    """Return/cancellation rows are retained with signed line_total and separate gross/returns."""
    flagged = flag_returns(sample_transactions, app_config)
    assert len(flagged) == len(sample_transactions)

    return_rows = flagged[flagged["invoice"] == "C536367"]
    assert len(return_rows) == 1
    row = next(return_rows.itertuples())
    assert bool(row.is_return) is True
    assert row.line_total == pytest.approx(-5.10)
    assert row.gross_sales == pytest.approx(0.0)
    assert row.returns == pytest.approx(5.10)


def test_filter_non_product_codes_removes_admin_items(
    sample_transactions: pd.DataFrame, app_config: AppConfig
) -> None:
    """Administrative StockCodes like POST are removed for product-level tables."""
    filtered = filter_non_product_codes(sample_transactions, app_config)
    assert "POST" not in set(filtered["stock_code"])
    assert len(filtered) == len(sample_transactions) - 1


def test_detect_outliers_separates_bulk_from_data_errors(
    sample_transactions: pd.DataFrame, app_config: AppConfig
) -> None:
    """Bulk orders (qty=150) are kept and flagged; data-entry errors (qty=80995) are removed."""
    cleaned, flagged_table = detect_outliers(sample_transactions, app_config)
    assert 80995 not in set(cleaned["quantity"])
    assert 150 in set(cleaned["quantity"])
    bulk_flag = bool(cleaned.loc[cleaned["quantity"] == 150, "is_bulk_order"].squeeze())
    assert bulk_flag is True
    assert 80995 in set(flagged_table["quantity"])


def test_standardize_descriptions_and_missing_customers(
    sample_transactions: pd.DataFrame, app_config: AppConfig
) -> None:
    """Descriptions standardize to uppercase mode; missing customer IDs split cleanly."""
    priced = handle_bad_prices(sample_transactions, app_config)
    standardized = standardize_descriptions(priced)
    assert set(standardized.loc[standardized["stock_code"] == "85123A", "description"]) == {
        "WHITE HANGING HEART T-LIGHT HOLDER"
    }

    cust_df, prod_df = flag_missing_customers(standardized)
    assert len(prod_df) == len(standardized)
    assert len(cust_df) == len(standardized) - 1
    assert cust_df["customer_id"].isna().sum() == 0
    assert all(cid.isdigit() for cid in cust_df["customer_id"])


def test_validation_raises_on_bad_schema_and_future_dates(
    sample_transactions: pd.DataFrame, app_config: AppConfig
) -> None:
    """validate_transactions raises DataValidationError on missing columns or future dates."""
    bad_schema = sample_transactions.drop(columns=["price"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_transactions(bad_schema, app_config, stage_label="test_bad_schema")

    future_df = sample_transactions.copy()
    future_df.loc[future_df.index.min(), "invoice_date"] = pd.Timestamp("2099-01-01")
    with pytest.raises(DataValidationError, match="exceeds allowed max_date"):
        validate_transactions(future_df, app_config, stage_label="test_future_date")
