"""Unit tests for strict point-in-time feature engineering, leakage prevention, and preprocessing."""

import pandas as pd
import pytest

from src.features.engineering import (
    build_point_in_time_features_and_targets,
    build_product_associations,
)
from src.features.preprocessing import (
    ID_AND_TARGET_COLS,
    build_preprocessor,
    get_feature_columns,
)
from src.utils.config import AppConfig, load_config


@pytest.fixture()
def app_config() -> AppConfig:
    """Load application configuration."""
    return load_config("config/config.yaml")


@pytest.fixture()
def synthetic_customer_tx() -> pd.DataFrame:
    """Create synthetic customer transactions spanning pre-cutoff and post-cutoff periods."""
    return pd.DataFrame(
        {
            "invoice": ["1001", "1002", "1003", "1004", "1005"],
            "stock_code": ["A1", "A2", "A1", "A3", "A2"],
            "description": ["ITEM A1", "ITEM A2", "ITEM A1", "ITEM A3", "ITEM A2"],
            "quantity": list((5, 2, 10, 4, 3)),
            "invoice_date": pd.to_datetime(
                [
                    "2011-06-01 10:00:00",
                    "2011-08-20 14:00:00",
                    "2011-07-15 09:00:00",
                    "2011-10-05 11:00:00",
                    "2011-12-20 11:00:00",
                ]
            ),
            "price": list((10.0, 25.0, 10.0, 50.0, 25.0)),
            "customer_id": ["101", "101", "102", "101", "102"],
            "country": ["United Kingdom", "United Kingdom", "Germany", "United Kingdom", "Germany"],
            "is_return": [False, False, False, False, False],
            "line_total": list((50.0, 50.0, 100.0, 200.0, 75.0)),
            "gross_sales": list((50.0, 50.0, 100.0, 200.0, 75.0)),
            "returns": list((0.0, 0.0, 0.0, 0.0, 0.0)),
            "is_bulk_order": [False, False, False, False, False],
        }
    )


def test_zero_feature_leakage_from_post_cutoff_transactions(
    synthetic_customer_tx: pd.DataFrame, app_config: AppConfig
) -> None:
    """Mutating post-cutoff transactions must change targets ONLY, never any feature column."""
    baseline = build_point_in_time_features_and_targets(synthetic_customer_tx, app_config)

    modified_tx = synthetic_customer_tx.copy()
    modified_tx.loc[modified_tx["invoice"] == "1004", "line_total"] = 999999.0
    modified_tx.loc[modified_tx["invoice"] == "1004", "quantity"] = 9999

    mutated = build_point_in_time_features_and_targets(modified_tx, app_config)

    feature_cols = [c for c in baseline.columns if c not in ID_AND_TARGET_COLS]
    pd.testing.assert_frame_equal(
        baseline[feature_cols],
        mutated[feature_cols],
        check_exact=False,
    )
    clv_101 = float(mutated.loc[mutated["customer_id"] == "101", "clv_90d_target"].squeeze())
    assert clv_101 == pytest.approx(999999.0)


def test_target_labels_and_window_boundaries(
    synthetic_customer_tx: pd.DataFrame, app_config: AppConfig
) -> None:
    """Customer 101 is active (churn=0, CLV=200); Customer 102 is churned (churn=1, CLV=0)."""
    df_feat = build_point_in_time_features_and_targets(synthetic_customer_tx, app_config)

    c101 = df_feat[df_feat["customer_id"] == "101"].squeeze()
    assert int(c101["churn_label"]) == 0
    assert float(c101["clv_90d_target"]) == pytest.approx(200.0)
    assert float(c101["historical_clv"]) == pytest.approx(100.0)

    c102 = df_feat[df_feat["customer_id"] == "102"].squeeze()
    assert int(c102["churn_label"]) == 1
    assert float(c102["clv_90d_target"]) == pytest.approx(0.0)


def test_preprocessor_transforms_without_nulls(
    synthetic_customer_tx: pd.DataFrame, app_config: AppConfig
) -> None:
    """Fitted TabularPreprocessor produces finite numeric array with expected column count."""
    df_feat = build_point_in_time_features_and_targets(synthetic_customer_tx, app_config)
    num_cols, cat_cols = get_feature_columns(df_feat)
    prep = build_preprocessor(num_cols, cat_cols)
    x_mat = prep.fit_transform(df_feat[num_cols + cat_cols])

    assert x_mat.shape == (len(df_feat), len(prep.get_feature_names_out()))
    assert pd.isna(x_mat).sum() == 0


def test_product_associations_ignore_post_cutoff_baskets(app_config: AppConfig) -> None:
    """Product co-occurrence pairs only count baskets strictly before cutoff_date."""
    tx = pd.DataFrame(
        {
            "invoice": ["B1", "B1", "B2", "B2"],
            "stock_code": ["X1", "X2", "X3", "X4"],
            "description": ["PROD X1", "PROD X2", "PROD X3", "PROD X4"],
            "quantity": list((1, 1, 1, 1)),
            "invoice_date": pd.to_datetime(
                [
                    "2011-05-01 10:00:00",
                    "2011-05-01 10:00:00",
                    "2011-10-15 10:00:00",
                    "2011-10-15 10:00:00",
                ]
            ),
            "price": list((5.0, 5.0, 5.0, 5.0)),
            "is_return": [False, False, False, False],
        }
    )
    pairs = build_product_associations(tx, app_config)
    assert "X3" not in set(pairs.get("item_a", pd.Series(dtype=str)))
