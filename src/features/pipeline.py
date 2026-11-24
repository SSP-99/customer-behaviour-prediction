"""End-to-end feature engineering, splitting, and preprocessor persistence pipeline."""

from pathlib import Path

import pandas as pd

from src.data.validation import validate_transactions
from src.features.engineering import (
    build_point_in_time_features_and_targets,
    build_product_associations,
)
from src.features.preprocessing import (
    fit_and_save_preprocessor,
    stratified_train_val_test_split,
)
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_feature_pipeline(config: AppConfig) -> pd.DataFrame:
    """Validate interim data, build point-in-time features/targets, fit scaler, and save outputs.

    Args:
        config: Application configuration dataclass.

    Returns:
        Complete customer-level features and targets DataFrame with split assignments.
    """
    customer_tx = pd.read_parquet(config.data.interim_customer_path)
    product_tx = pd.read_parquet(config.data.interim_product_path)

    # Pre-feature validation gate (required by engineering rules)
    validate_transactions(
        customer_tx,
        config,
        stage_label="pre_features_customer",
        require_non_null_customer=True,
    )

    # Build strict point-in-time features & future 90d targets
    customer_features = build_point_in_time_features_and_targets(customer_tx, config)

    # Build pre-cutoff product association pairs
    product_pairs = build_product_associations(product_tx, config)

    # Stratified 70/15/15 split and fit preprocessor strictly on training split
    train_df, val_df, test_df = stratified_train_val_test_split(customer_features, config)
    fit_and_save_preprocessor(train_df, config)

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    combined_with_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)

    features_out = Path(config.data.processed_features_path)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    combined_with_splits.to_parquet(features_out, index=False)
    logger.info(
        "Saved processed customer features (%d rows) to %s", len(combined_with_splits), features_out
    )

    pairs_out = Path(config.data.processed_product_pairs_path)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    product_pairs.to_parquet(pairs_out, index=False)
    logger.info("Saved product associations (%d pairs) to %s", len(product_pairs), pairs_out)

    return combined_with_splits
