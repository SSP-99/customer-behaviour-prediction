"""Orchestrator for RFM customer segmentation and product recommender persistence."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.segmentation.recommender import fit_and_save_recommender
from src.segmentation.rfm_clustering import fit_rfm_segmentation
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_segmentation_pipeline(config: AppConfig) -> dict[str, Any]:
    """Run RFM K-Means clustering and fit/save the product recommendation engine."""
    df_features = pd.read_parquet(config.data.processed_features_path)
    seg_result = fit_rfm_segmentation(df_features, config)

    scored_out = Path(config.data.processed_features_path).parent / "customer_segments.parquet"
    seg_result["scored_df"].to_parquet(scored_out, index=False)
    logger.info("Saved scored customer segments to %s", scored_out)

    recommender = fit_and_save_recommender(config)
    return {
        "default_k": seg_result["default_k"],
        "persona_map": seg_result["persona_map"],
        "segment_profiles": seg_result["segment_profiles"],
        "indexed_products": len(recommender.rules_by_item),
    }
