"""Orchestrator for global and local model explainability generation."""

from typing import Any

import joblib
import pandas as pd

from src.explainability.shap_lime import build_and_save_explainer
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_explainability_pipeline(config: AppConfig) -> dict[str, Any]:
    """Load best churn model and preprocessor, build SHAP explainer, and save artifacts."""
    if config.artifacts is None:
        raise ValueError("Artifacts configuration must be defined for explainability.")

    df_features = pd.read_parquet(config.data.processed_features_path)
    preprocessor = joblib.load(config.artifacts.preprocessor_path)
    churn_bundle = joblib.load(config.artifacts.best_churn_model_path)

    explainer = build_and_save_explainer(df_features, preprocessor, churn_bundle, config)
    top_5 = explainer.global_importance[:5]
    logger.info("Top 5 global churn features: %s", top_5)
    return {
        "model_name": explainer.model_name,
        "top_5_features": top_5,
    }
