"""End-to-end training orchestrator for churn classification and CLV regression."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.train_churn import train_and_evaluate_churn
from src.models.train_clv import train_and_evaluate_clv
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline(config: AppConfig) -> dict[str, Any]:
    """Load processed features & fitted preprocessor, train churn & CLV models, and persist winners."""
    if config.artifacts is None:
        raise ValueError("Artifacts configuration must be defined for training.")

    log_path = Path(config.artifacts.experiment_log_path)
    if log_path.exists():
        log_path.unlink()

    df_features = pd.read_parquet(config.data.processed_features_path)
    preprocessor = joblib.load(config.artifacts.preprocessor_path)

    logger.info("Loaded processed features (%d rows) for model training.", len(df_features))

    churn_summary = train_and_evaluate_churn(df_features, preprocessor, config)
    clv_summary = train_and_evaluate_clv(df_features, preprocessor, config)

    summary = {
        "churn": churn_summary,
        "clv": clv_summary,
        "experiment_log": str(log_path),
    }
    logger.info("Training pipeline summary: %s", summary)
    return summary
