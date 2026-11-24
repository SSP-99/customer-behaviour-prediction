"""End-to-end CLI pipeline runner for Vantara Customer Behavior Prediction Platform."""

import argparse
import time

from src.data.pipeline import run_data_pipeline
from src.explainability.pipeline import run_explainability_pipeline
from src.features.pipeline import run_feature_pipeline
from src.models.pipeline import run_training_pipeline
from src.segmentation.pipeline import run_segmentation_pipeline
from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

VALID_STAGES = ("data", "features", "train", "segment", "explain", "all")


def run_data_stage(config: AppConfig) -> None:
    """Execute raw data ingestion, validation, cleaning, and audit persistence."""
    audit = run_data_pipeline(config)
    logger.info("Data stage summary: %s", audit["summary"])


def run_features_stage(config: AppConfig) -> None:
    """Execute strict point-in-time feature engineering and target generation."""
    df_features = run_feature_pipeline(config)
    logger.info("Features stage complete | shape=%s", df_features.shape)


def run_train_stage(config: AppConfig) -> None:
    """Execute churn classification and CLV regression model training and selection."""
    summary = run_training_pipeline(config)
    logger.info(
        "Train stage complete | best_churn=%s | best_clv=%s",
        summary["churn"]["best_model_name"],
        summary["clv"]["best_model_name"],
    )


def run_segment_stage(config: AppConfig) -> None:
    """Execute RFM customer segmentation and market-basket product recommender fitting."""
    summary = run_segmentation_pipeline(config)
    logger.info(
        "Segment stage complete | k=%d | personas=%s | indexed_products=%d",
        summary["default_k"],
        summary["persona_map"],
        summary["indexed_products"],
    )


def run_explain_stage(config: AppConfig) -> None:
    """Execute global and local SHAP explainability artifact generation."""
    summary = run_explainability_pipeline(config)
    logger.info("Explain stage complete | model=%s", summary["model_name"])


def execute_pipeline(stage: str, config_path: str = "config/config.yaml") -> None:
    """Run one or all pipeline stages sequentially with global seed and structured logging."""
    start_time = time.perf_counter()
    config = load_config(config_path)
    set_global_seed(config.project.seed)

    logger.info(
        "Starting %s | stage=%s | seed=%d",
        config.project.name,
        stage,
        config.project.seed,
    )

    stage_dispatch = {
        "data": run_data_stage,
        "features": run_features_stage,
        "train": run_train_stage,
        "segment": run_segment_stage,
        "explain": run_explain_stage,
    }

    if stage == "all":
        for step_name, step_fn in stage_dispatch.items():
            step_start = time.perf_counter()
            step_fn(config)
            logger.info(
                "Completed stage [%s] in %.2fs", step_name, time.perf_counter() - step_start
            )
    else:
        stage_dispatch[stage](config)

    total_elapsed = time.perf_counter() - start_time
    logger.info("Pipeline execution finished | stage=%s | total_time=%.2fs", stage, total_elapsed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Vantara Retail Solutions - Customer Behavior Prediction Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=VALID_STAGES,
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    execute_pipeline(stage=args.stage, config_path=args.config)
