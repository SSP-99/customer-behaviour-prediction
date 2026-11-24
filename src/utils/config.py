"""Configuration loader returning strongly typed dataclasses from config/config.yaml."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project metadata and global execution settings."""

    name: str
    company: str
    seed: int
    log_level: str


@dataclass(frozen=True)
class DataConfig:
    """Paths and sources for raw, interim, and processed datasets."""

    uci_repo_id: int
    raw_excel_path: str
    raw_sheets: list[str]
    raw_combined_path: str
    interim_customer_path: str
    interim_product_path: str
    cleaning_audit_path: str
    processed_features_path: str
    processed_product_pairs_path: str


@dataclass(frozen=True)
class PointInTimeConfig:
    """Strict point-in-time cutoff and future prediction window settings."""

    cutoff_date: str
    churn_window_days: int
    min_purchases_before_cutoff: int


@dataclass(frozen=True)
class CleaningRulesConfig:
    """Thresholds and rules for transaction data cleaning and validation."""

    max_null_rate: float
    min_valid_price: float
    max_valid_price: float
    max_valid_abs_quantity: int
    bulk_order_qty_threshold: int
    iqr_multiplier: float
    min_date: str
    max_date: str
    cancellation_prefix: str
    non_product_stockcodes: list[str]


@dataclass(frozen=True)
class SplittingConfig:
    """Train/validation/test split ratios and cross-validation configuration."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    cv_folds: int
    stratify_col: str


@dataclass(frozen=True)
class ArtifactsConfig:
    """File paths for persisted models, scalers, explainers, and logs."""

    dir: str
    preprocessor_path: str
    rfm_scaler_path: str
    best_churn_model_path: str
    best_clv_model_path: str
    kmeans_model_path: str
    recommender_rules_path: str
    shap_explainer_path: str
    experiment_log_path: str
    feature_columns_path: str


@dataclass(frozen=True)
class APIConfig:
    """FastAPI server settings."""

    host: str
    port: int
    title: str
    version: str


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL connection configuration resolved from environment variables."""

    url: str


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration container."""

    project: ProjectConfig
    data: DataConfig
    point_in_time: PointInTimeConfig
    cleaning_rules: CleaningRulesConfig
    splitting: SplittingConfig
    models: dict[str, Any] = field(default_factory=dict)
    artifacts: ArtifactsConfig | None = None
    api: APIConfig | None = None
    database: DatabaseConfig | None = None


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Load YAML configuration from disk and return an AppConfig dataclass.

    Args:
        config_path: Relative or absolute path to the YAML configuration file.

    Returns:
        Populated AppConfig instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    db_section = raw["database"]
    db_url = os.getenv(db_section["url_env_var"], db_section["default_url"])

    return AppConfig(
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(**raw["data"]),
        point_in_time=PointInTimeConfig(**raw["point_in_time"]),
        cleaning_rules=CleaningRulesConfig(**raw["cleaning_rules"]),
        splitting=SplittingConfig(**raw["splitting"]),
        models=raw["models"],
        artifacts=ArtifactsConfig(**raw["artifacts"]),
        api=APIConfig(**raw["api"]),
        database=DatabaseConfig(url=db_url),
    )
