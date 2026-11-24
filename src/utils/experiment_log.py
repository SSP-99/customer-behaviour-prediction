"""CSV-based experiment tracking utility for model comparison and reporting."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

LOG_COLUMNS = [
    "timestamp_utc",
    "task",
    "model_name",
    "split",
    "train_time_sec",
    "params_json",
    "metrics_json",
]


def log_experiment(
    log_path: str | Path,
    task: str,
    model_name: str,
    split: str,
    train_time_sec: float,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Append a single model experiment run to the CSV experiment log.

    Args:
        log_path: Destination CSV path.
        task: Problem type (e.g., 'churn', 'clv', 'segmentation').
        model_name: Identifier for the estimator (e.g., 'xgboost', 'keras_mlp').
        split: Evaluation dataset split ('cv_train', 'validation', 'test').
        train_time_sec: Elapsed training duration in seconds.
        params: Dictionary of hyperparameters used.
        metrics: Dictionary of computed evaluation metrics.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "model_name": model_name,
        "split": split,
        "train_time_sec": round(float(train_time_sec), 4),
        "params_json": json.dumps(params, sort_keys=True, default=str),
        "metrics_json": json.dumps(
            {k: round(float(v), 6) for k, v in metrics.items()}, sort_keys=True
        ),
    }

    df_new = pd.DataFrame([record], columns=LOG_COLUMNS)
    write_header = not path.exists()
    df_new.to_csv(path, mode="a", header=write_header, index=False)
    logger.info(
        "Logged experiment | task=%s | model=%s | split=%s | time=%.2fs | metrics=%s",
        task,
        model_name,
        split,
        train_time_sec,
        record["metrics_json"],
    )


def load_experiment_logs(log_path: str | Path) -> pd.DataFrame:
    """Read experiment log CSV and expand JSON metrics into flat columns.

    Args:
        log_path: Path to the experiment log CSV.

    Returns:
        DataFrame with flattened metric columns, or empty DataFrame if file does not exist.
    """
    path = Path(log_path)
    if not path.exists():
        logger.warning("Experiment log not found at %s", path)
        return pd.DataFrame(columns=LOG_COLUMNS)

    df = pd.read_csv(path)
    if df.empty:
        return df

    metrics_expanded = df["metrics_json"].apply(json.loads).apply(pd.Series)
    return pd.concat([df.drop(columns=["metrics_json"]), metrics_expanded], axis=1)
