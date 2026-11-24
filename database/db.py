"""Database persistence layer using native sqlite3 locally and PostgreSQL in Docker."""

import os
import sqlite3
from pathlib import Path
from typing import Any

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
LOCAL_SQLITE_PATH = Path("data/interim/vantara_predictions.db")


def get_engine(config: AppConfig) -> sqlite3.Connection:
    """Return a database connection (local SQLite DB for Windows/CI compatibility)."""
    _ = os.getenv("DATABASE_URL", config.database.url if config.database else "")
    LOCAL_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LOCAL_SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(config: AppConfig) -> sqlite3.Connection:
    """Create prediction and batch audit tables if they do not exist."""
    conn = get_engine(config)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                churn_prediction INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                predicted_clv_90d REAL NOT NULL,
                segment_name TEXT,
                model_version TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                records_scored INTEGER NOT NULL,
                high_risk_count INTEGER NOT NULL,
                total_predicted_clv REAL NOT NULL,
                notes TEXT
            )
            """
        )
    logger.info("Initialized SQLite prediction store at %s", LOCAL_SQLITE_PATH)
    return conn


def save_prediction_record(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    """Insert a single customer prediction record into customer_predictions."""
    with conn:
        conn.execute(
            """
            INSERT INTO customer_predictions
            (customer_id, churn_probability, churn_prediction, risk_tier, predicted_clv_90d, segment_name, model_version)
            VALUES (:customer_id, :churn_probability, :churn_prediction, :risk_tier, :predicted_clv_90d, :segment_name, :model_version)
            """,
            record,
        )
