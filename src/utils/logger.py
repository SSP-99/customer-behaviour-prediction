"""Structured logging setup for Vantara Retail Solutions."""

import logging
import os
import sys
from pathlib import Path


def get_logger(
    name: str, level: str | None = None, log_file: str | None = "logs/pipeline.log"
) -> logging.Logger:
    """Create and configure a structured logger instance.

    Args:
        name: Module or component logger name (typically __name__).
        level: Logging level string (e.g., 'INFO', 'DEBUG'). Defaults to LOG_LEVEL env or INFO.
        log_file: Optional path to write persistent log output.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
