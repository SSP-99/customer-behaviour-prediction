"""Deterministic random seed management across standard library, NumPy, and TensorFlow."""

import os
import random

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer random seed (defaults to 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        logger.warning("TensorFlow not installed; skipped TensorFlow seed initialization.")

    logger.info("Global random seed fixed to %d", seed)
