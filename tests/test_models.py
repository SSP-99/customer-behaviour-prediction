"""Unit tests for custom evaluation metrics, CV splitters, and baseline estimators."""

import numpy as np
import pytest

from src.models.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_roc_auc,
    stratified_kfold_indices,
)
from src.models.train_churn import BalancedLogisticRegression
from src.models.train_clv import ClosedFormRidgeRegressor


def test_roc_auc_and_classification_metrics() -> None:
    """Verify exact ROC-AUC, recall, precision, and confusion matrix counts."""
    y_true = np.array(list((0, 0, 1, 1)))
    y_prob = np.array(list((0.1, 0.4, 0.6, 0.9)))

    assert compute_roc_auc(y_true, y_prob) == pytest.approx(1.0)
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["tp"] == 2.0
    assert metrics["tn"] == 2.0


def test_regression_metrics_exact_and_noisy() -> None:
    """Verify MAE, RMSE, and R2 calculations."""
    y_true = np.array(list((100.0, 200.0, 300.0)))
    y_pred = np.array(list((110.0, 190.0, 300.0)))

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(20.0 / 3.0)
    assert metrics["r2"] > 0.98


def test_stratified_kfold_and_numpy_baselines() -> None:
    """Verify stratified K-fold preserves class balance and NumPy baselines fit cleanly."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 4))
    y_cls = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    y_reg = np.maximum(100.0 * (x[:, 0] + 2.0), 0.0)

    folds = stratified_kfold_indices(y_cls, n_splits=3, seed=42)
    assert len(folds) == 3

    clf = BalancedLogisticRegression(c_reg=1.0, max_iter=200).fit(x, y_cls)
    probs = clf.predict_proba(x)
    assert probs.shape == (60, 2)

    reg = ClosedFormRidgeRegressor(alpha=1.0).fit(x, y_reg)
    preds = reg.predict(x)
    assert preds.shape == (60,)
    assert np.all(preds >= 0.0)
