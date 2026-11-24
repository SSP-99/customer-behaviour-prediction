"""Churn classification training, 5-fold CV tuning, validation selection, and test evaluation."""

import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.features.preprocessing import TabularPreprocessor, transform_features
from src.models.metrics import compute_classification_metrics, stratified_kfold_indices
from src.models.neural_nets import KerasChurnClassifier
from src.utils.config import AppConfig
from src.utils.experiment_log import log_experiment
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BalancedLogisticRegression:
    """Pure NumPy L2-regularized binary logistic regression with balanced class weights."""

    def __init__(self, c_reg: float = 1.0, max_iter: int = 500, lr: float = 0.05) -> None:
        """Initialize regularization strength and optimization settings."""
        self.c_reg = c_reg
        self.max_iter = max_iter
        self.lr = lr
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clip = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BalancedLogisticRegression":
        """Fit L2 logistic regression using batch gradient descent with class weights."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = x_arr.shape

        n_pos = max(float(np.sum(y_arr == 1)), 1.0)
        n_neg = max(float(np.sum(y_arr == 0)), 1.0)
        w_pos = n_samples / (2.0 * n_pos)
        w_neg = n_samples / (2.0 * n_neg)
        sample_weights = np.where(y_arr == 1, w_pos, w_neg)

        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        reg_lambda = 1.0 / max(self.c_reg * n_samples, 1e-6)

        for _ in range(self.max_iter):
            logits = x_arr @ self.weights_ + self.bias_
            preds = self._sigmoid(logits)
            errors = (preds - y_arr) * sample_weights
            grad_w = (x_arr.T @ errors) / n_samples + reg_lambda * self.weights_
            grad_b = float(np.mean(errors))
            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict 2-column probabilities [P(y=0), P(y=1)]."""
        if self.weights_ is None:
            raise RuntimeError("BalancedLogisticRegression must be fitted first.")
        x_arr = np.asarray(x, dtype=np.float64)
        p1 = self._sigmoid(x_arr @ self.weights_ + self.bias_)
        return np.column_stack([1.0 - p1, p1])


def _param_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid dictionary into a list of parameter dicts."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def _build_churn_estimator(
    model_name: str, params: dict[str, Any], seed: int, pos_weight: float
) -> Any:
    """Instantiate a churn classifier given model name and hyperparameter dictionary."""
    if model_name == "logistic_regression":
        return BalancedLogisticRegression(
            c_reg=float(params["C"]),
            max_iter=int(params["max_iter"]),
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            scale_pos_weight=pos_weight,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=float(params["learning_rate"]),
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=int(params["iterations"]),
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=0,
        )
    if model_name == "keras_mlp":
        return KerasChurnClassifier(
            hidden_units=list(params["hidden_units"]),
            dropout_rate=float(params["dropout_rate"]),
            learning_rate=float(params["learning_rate"]),
            batch_size=int(params["batch_size"]),
            epochs=int(params["epochs"]),
            patience=int(params["patience"]),
            seed=seed,
        )
    raise ValueError(f"Unsupported churn model: {model_name}")


def _evaluate_cv_churn(
    model_name: str,
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    threshold: float,
    seed: int,
) -> dict[str, float]:
    """Run stratified K-fold CV on training data only and return averaged metrics."""
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va, y_va = x_train[va_idx], y_train[va_idx]
        pos_weight = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)

        est = _build_churn_estimator(model_name, params, seed + fold_idx, pos_weight)
        est.fit(x_tr, y_tr)
        probs = est.predict_proba(x_va)[:, 1]
        fold_metrics.append(compute_classification_metrics(y_va, probs, threshold=threshold))

    keys = list(fold_metrics[0].keys())
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def train_and_evaluate_churn(
    df_features: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    config: AppConfig,
) -> dict[str, Any]:
    """Tune, train, select, and test all churn classification models."""
    seed = config.project.seed
    churn_cfg = config.models["churn"]
    threshold = float(churn_cfg["decision_threshold"])
    log_path = (
        config.artifacts.experiment_log_path
        if config.artifacts
        else "models_artifacts/experiment_log.csv"
    )

    train_df = df_features[df_features["split"] == "train"].reset_index(drop=True)
    val_df = df_features[df_features["split"] == "validation"].reset_index(drop=True)
    test_df = df_features[df_features["split"] == "test"].reset_index(drop=True)

    x_train = transform_features(preprocessor, train_df, config)
    y_train = train_df["churn_label"].to_numpy(dtype=np.int64)

    x_val = transform_features(preprocessor, val_df, config)
    y_val = val_df["churn_label"].to_numpy(dtype=np.int64)

    x_test = transform_features(preprocessor, test_df, config)
    y_test = test_df["churn_label"].to_numpy(dtype=np.int64)

    folds = stratified_kfold_indices(y_train, n_splits=config.splitting.cv_folds, seed=seed)
    full_pos_weight = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1.0)

    candidates = ("logistic_regression", "xgboost", "lightgbm", "catboost", "keras_mlp")
    val_results: list[dict[str, Any]] = []

    for model_name in candidates:
        model_cfg = churn_cfg[model_name]
        t0 = time.perf_counter()

        if model_name == "keras_mlp":
            best_params = dict(model_cfg)
            cv_metrics = _evaluate_cv_churn(
                model_name, best_params, x_train, y_train, folds, threshold, seed
            )
        else:
            combos = _param_combinations(model_cfg)
            best_score = -1.0
            best_params = combos[0]
            cv_metrics = {}
            for params in combos:
                metrics = _evaluate_cv_churn(
                    model_name, params, x_train, y_train, folds, threshold, seed
                )
                score = 0.6 * metrics["roc_auc"] + 0.4 * metrics["recall"]
                if score > best_score:
                    best_score = score
                    best_params = params
                    cv_metrics = metrics

        cv_elapsed = time.perf_counter() - t0
        log_experiment(
            log_path, "churn", model_name, "cv_train", cv_elapsed, best_params, cv_metrics
        )

        # Fit best CV config on full training set and evaluate on validation set
        t_val = time.perf_counter()
        fitted_model = _build_churn_estimator(model_name, best_params, seed, full_pos_weight)
        fitted_model.fit(x_train, y_train)
        val_elapsed = time.perf_counter() - t_val

        val_probs = fitted_model.predict_proba(x_val)[:, 1]
        val_metrics = compute_classification_metrics(y_val, val_probs, threshold=threshold)
        log_experiment(
            log_path, "churn", model_name, "validation", val_elapsed, best_params, val_metrics
        )

        val_results.append(
            {
                "model_name": model_name,
                "model": fitted_model,
                "params": best_params,
                "val_metrics": val_metrics,
                "train_time_sec": val_elapsed,
            }
        )

    # Select winning model on validation (prioritize ROC-AUC while ensuring high churn recall)
    val_results.sort(
        key=lambda r: (r["val_metrics"]["roc_auc"], r["val_metrics"]["recall"]),
        reverse=True,
    )
    winner = val_results[0]
    best_model = winner["model"]

    # Touch untouched test set ONCE for final reporting
    test_probs = best_model.predict_proba(x_test)[:, 1]
    test_metrics = compute_classification_metrics(y_test, test_probs, threshold=threshold)
    log_experiment(
        log_path,
        "churn",
        winner["model_name"],
        "test",
        winner["train_time_sec"],
        winner["params"],
        test_metrics,
    )

    if config.artifacts is not None:
        out_path = Path(config.artifacts.best_churn_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_bundle = {
            "model": best_model,
            "model_name": winner["model_name"],
            "threshold": threshold,
            "params": winner["params"],
            "val_metrics": winner["val_metrics"],
            "test_metrics": test_metrics,
            "feature_names": preprocessor.get_feature_names_out().tolist(),
        }
        joblib.dump(artifact_bundle, out_path)
        logger.info(
            "Saved best churn model (%s) to %s | test_roc_auc=%.4f | test_recall=%.4f",
            winner["model_name"],
            out_path,
            test_metrics["roc_auc"],
            test_metrics["recall"],
        )

    return {
        "best_model_name": winner["model_name"],
        "val_metrics": winner["val_metrics"],
        "test_metrics": test_metrics,
    }
