"""90-day CLV regression training, 5-fold CV tuning, validation selection, and test evaluation."""

import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features.preprocessing import TabularPreprocessor, transform_features
from src.models.metrics import compute_regression_metrics, kfold_indices
from src.models.neural_nets import KerasCLVRegressor
from src.utils.config import AppConfig
from src.utils.experiment_log import log_experiment
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClosedFormRidgeRegressor:
    """Pure NumPy L2-regularized Ridge regressor trained on log1p-transformed targets."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize L2 penalty strength."""
        self.alpha = alpha
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClosedFormRidgeRegressor":
        """Solve (X^T X + alpha I) w = X^T y_log in closed form."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_log = np.log1p(np.maximum(np.asarray(y, dtype=np.float64), 0.0))

        x_mean = np.mean(x_arr, axis=0)
        y_mean = float(np.mean(y_log))
        x_centered = x_arr - x_mean
        y_centered = y_log - y_mean

        n_features = x_arr.shape[1]
        gram = x_centered.T @ x_centered + self.alpha * np.eye(n_features)
        self.weights_ = np.linalg.solve(gram, x_centered.T @ y_centered)
        self.bias_ = y_mean - float(x_mean @ self.weights_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict non-negative 90-day CLV on original currency scale."""
        if self.weights_ is None:
            raise RuntimeError("ClosedFormRidgeRegressor must be fitted first.")
        x_arr = np.asarray(x, dtype=np.float64)
        pred_log = np.clip(x_arr @ self.weights_ + self.bias_, 0.0, 15.0)
        return np.maximum(np.expm1(pred_log), 0.0)


class LogTargetTreeRegressor:
    """Wrapper fitting a gradient-boosted regressor on log1p(y) and inverting via expm1."""

    def __init__(self, base_estimator: Any) -> None:
        """Wrap any regressor supporting fit(x, y) and predict(x)."""
        self.base_estimator = base_estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogTargetTreeRegressor":
        """Fit underlying tree regressor on log1p-transformed spend."""
        y_log = np.log1p(np.maximum(np.asarray(y, dtype=np.float64), 0.0))
        self.base_estimator.fit(x, y_log)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict non-negative 90-day CLV on original currency scale."""
        pred_log = np.clip(np.asarray(self.base_estimator.predict(x), dtype=np.float64), 0.0, 15.0)
        return np.maximum(np.expm1(pred_log), 0.0)


def _param_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand hyperparameter grid into list of parameter dictionaries."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def _build_clv_estimator(model_name: str, params: dict[str, Any], seed: int) -> Any:
    """Instantiate a CLV regression estimator given model name and hyperparameters."""
    if model_name == "ridge":
        return ClosedFormRidgeRegressor(alpha=float(params["alpha"]))
    if model_name == "xgboost":
        return LogTargetTreeRegressor(
            XGBRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
            )
        )
    if model_name == "lightgbm":
        return LogTargetTreeRegressor(
            lgb.LGBMRegressor(
                n_estimators=int(params["n_estimators"]),
                num_leaves=int(params["num_leaves"]),
                learning_rate=float(params["learning_rate"]),
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            )
        )
    if model_name == "keras_mlp":
        return KerasCLVRegressor(
            hidden_units=list(params["hidden_units"]),
            dropout_rate=float(params["dropout_rate"]),
            learning_rate=float(params["learning_rate"]),
            batch_size=int(params["batch_size"]),
            epochs=int(params["epochs"]),
            patience=int(params["patience"]),
            seed=seed,
        )
    raise ValueError(f"Unsupported CLV model: {model_name}")


def _evaluate_cv_clv(
    model_name: str,
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, float]:
    """Run K-fold CV on training data only and return averaged regression metrics."""
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        est = _build_clv_estimator(model_name, params, seed + fold_idx)
        est.fit(x_train[tr_idx], y_train[tr_idx])
        preds = est.predict(x_train[va_idx])
        fold_metrics.append(compute_regression_metrics(y_train[va_idx], preds))

    first_fold = next(iter(fold_metrics))
    keys = list(first_fold.keys())
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def train_and_evaluate_clv(
    df_features: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    config: AppConfig,
) -> dict[str, Any]:
    """Tune, train, select, and test all 90-day CLV regression models."""
    seed = config.project.seed
    clv_cfg = config.models["clv"]
    log_path = (
        config.artifacts.experiment_log_path
        if config.artifacts
        else "models_artifacts/experiment_log.csv"
    )

    train_df = df_features[df_features["split"] == "train"].reset_index(drop=True)
    val_df = df_features[df_features["split"] == "validation"].reset_index(drop=True)
    test_df = df_features[df_features["split"] == "test"].reset_index(drop=True)

    x_train = transform_features(preprocessor, train_df, config)
    y_train = train_df["clv_90d_target"].to_numpy(dtype=np.float64)

    x_val = transform_features(preprocessor, val_df, config)
    y_val = val_df["clv_90d_target"].to_numpy(dtype=np.float64)

    x_test = transform_features(preprocessor, test_df, config)
    y_test = test_df["clv_90d_target"].to_numpy(dtype=np.float64)

    folds = kfold_indices(len(train_df), n_splits=config.splitting.cv_folds, seed=seed)
    candidates = ("ridge", "xgboost", "lightgbm", "keras_mlp")
    val_results: list[dict[str, Any]] = []

    for model_name in candidates:
        model_cfg = clv_cfg[model_name]
        t0 = time.perf_counter()

        if model_name == "keras_mlp":
            best_params = dict(model_cfg)
            cv_metrics = _evaluate_cv_clv(model_name, best_params, x_train, y_train, folds, seed)
        else:
            combos = _param_combinations(model_cfg)
            best_r2 = -1e9
            best_params = next(iter(combos))
            cv_metrics = {}
            for params in combos:
                metrics = _evaluate_cv_clv(model_name, params, x_train, y_train, folds, seed)
                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    best_params = params
                    cv_metrics = metrics

        cv_elapsed = time.perf_counter() - t0
        log_experiment(log_path, "clv", model_name, "cv_train", cv_elapsed, best_params, cv_metrics)

        t_val = time.perf_counter()
        fitted_model = _build_clv_estimator(model_name, best_params, seed)
        fitted_model.fit(x_train, y_train)
        val_elapsed = time.perf_counter() - t_val

        val_preds = fitted_model.predict(x_val)
        val_metrics = compute_regression_metrics(y_val, val_preds)
        log_experiment(
            log_path, "clv", model_name, "validation", val_elapsed, best_params, val_metrics
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

    val_results.sort(key=lambda r: r["val_metrics"]["r2"], reverse=True)
    winner = next(iter(val_results))
    best_model = winner["model"]

    test_preds = best_model.predict(x_test)
    test_metrics = compute_regression_metrics(y_test, test_preds)
    log_experiment(
        log_path,
        "clv",
        winner["model_name"],
        "test",
        winner["train_time_sec"],
        winner["params"],
        test_metrics,
    )

    if config.artifacts is not None:
        out_path = Path(config.artifacts.best_clv_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_bundle = {
            "model": best_model,
            "model_name": winner["model_name"],
            "params": winner["params"],
            "val_metrics": winner["val_metrics"],
            "test_metrics": test_metrics,
            "feature_names": preprocessor.get_feature_names_out().tolist(),
        }
        joblib.dump(artifact_bundle, out_path)
        logger.info(
            "Saved best CLV model (%s) to %s | test_r2=%.4f | test_mae=%.2f",
            winner["model_name"],
            out_path,
            test_metrics["r2"],
            test_metrics["mae"],
        )

    return {
        "best_model_name": winner["model_name"],
        "val_metrics": winner["val_metrics"],
        "test_metrics": test_metrics,
    }
