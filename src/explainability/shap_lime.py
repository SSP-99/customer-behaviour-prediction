"""Global and local model explainability using SHAP with deterministic fallback."""

from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.preprocessing import TabularPreprocessor, transform_features
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerExplainer:
    """Global and per-customer feature attribution explainer persisted via joblib."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_mean: np.ndarray,
        global_importance: list[dict[str, Any]],
        model_name: str = "churn_model",
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.background_mean = np.asarray(background_mean, dtype=np.float64)
        self.global_importance = global_importance
        self.model_name = model_name

    def _predict_positive_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(x_arr)[:, 1], dtype=np.float64)
        return np.asarray(self.model.predict(x_arr), dtype=np.float64)

    def explain_instance(self, x_row: np.ndarray, top_k: int = 8) -> dict[str, Any]:
        """Compute local feature contributions for a single preprocessed customer vector."""
        row = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
        base_prob = float(self._predict_positive_prob(self.background_mean.reshape(1, -1)))
        pred_prob = float(self._predict_positive_prob(row))

        shap_vals: np.ndarray | None = None
        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            raw_sv = explainer.shap_values(row)
            if isinstance(raw_sv, list):
                shap_vals = np.asarray(raw_sv[-1], dtype=np.float64).reshape(-1)
            else:
                arr_sv = np.asarray(raw_sv, dtype=np.float64)
                shap_vals = arr_sv[..., -1].reshape(-1) if arr_sv.ndim == 3 else arr_sv.reshape(-1)
        except Exception:
            n_feat = row.shape[-1]
            batch = np.repeat(row, n_feat, axis=0)
            for j in range(n_feat):
                batch[j, j] = self.background_mean[j]
            ablated_probs = self._predict_positive_prob(batch)
            shap_vals = pred_prob - ablated_probs

        contributions: list[dict[str, Any]] = []
        for idx, fname in enumerate(self.feature_names):
            val = float(shap_vals[idx])
            contributions.append(
                {
                    "feature": fname,
                    "shap_value": round(val, 6),
                    "abs_impact": round(abs(val), 6),
                    "direction": "increases_churn_risk" if val >= 0 else "reduces_churn_risk",
                }
            )

        contributions.sort(key=lambda d: d["abs_impact"], reverse=True)
        return {
            "model_name": self.model_name,
            "base_probability": round(base_prob, 6),
            "predicted_probability": round(pred_prob, 6),
            "top_drivers": contributions[:top_k],
        }


def build_and_save_explainer(
    df_features: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    churn_bundle: dict[str, Any],
    config: AppConfig,
) -> CustomerExplainer:
    """Compute global SHAP importances, generate summary chart, and save CustomerExplainer."""
    model = churn_bundle["model"]
    model_name = str(churn_bundle.get("model_name", "churn_model"))
    feature_names: list[str] = list(
        churn_bundle.get("feature_names", preprocessor.get_feature_names_out().tolist())
    )

    train_df = df_features[df_features["split"] == "train"].reset_index(drop=True)
    test_df = df_features[df_features["split"] == "test"].reset_index(drop=True)

    x_train = transform_features(preprocessor, train_df, config)
    x_test = transform_features(preprocessor, test_df, config)
    bg_mean = np.mean(x_train, axis=0)

    sample_n = min(len(x_test), 200)
    x_sample = x_test[:sample_n]

    temp_explainer = CustomerExplainer(
        model=model,
        feature_names=feature_names,
        background_mean=bg_mean,
        global_importance=[],
        model_name=model_name,
    )

    abs_matrix = np.zeros((sample_n, len(feature_names)), dtype=np.float64)
    for i in range(sample_n):
        local = temp_explainer.explain_instance(x_sample[i], top_k=len(feature_names))
        impact_map = {d["feature"]: d["abs_impact"] for d in local["top_drivers"]}
        for j, fname in enumerate(feature_names):
            abs_matrix[i, j] = impact_map.get(fname, 0.0)

    mean_abs = np.mean(abs_matrix, axis=0)
    global_importance = [
        {"feature": fname, "mean_abs_shap": round(float(mean_abs[j]), 6)}
        for j, fname in enumerate(feature_names)
    ]
    global_importance.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
    temp_explainer.global_importance = global_importance

    fig_path = Path("docs/figures/global_feature_importance.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    top_15 = global_importance[:15][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([d["feature"] for d in top_15], [d["mean_abs_shap"] for d in top_15], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP Attribution| (Impact on Churn Probability)")
    ax.set_title(f"Top 15 Global Churn Drivers ({model_name})")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Saved global feature importance chart to %s", fig_path)

    if config.artifacts is not None:
        out_path = Path(config.artifacts.shap_explainer_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(temp_explainer, out_path)
        logger.info("Saved CustomerExplainer artifact to %s", out_path)

    return temp_explainer
