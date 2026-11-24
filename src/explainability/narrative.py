"""Plain-language marketing narrative generator, LIME local explanation, and batch SHAP export."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.preprocessing import TabularPreprocessor, transform_features
from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

FRIENDLY_NAMES: dict[str, str] = {
    "recency_days": "days since last order",
    "frequency": "total historical orders",
    "monetary_gross": "total gross spend",
    "monetary_net": "net historical spend",
    "avg_order_value": "average order value",
    "unique_products": "distinct products purchased",
    "active_span_days": "active purchasing span (days)",
    "purchase_frequency_per_month": "monthly purchase velocity",
    "recency_to_cadence_ratio": "inactivity vs. normal cycle ratio",
    "spend_trend_ratio": "recent 30-day spend momentum",
    "return_rate_value": "monetary return rate",
}


def classify_risk_tier(churn_prob: float) -> tuple[str, str]:
    """Map churn probability into Low/Medium/High tier and recommended marketing action."""
    if churn_prob >= 0.65:
        return (
            "High",
            "Trigger immediate win-back campaign: 15% loyalty voucher + account outreach within 48h.",
        )
    if churn_prob >= 0.40:
        return (
            "Medium",
            "Enroll in automated replenishment reminder + personalized top-affinity bundle email.",
        )
    return (
        "Low",
        "Maintain VIP engagement: offer early access to seasonal catalog and cross-sell bundles.",
    )


def build_plain_language_narrative(
    customer_row: pd.Series,
    feature_df: pd.DataFrame,
    top_drivers: list[dict[str, Any]],
    churn_prob: float,
) -> dict[str, Any]:
    """Convert top SHAP attributions into a plain-English explanation with percentile context."""
    tier, action = classify_risk_tier(churn_prob)
    pct_prob = round(100.0 * churn_prob, 1)

    risk_up: list[str] = []
    risk_down: list[str] = []

    for drv in top_drivers[:5]:
        fname = str(drv["feature"])
        friendly = FRIENDLY_NAMES.get(fname, fname.replace("_", " "))
        if fname in feature_df.columns and fname in customer_row.index:
            val = float(customer_row[fname])
            pct = float((feature_df[fname] <= val).mean() * 100.0)
            phrase = f"{friendly} ({val:.1f}, higher than {pct:.0f}% of customers)"
        else:
            phrase = friendly

        if drv["direction"] == "increases_churn_risk":
            risk_up.append(phrase)
        else:
            risk_down.append(phrase)

    up_text = (
        f"driven primarily by {', '.join(risk_up[:2])}"
        if risk_up
        else "with minimal upward risk drivers"
    )
    down_text = (
        f" However, {', '.join(risk_down[:2])} helps moderate the risk." if risk_down else ""
    )
    narrative = f"This customer is flagged as {tier.upper()} churn risk ({pct_prob}%), {up_text}.{down_text}"

    return {
        "risk_tier": tier,
        "churn_probability": round(churn_prob, 4),
        "narrative": narrative,
        "recommended_action": action,
    }


def explain_with_lime_or_surrogate(
    x_train: np.ndarray,
    x_instance: np.ndarray,
    model: Any,
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Compute local LIME feature weights (with local ridge surrogate fallback)."""
    row = np.asarray(x_instance, dtype=np.float64).reshape(1, -1)
    try:
        from lime.lime_tabular import LimeTabularExplainer

        lime_exp = LimeTabularExplainer(
            training_data=np.asarray(x_train, dtype=np.float64),
            feature_names=feature_names,
            class_names=["Retained", "Churn"],
            mode="classification",
            random_state=42,
        )
        exp = lime_exp.explain_instance(
            row.reshape(-1),
            model.predict_proba,
            num_features=top_k,
            num_samples=250,
        )
        return [
            {
                "rule": str(rule),
                "weight": round(float(w), 6),
                "direction": "increases_churn_risk" if w >= 0 else "reduces_churn_risk",
            }
            for rule, w in exp.as_list()
        ]
    except Exception:
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.25, size=(200, row.shape[-1]))
        pert = row + noise
        probs = model.predict_proba(pert)[:, 1]
        gram = noise.T @ noise + 1.0 * np.eye(row.shape[-1])
        weights = np.linalg.solve(gram, noise.T @ (probs - float(np.mean(probs))))
        ranked_idx = np.argsort(np.abs(weights))[::-1][:top_k]
        return [
            {
                "rule": f"{feature_names[int(i)]} (local slope)",
                "weight": round(float(weights[int(i)]), 6),
                "direction": "increases_churn_risk"
                if weights[int(i)] >= 0
                else "reduces_churn_risk",
            }
            for i in ranked_idx
        ]


def precompute_all_explanations(config: AppConfig) -> Path:
    """Precompute SHAP top drivers and plain-language narratives for all customers."""
    if config.artifacts is None:
        raise ValueError("Artifacts configuration must be defined.")

    df_features = pd.read_parquet(config.data.processed_features_path)
    preprocessor: TabularPreprocessor = joblib.load(config.artifacts.preprocessor_path)
    explainer = joblib.load(config.artifacts.shap_explainer_path)
    churn_bundle = joblib.load(config.artifacts.best_churn_model_path)
    model = churn_bundle["model"]

    x_all = transform_features(preprocessor, df_features, config)
    probs = model.predict_proba(x_all)[:, 1]

    records: list[dict[str, Any]] = []
    for idx in range(len(df_features)):
        row = df_features.iloc[idx]
        local = explainer.explain_instance(x_all[idx], top_k=5)
        narr = build_plain_language_narrative(
            row, df_features, local["top_drivers"], float(probs[idx])
        )
        records.append(
            {
                "customer_id": str(row["customer_id"]),
                "churn_probability": round(float(probs[idx]), 6),
                "risk_tier": narr["risk_tier"],
                "narrative": narr["narrative"],
                "recommended_action": narr["recommended_action"],
                "top_drivers_json": pd.Series(local["top_drivers"]).to_json(orient="values"),
            }
        )

    out_path = Path("data/processed/shap_values.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)
    logger.info("Precomputed SHAP narratives for %d customers at %s", len(records), out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    precompute_all_explanations(cfg)
