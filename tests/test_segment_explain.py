"""Unit tests for RFM K-Means clustering, ProductRecommender, and CustomerExplainer."""

import numpy as np

from src.explainability.shap_lime import CustomerExplainer
from src.models.train_churn import BalancedLogisticRegression
from src.segmentation.recommender import ProductRecommender
from src.segmentation.rfm_clustering import PureKMeans, compute_silhouette_score


def test_pure_kmeans_and_silhouette_score() -> None:
    """PureKMeans separates two well-defined clusters with high Silhouette score."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=-3.0, scale=0.3, size=(30, 3))
    c2 = rng.normal(loc=3.0, scale=0.3, size=(30, 3))
    x = np.vstack([c1, c2])

    km = PureKMeans(n_clusters=2, seed=42).fit(x)
    labels = km.predict(x)
    assert len(np.unique(labels)) == 2
    assert compute_silhouette_score(x, labels) > 0.80


def test_product_recommender_basket_and_fallback() -> None:
    """Recommender excludes owned items and falls back to bestsellers on unknown SKUs."""
    rules = {
        "SKU_A": [
            {
                "stock_code": "SKU_B",
                "description": "ITEM B",
                "lift": 3.5,
                "confidence": 0.4,
                "support": 0.05,
                "score": 1.4,
                "reason": "Co-purchased with SKU_A",
            },
            {
                "stock_code": "SKU_C",
                "description": "ITEM C",
                "lift": 2.0,
                "confidence": 0.3,
                "support": 0.04,
                "score": 0.6,
                "reason": "Co-purchased with SKU_A",
            },
        ]
    }
    fallback = [
        {
            "stock_code": "SKU_TOP",
            "description": "TOP SELLER",
            "lift": 1.0,
            "confidence": 0.0,
            "support": 0.0,
            "score": 500.0,
            "reason": "Top storewide bestseller",
        }
    ]
    rec = ProductRecommender(rules, fallback, {"SKU_A": "ITEM A", "SKU_B": "ITEM B"})

    basket_recs = rec.recommend_for_basket(["SKU_A", "SKU_B"], top_n=2)
    rec_codes = [r["stock_code"] for r in basket_recs]
    assert "SKU_A" not in rec_codes
    assert "SKU_B" not in rec_codes
    assert "SKU_C" in rec_codes

    cold_recs = rec.recommend_for_item("UNKNOWN_999", top_n=1)
    assert next(iter(cold_recs))["stock_code"] == "SKU_TOP"


def test_customer_explainer_local_attributions() -> None:
    """CustomerExplainer produces sorted feature contributions and valid probabilities."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 3))
    y = (x[:, 0] > 0).astype(np.int64)
    clf = BalancedLogisticRegression(c_reg=1.0, max_iter=200).fit(x, y)

    explainer = CustomerExplainer(
        model=clf,
        feature_names=["recency_days", "frequency", "monetary_net"],
        background_mean=np.zeros(3),
        global_importance=[],
        model_name="logistic_regression",
    )
    res = explainer.explain_instance(np.array([2.5, -1.0, 0.0]), top_k=3)
    assert res["model_name"] == "logistic_regression"
    assert 0.0 <= res["predicted_probability"] <= 1.0
    assert len(res["top_drivers"]) == 3
    assert next(iter(res["top_drivers"]))["feature"] == "recency_days"
