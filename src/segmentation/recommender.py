"""Market-basket item-to-item product recommendation engine."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProductRecommender:
    """Lookup-based item-to-item recommender ranked by lift and confidence with cold-start fallback."""

    def __init__(
        self,
        rules_by_item: dict[str, list[dict[str, Any]]],
        popular_fallback: list[dict[str, Any]],
        description_map: dict[str, str],
    ) -> None:
        self.rules_by_item = rules_by_item
        self.popular_fallback = popular_fallback
        self.description_map = description_map

    def recommend_for_item(self, stock_code: str, top_n: int = 5) -> list[dict[str, Any]]:
        """Return top-N co-purchased products for a given StockCode."""
        code = str(stock_code).strip().upper()
        matches = self.rules_by_item.get(code, [])
        return matches[:top_n] if matches else self.popular_fallback[:top_n]

    def recommend_for_basket(self, stock_codes: list[str], top_n: int = 5) -> list[dict[str, Any]]:
        """Aggregate recommendations across a customer's purchased StockCodes, excluding owned items."""
        owned = {str(c).strip().upper() for c in stock_codes if str(c).strip()}
        candidate_scores: dict[str, dict[str, Any]] = {}

        for code in owned:
            for rec in self.rules_by_item.get(code, []):
                target = rec["stock_code"]
                if target in owned:
                    continue
                score = float(rec["lift"]) * float(rec["confidence"])
                if target not in candidate_scores or score > candidate_scores[target]["score"]:
                    candidate_scores[target] = {
                        "stock_code": target,
                        "description": rec["description"],
                        "lift": rec["lift"],
                        "confidence": rec["confidence"],
                        "support": rec["support"],
                        "score": round(score, 6),
                        "reason": f"Frequently bought with {code}",
                    }

        ranked = sorted(candidate_scores.values(), key=lambda r: r["score"], reverse=True)
        if len(ranked) < top_n:
            seen = {r["stock_code"] for r in ranked} | owned
            for fb in self.popular_fallback:
                if fb["stock_code"] not in seen:
                    ranked.append(fb)
                    seen.add(fb["stock_code"])
                if len(ranked) >= top_n:
                    break
        return ranked[:top_n]


def fit_and_save_recommender(config: AppConfig) -> ProductRecommender:
    """Build ProductRecommender from processed product association pairs and persist via joblib."""
    assoc_df = pd.read_parquet(config.data.processed_product_pairs_path)
    top_n = int(config.models["recommender"]["top_n"])

    rules_by_item: dict[str, list[dict[str, Any]]] = {}
    description_map: dict[str, str] = {}

    if not assoc_df.empty:
        for row in assoc_df.itertuples(index=False):
            item_a, item_b = str(row.item_a), str(row.item_b)
            description_map[item_a] = str(row.item_a_desc)
            description_map[item_b] = str(row.item_b_desc)
            bucket = rules_by_item.setdefault(item_a, [])
            if len(bucket) < top_n * 3:
                bucket.append(
                    {
                        "stock_code": item_b,
                        "description": str(row.item_b_desc),
                        "lift": float(row.lift),
                        "confidence": float(row.confidence),
                        "support": float(row.support),
                        "score": round(float(row.lift) * float(row.confidence), 6),
                        "reason": f"Co-purchased with {item_a}",
                    }
                )

        pop_df = (
            assoc_df.groupby(["item_a", "item_a_desc"], as_index=False)["co_occurrence_count"]
            .sum()
            .sort_values("co_occurrence_count", ascending=False)
            .head(top_n * 3)
        )
        popular_fallback = [
            {
                "stock_code": str(r.item_a),
                "description": str(r.item_a_desc),
                "lift": 1.0,
                "confidence": 0.0,
                "support": 0.0,
                "score": float(r.co_occurrence_count),
                "reason": "Top storewide bestseller",
            }
            for r in pop_df.itertuples(index=False)
        ]
    else:
        popular_fallback = []

    recommender = ProductRecommender(rules_by_item, popular_fallback, description_map)
    if config.artifacts is not None:
        out_path = Path(config.artifacts.recommender_rules_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(recommender, out_path)
        logger.info("Saved ProductRecommender (%d products) to %s", len(rules_by_item), out_path)
    return recommender
