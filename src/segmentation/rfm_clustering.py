"""RFM customer segmentation via K-Means with Silhouette scoring and persona labeling."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PureKMeans:
    """Deterministic K-Means++ clustering implemented in pure NumPy."""

    def __init__(self, n_clusters: int = 4, max_iter: int = 200, seed: int = 42) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.centroids_: np.ndarray | None = None
        self.inertia_: float = 0.0

    def _init_centroids(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = len(x)
        centroids = [x[int(rng.integers(0, n_samples))]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                np.stack([np.sum((x - c) ** 2, axis=1) for c in centroids], axis=1),
                axis=1,
            )
            total = float(np.sum(dist_sq))
            probs = dist_sq / total if total > 0 else np.full(n_samples, 1.0 / n_samples)
            centroids.append(x[int(rng.choice(n_samples, p=probs))])
        return np.vstack(centroids)

    def fit(self, x: np.ndarray) -> "PureKMeans":
        x_arr = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        centroids = self._init_centroids(x_arr, rng)

        for _ in range(self.max_iter):
            dists = np.stack([np.sum((x_arr - c) ** 2, axis=1) for c in centroids], axis=1)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                members = x_arr[labels == k]
                new_centroids[k] = (
                    np.mean(members, axis=0)
                    if len(members) > 0
                    else x_arr[int(rng.integers(0, len(x_arr)))]
                )
            if np.allclose(centroids, new_centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        final_dists = np.stack([np.sum((x_arr - c) ** 2, axis=1) for c in centroids], axis=1)
        self.centroids_ = centroids
        self.inertia_ = float(np.sum(np.min(final_dists, axis=1)))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("PureKMeans must be fitted before predict.")
        x_arr = np.asarray(x, dtype=np.float64)
        dists = np.stack([np.sum((x_arr - c) ** 2, axis=1) for c in self.centroids_], axis=1)
        return np.argmin(dists, axis=1)


def compute_silhouette_score(x: np.ndarray, labels: np.ndarray, max_samples: int = 1200) -> float:
    """Compute mean Silhouette Coefficient in pure NumPy."""
    x_arr = np.asarray(x, dtype=np.float64)
    lbl_arr = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(lbl_arr)
    if len(unique_labels) < 2:
        return 0.0

    if len(x_arr) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_arr), size=max_samples, replace=False)
        x_arr = x_arr[idx]
        lbl_arr = lbl_arr[idx]

    diff = x_arr[:, None, :] - x_arr[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    scores = np.zeros(len(x_arr), dtype=np.float64)
    for i in range(len(x_arr)):
        own = lbl_arr[i]
        same_mask = (lbl_arr == own) & (np.arange(len(x_arr)) != i)
        a_i = float(np.mean(dist_matrix[i, same_mask])) if np.any(same_mask) else 0.0
        b_i = min(
            float(np.mean(dist_matrix[i, lbl_arr == other]))
            for other in unique_labels
            if other != own and np.any(lbl_arr == other)
        )
        scores[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)
    return float(np.mean(scores))


def _assign_persona_labels(rfm_df: pd.DataFrame) -> dict[int, str]:
    summary = rfm_df.groupby("cluster_id").agg(
        recency=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary_net", "mean"),
    )
    composite = (
        summary["recency"].rank(ascending=False)
        + summary["frequency"].rank(ascending=True)
        + summary["monetary"].rank(ascending=True)
    )
    ordered = composite.sort_values(ascending=False).index.tolist()
    names = ["Champions", "Loyal Customers", "At-Risk High-Value", "Hibernating / Lost"]
    return {
        int(cid): names[i] if i < len(names) else f"Segment {i + 1}"
        for i, cid in enumerate(ordered)
    }


def fit_rfm_segmentation(df_features: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Fit RFM K-Means segmentation on training customers, score all customers, and save artifact."""
    seg_cfg = config.models["segmentation"]
    rfm_cols: list[str] = list(seg_cfg["rfm_features"])
    default_k = int(seg_cfg["default_k"])
    seed = config.project.seed

    train_df = df_features[df_features["split"] == "train"].copy()
    x_tr_log = np.log1p(np.maximum(train_df[rfm_cols].to_numpy(dtype=np.float64), 0.0))
    means = np.mean(x_tr_log, axis=0)
    stds = np.where(np.std(x_tr_log, axis=0) > 1e-9, np.std(x_tr_log, axis=0), 1.0)
    x_tr_scaled = (x_tr_log - means) / stds

    k_diagnostics: list[dict[str, float]] = []
    for k in [int(v) for v in seg_cfg["k_range"]]:
        km = PureKMeans(n_clusters=k, seed=seed).fit(x_tr_scaled)
        sil = compute_silhouette_score(x_tr_scaled, km.predict(x_tr_scaled))
        k_diagnostics.append({"k": float(k), "inertia": km.inertia_, "silhouette": sil})

    final_km = PureKMeans(n_clusters=default_k, seed=seed).fit(x_tr_scaled)
    train_df["cluster_id"] = final_km.predict(x_tr_scaled)
    persona_map = _assign_persona_labels(train_df)

    x_all_scaled = (
        np.log1p(np.maximum(df_features[rfm_cols].to_numpy(dtype=np.float64), 0.0)) - means
    ) / stds
    scored_df = df_features[
        ["customer_id", "split", *rfm_cols, "churn_label", "clv_90d_target"]
    ].copy()
    scored_df["cluster_id"] = final_km.predict(x_all_scaled)
    scored_df["segment_name"] = scored_df["cluster_id"].map(persona_map)

    profiles = (
        scored_df.groupby(["cluster_id", "segment_name"], as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary_net=("monetary_net", "mean"),
            churn_rate=("churn_label", "mean"),
            avg_clv_90d=("clv_90d_target", "mean"),
        )
        .sort_values("avg_monetary_net", ascending=False)
    )

    if config.artifacts is not None:
        out_path = Path(config.artifacts.kmeans_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final_km,
                "rfm_cols": rfm_cols,
                "means": means,
                "stds": stds,
                "persona_map": persona_map,
                "k_diagnostics": k_diagnostics,
                "segment_profiles": profiles.to_dict(orient="records"),
            },
            out_path,
        )

    return {
        "default_k": default_k,
        "persona_map": persona_map,
        "k_diagnostics": k_diagnostics,
        "segment_profiles": profiles.to_dict(orient="records"),
        "scored_df": scored_df,
    }
