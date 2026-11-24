"""Symmetric Keras Autoencoder for unsupervised customer spend-pattern anomaly detection."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)

ANOMALY_FEATURE_COLS = [
    "frequency",
    "monetary_gross",
    "avg_order_value",
    "max_order_value",
    "avg_basket_size",
    "return_rate_value",
    "bulk_order_lines",
]


class SpendAnomalyAutoencoder:
    """Symmetric 32-16-8-16-32 dense autoencoder trained on normal training spend profiles."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        percentile_threshold: float = 95.0,
        epochs: int = 40,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.feature_cols = feature_cols or list(ANOMALY_FEATURE_COLS)
        self.percentile_threshold = percentile_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.threshold_: float = 0.0
        self.model_: tf.keras.Model | None = None

    def _prepare_matrix(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cols = [c for c in self.feature_cols if c in df.columns]
        raw = np.log1p(np.maximum(df[cols].fillna(0.0).to_numpy(dtype=np.float32), 0.0))
        if fit:
            self.feature_cols = cols
            self.means_ = np.mean(raw, axis=0)
            std = np.std(raw, axis=0)
            self.stds_ = np.where(std > 1e-6, std, 1.0)
        assert self.means_ is not None and self.stds_ is not None
        return (raw - self.means_) / self.stds_

    def fit(self, train_df: pd.DataFrame) -> "SpendAnomalyAutoencoder":
        """Fit autoencoder on training customers only and set reconstruction threshold."""
        tf.keras.utils.set_random_seed(self.seed)
        x_tr = self._prepare_matrix(train_df, fit=True)
        in_dim = x_tr.shape[-1]

        inp = tf.keras.Input(shape=(in_dim,))
        x = tf.keras.layers.Dense(32, activation="relu")(inp)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        bottleneck = tf.keras.layers.Dense(8, activation="relu")(x)
        x = tf.keras.layers.Dense(16, activation="relu")(bottleneck)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out = tf.keras.layers.Dense(in_dim, activation="linear")(x)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )
        model.fit(
            x_tr,
            x_tr,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            verbose=0,
        )
        self.model_ = model

        recon = model.predict(x_tr, verbose=0)
        train_mse = np.mean((x_tr - recon) ** 2, axis=1)
        self.threshold_ = float(np.percentile(train_mse, self.percentile_threshold))
        logger.info(
            "Fitted SpendAnomalyAutoencoder | threshold_p%.0f=%.4f",
            self.percentile_threshold,
            self.threshold_,
        )
        return self

    def score_anomaly(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Compute anomaly_score, is_anomaly flag, and top reconstruction error feature per row."""
        if self.model_ is None:
            raise RuntimeError("SpendAnomalyAutoencoder must be fitted before scoring.")
        x = self._prepare_matrix(features_df, fit=False)
        recon = self.model_.predict(x, verbose=0)
        sq_err = (x - recon) ** 2
        mse = np.mean(sq_err, axis=1)
        top_idx = np.argmax(sq_err, axis=1)
        top_reasons = [self.feature_cols[int(i)] for i in top_idx]

        out = pd.DataFrame(
            {
                "anomaly_score": np.round(mse, 6),
                "is_anomaly": (mse > self.threshold_).astype(int),
                "anomaly_reason": top_reasons,
            },
            index=features_df.index,
        )
        if "customer_id" in features_df.columns:
            out.insert(0, "customer_id", features_df["customer_id"].values)
        return out


def fit_and_save_autoencoder(
    df_features: pd.DataFrame,
    artifacts_dir: str | Path = "models_artifacts",
    processed_dir: str | Path = "data/processed",
    seed: int = 42,
) -> dict[str, Any]:
    """Train autoencoder on training split, score all customers, and save artifacts + anomalies."""
    train_df = df_features[df_features["split"] == "train"].reset_index(drop=True)
    ae = SpendAnomalyAutoencoder(seed=seed).fit(train_df)
    scored = ae.score_anomaly(df_features)

    art_path = Path(artifacts_dir) / "autoencoder.joblib"
    art_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ae, art_path)

    anom_path = Path(processed_dir) / "anomalies.parquet"
    anom_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(anom_path, index=False)

    logger.info(
        "Saved Autoencoder to %s | flagged_anomalies=%d / %d",
        art_path,
        int(scored["is_anomaly"].sum()),
        len(scored),
    )
    return {
        "threshold": ae.threshold_,
        "anomaly_count": int(scored["is_anomaly"].sum()),
        "anomalies_path": str(anom_path),
    }
