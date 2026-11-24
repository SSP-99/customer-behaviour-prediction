"""Country one-hot encoding, StockCode frequency encoding, and StandardScaler persistence."""

from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEncoders:
    """Fit/transform encoder for Country one-hot (rare -> 'Other') and StockCode frequency."""

    def __init__(self, top_countries: list[str] | None = None, min_country_freq: int = 20) -> None:
        """Initialize encoder settings."""
        self.top_countries = top_countries
        self.min_country_freq = min_country_freq
        self.allowed_countries_: list[str] = []
        self.country_columns_: list[str] = []
        self.stockcode_freq_map_: dict[str, float] = {}

    def fit(
        self,
        customer_df: pd.DataFrame,
        history_tx_df: pd.DataFrame | None = None,
    ) -> Self:
        """Fit country categories and StockCode purchase frequencies strictly on training data."""
        if "country" in customer_df.columns:
            countries = customer_df["country"].fillna("Other").astype(str).str.strip()
        elif "country_grouped" in customer_df.columns:
            countries = customer_df["country_grouped"].fillna("Other").astype(str).str.strip()
        else:
            countries = pd.Series(["Other"] * len(customer_df), index=customer_df.index)

        if self.top_countries is not None:
            keep = sorted(set(self.top_countries))
        else:
            counts = countries.value_counts()
            keep = sorted(counts[counts >= self.min_country_freq].index.tolist())

        if "Other" not in keep:
            keep.append("Other")
        self.allowed_countries_ = keep
        self.country_columns_ = [f"country_ohe_{c.replace(' ', '_')}" for c in keep]

        if (
            history_tx_df is not None
            and not history_tx_df.empty
            and "stock_code" in history_tx_df.columns
        ):
            codes = history_tx_df["stock_code"].astype(str).str.strip().str.upper()
            freqs = codes.value_counts(normalize=True)
            self.stockcode_freq_map_ = {str(k): float(v) for k, v in freqs.items()}
        else:
            self.stockcode_freq_map_ = {}

        return self

    def transform(
        self,
        customer_df: pd.DataFrame,
        history_tx_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Transform customer rows with fitted Country OHE and StockCode frequency features."""
        out = customer_df.copy()
        if "country" in out.columns:
            raw_c = out["country"].fillna("Other").astype(str).str.strip()
        elif "country_grouped" in out.columns:
            raw_c = out["country_grouped"].fillna("Other").astype(str).str.strip()
        else:
            raw_c = pd.Series(["Other"] * len(out), index=out.index)

        grouped_c = raw_c.where(raw_c.isin(set(self.allowed_countries_)), "Other")
        for country_val, col_name in zip(
            self.allowed_countries_, self.country_columns_, strict=True
        ):
            out[col_name] = (grouped_c == country_val).astype(np.float64)

        if (
            history_tx_df is not None
            and not history_tx_df.empty
            and "customer_id" in history_tx_df.columns
        ):
            tx = history_tx_df.copy()
            tx["customer_id"] = tx["customer_id"].astype(str).str.strip()
            tx["sku_freq"] = (
                tx["stock_code"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(self.stockcode_freq_map_)
                .fillna(0.0)
            )
            agg = tx.groupby("customer_id")["sku_freq"].agg(
                stockcode_freq_mean="mean",
                stockcode_freq_max="max",
            )
            out["stockcode_freq_mean"] = (
                out["customer_id"].astype(str).map(agg["stockcode_freq_mean"]).fillna(0.0)
            )
            out["stockcode_freq_max"] = (
                out["customer_id"].astype(str).map(agg["stockcode_freq_max"]).fillna(0.0)
            )
        elif "stockcode_freq_mean" not in out.columns:
            out["stockcode_freq_mean"] = 0.0
            out["stockcode_freq_max"] = 0.0

        return out


class StandardScalerWrapper:
    """Pure NumPy zero-mean unit-variance scaler for scale-sensitive models."""

    def __init__(self, feature_cols: list[str] | None = None) -> None:
        """Initialize scaler with optional explicit column list."""
        self.feature_cols = feature_cols
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        """Compute column means and standard deviations on training split only."""
        cols = self.feature_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = list(cols)
        self.means_ = {}
        self.stds_ = {}
        for col in self.feature_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            self.means_[col] = mean_v
            self.stds_[col] = std_v if std_v > 1e-9 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardize columns using training-fitted means and standard deviations."""
        if not self.feature_cols:
            raise RuntimeError("StandardScalerWrapper must be fitted before transform.")
        cols_out: list[np.ndarray] = []
        for col in self.feature_cols:
            vals = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(self.means_[col])
                .to_numpy(dtype=np.float64)
            )
            z = (vals - self.means_[col]) / self.stds_[col]
            cols_out.append(z.reshape(-1, 1))
        return np.hstack(cols_out)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return standardized matrix."""
        return self.fit(df).transform(df)


def save_encoding_artifacts(
    encoders: FeatureEncoders,
    scaler: StandardScalerWrapper,
    artifacts_dir: str | Path = "models_artifacts",
) -> tuple[Path, Path]:
    """Persist fitted FeatureEncoders and StandardScalerWrapper to disk via joblib."""
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_path = out_dir / "encoders.joblib"
    scl_path = out_dir / "scaler.joblib"
    joblib.dump(encoders, enc_path)
    joblib.dump(scaler, scl_path)
    logger.info("Saved encoding artifacts to %s and %s", enc_path, scl_path)
    return enc_path, scl_path


def load_artifacts(artifacts_dir: str | Path = "models_artifacts") -> dict[str, Any]:
    """Load persisted encoders and scaler artifacts for inference in the API."""
    base_dir = Path(artifacts_dir)
    return {
        "encoders": joblib.load(base_dir / "encoders.joblib"),
        "scaler": joblib.load(base_dir / "scaler.joblib"),
    }
