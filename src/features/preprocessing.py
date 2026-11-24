"""Leakage-free stratified splitting and preprocessing pipeline persistence."""

import json
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_AND_TARGET_COLS = {"customer_id", "churn_label", "clv_90d_target"}
CATEGORICAL_COLS = ["country_grouped"]


class TabularPreprocessor:
    """Leakage-free median-imputer + RobustScaler + OneHotEncoder persisted via joblib."""

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]) -> None:
        """Initialize preprocessor with explicit numeric and categorical feature names."""
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.medians_: dict[str, float] = {}
        self.iqrs_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        self.feature_names_out_: list[str] = []

    def fit(self, df: pd.DataFrame) -> Self:
        """Compute medians, IQRs, and category levels strictly on training data."""
        self.medians_ = {}
        self.iqrs_ = {}
        self.categories_ = {}
        out_names: list[str] = []

        for col in self.numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            med = float(series.median()) if series.notna().any() else 0.0
            q25 = float(series.quantile(0.25)) if series.notna().any() else 0.0
            q75 = float(series.quantile(0.75)) if series.notna().any() else 1.0
            iqr = q75 - q25
            self.medians_[col] = med
            self.iqrs_[col] = iqr if iqr > 1e-9 else 1.0
            out_names.append(col)

        for col in self.categorical_cols:
            vals = df[col].fillna("Other").astype(str)
            cats = sorted(vals.unique().tolist())
            self.categories_[col] = cats
            for cat in cats:
                out_names.append(f"{col}_{cat}")

        self.feature_names_out_ = out_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a feature DataFrame into a scaled float64 matrix."""
        blocks: list[np.ndarray] = []

        for col in self.numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").fillna(self.medians_[col])
            scaled = (series.to_numpy(dtype=np.float64) - self.medians_[col]) / self.iqrs_[col]
            blocks.append(scaled.reshape(-1, 1))

        for col in self.categorical_cols:
            vals = df[col].fillna("Other").astype(str).to_numpy()
            for cat in self.categories_[col]:
                encoded = (vals == cat).astype(np.float64).reshape(-1, 1)
                blocks.append(encoded)

        if not blocks:
            return np.empty((len(df), 0), dtype=np.float64)
        return np.hstack(blocks)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return transformed feature matrix."""
        return self.fit(df).transform(df)

    def get_feature_names_out(self) -> np.ndarray:
        """Return output transformed feature names."""
        return np.asarray(self.feature_names_out_, dtype=object)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns from the customer dataset."""
    excluded = ID_AND_TARGET_COLS | {"split"}
    feature_cols = [c for c in df.columns if c not in excluded]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def stratified_train_val_test_split(
    df: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split customer dataset into deterministic stratified 70/15/15 train/val/test sets."""
    split_cfg = config.splitting
    rng = np.random.default_rng(config.project.seed)
    strat_col = split_cfg.stratify_col

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(strat_col, sort=True):
        perm = rng.permutation(len(group))
        shuffled = group.iloc[perm]
        n_total = len(shuffled)
        n_train = int(round(n_total * split_cfg.train_ratio))
        n_val = int(round(n_total * split_cfg.val_ratio))

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val :])

    train_df = (
        pd.concat(train_parts)
        .sample(frac=1.0, random_state=config.project.seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts)
        .sample(frac=1.0, random_state=config.project.seed)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_parts)
        .sample(frac=1.0, random_state=config.project.seed)
        .reset_index(drop=True)
    )

    logger.info(
        "Stratified split complete | train=%d (%.1f%%) | val=%d (%.1f%%) | test=%d (%.1f%%)",
        len(train_df),
        100.0 * len(train_df) / len(df),
        len(val_df),
        100.0 * len(val_df) / len(df),
        len(test_df),
        100.0 * len(test_df) / len(df),
    )
    return train_df, val_df, test_df


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> TabularPreprocessor:
    """Create an unfitted TabularPreprocessor instance."""
    return TabularPreprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def fit_and_save_preprocessor(
    train_df: pd.DataFrame,
    config: AppConfig,
) -> tuple[TabularPreprocessor, list[str]]:
    """Fit TabularPreprocessor on training data only and persist artifact + schema JSON."""
    num_cols, cat_cols = get_feature_columns(train_df)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(train_df[num_cols + cat_cols])

    transformed_names = preprocessor.get_feature_names_out().tolist()

    if config.artifacts is not None:
        prep_path = Path(config.artifacts.preprocessor_path)
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, prep_path)

        schema_path = Path(config.artifacts.feature_columns_path)
        schema_payload = {
            "raw_numeric_cols": num_cols,
            "raw_categorical_cols": cat_cols,
            "transformed_feature_names": transformed_names,
        }
        with schema_path.open("w", encoding="utf-8") as f:
            json.dump(schema_payload, f, indent=2)

        logger.info(
            "Saved fitted preprocessor to %s (%d transformed features)",
            prep_path,
            len(transformed_names),
        )

    return preprocessor, transformed_names


def transform_features(
    preprocessor: TabularPreprocessor,
    df: pd.DataFrame,
    config: AppConfig,
) -> np.ndarray:
    """Transform a feature DataFrame using a pre-fitted TabularPreprocessor."""
    num_cols, cat_cols = get_feature_columns(df)
    return preprocessor.transform(df[num_cols + cat_cols])
