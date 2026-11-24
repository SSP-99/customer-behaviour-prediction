"""Unit tests for build_snapshot, build_labels, and FeatureEncoders/StandardScalerWrapper."""

import pandas as pd
import pytest

from src.data.snapshot import build_snapshot
from src.features.encoding import (
    FeatureEncoders,
    StandardScalerWrapper,
    load_artifacts,
    save_encoding_artifacts,
)
from src.features.labels import build_labels, derive_product_category


@pytest.fixture()
def sample_retail_tx() -> pd.DataFrame:
    """Create sample transactions spanning pre-cutoff, future 90d window, and post-window."""
    return pd.DataFrame(
        {
            "invoice": ["5001", "5002", "5003", "C5004", "5005", "5006"],
            "stock_code": ["85123A", "22423", "20725", "22423", "85123A", "21212"],
            "description": [
                "WHITE HANGING HEART T-LIGHT HOLDER",
                "REGENCY CAKESTAND 3 TIER",
                "LUNCH BAG RED RETROSPOT",
                "REGENCY CAKESTAND 3 TIER",
                "WHITE HANGING HEART T-LIGHT HOLDER",
                "PACK OF 72 RETROSPOT CAKE CASES",
            ],
            "quantity": list((4, 2, 5, -1, 10, 6)),
            "invoice_date": pd.to_datetime(
                [
                    "2011-08-01 10:00:00",
                    "2011-08-15 11:00:00",
                    "2011-09-20 12:00:00",
                    "2011-10-01 14:00:00",
                    "2011-10-10 15:00:00",
                    "2011-12-25 09:00:00",
                ]
            ),
            "price": list((5.0, 30.0, 10.0, 30.0, 4.0, 2.0)),
            "line_total": list((20.0, 60.0, 50.0, -30.0, 40.0, 12.0)),
            "customer_id": ["C1", "C2", "C1", "C1", "C1", "C2"],
            "country": [
                "United Kingdom",
                "Brazil",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "Brazil",
            ],
            "is_return": [False, False, False, True, False, False],
        }
    )


def test_build_snapshot_strict_temporal_isolation(sample_retail_tx: pd.DataFrame) -> None:
    """History must be strictly < cutoff, future in [cutoff, cutoff+90d), with zero leakage."""
    cutoff = pd.Timestamp("2011-09-10")
    history_df, future_df = build_snapshot(sample_retail_tx, cutoff_date=cutoff, window_days=90)

    assert len(history_df) == 2
    assert len(future_df) == 3
    assert history_df["invoice_date"].max() < cutoff
    assert future_df["invoice_date"].min() >= cutoff
    assert future_df["invoice_date"].max() < cutoff + pd.Timedelta(days=90)
    assert set(history_df["invoice"]).isdisjoint(set(future_df["invoice"]))


def test_build_labels_churn_spend_and_next_category(sample_retail_tx: pd.DataFrame) -> None:
    """Verify churn, net future_spend clipping, next_purchase_flag, and next_purchase_category."""
    history_df, future_df = build_snapshot(
        sample_retail_tx, cutoff_date="2011-09-10", window_days=90
    )
    labels = build_labels(history_df, future_df)

    assert len(labels) == 2

    c1 = labels[labels["customer_id"] == "C1"].squeeze()
    assert int(c1["churn"]) == 0
    assert int(c1["next_purchase_flag"]) == 1
    assert float(c1["future_spend"]) == pytest.approx(60.0)
    assert c1["next_purchase_category"] == "HOME_DECOR"

    c2 = labels[labels["customer_id"] == "C2"].squeeze()
    assert int(c2["churn"]) == 1
    assert int(c2["next_purchase_flag"]) == 0
    assert float(c2["future_spend"]) == pytest.approx(0.0)
    assert c2["next_purchase_category"] == "NONE"

    assert derive_product_category("99999", "UNKNOWN ITEM") == "SKU_99"


def test_encoding_and_scaler_fit_transform_persistence(
    sample_retail_tx: pd.DataFrame, tmp_path: object
) -> None:
    """Verify Country rare grouping, StockCode frequency encoding, and joblib load_artifacts."""
    history_df, _ = build_snapshot(sample_retail_tx, cutoff_date="2011-09-10", window_days=90)
    cust_df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "country": ["United Kingdom", "Brazil"],
            "monetary": list((20.0, 60.0)),
        }
    )

    enc = FeatureEncoders(top_countries=["United Kingdom"]).fit(cust_df, history_df)
    transformed = enc.transform(cust_df, history_df)
    assert "country_ohe_United_Kingdom" in transformed.columns
    assert "country_ohe_Other" in transformed.columns
    assert (
        float(transformed.loc[transformed["customer_id"] == "C2", "country_ohe_Other"].squeeze())
        == 1.0
    )
    assert float(
        transformed.loc[transformed["customer_id"] == "C1", "stockcode_freq_mean"].squeeze()
    ) == pytest.approx(0.5)

    scaler = StandardScalerWrapper(feature_cols=["monetary", "stockcode_freq_mean"]).fit(
        transformed
    )
    scaled_mat = scaler.transform(transformed)
    assert scaled_mat.shape == (2, 2)

    save_encoding_artifacts(enc, scaler, artifacts_dir=str(tmp_path))
    loaded = load_artifacts(artifacts_dir=str(tmp_path))
    assert "encoders" in loaded and "scaler" in loaded
