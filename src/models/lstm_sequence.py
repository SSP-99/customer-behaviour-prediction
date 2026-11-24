"""Sequence builder and TensorFlow/Keras LSTM model for customer churn prediction.

Architectural Trade-off Note (PRD Risk Register):
While an LSTM captures sequential order cadence and basket progression across up to
20 recent invoices, with ~5,280 eligible customers (~3,696 training rows) and an average
of ~4-6 orders per customer, gradient-boosted trees (XGBoost/LightGBM) on engineered
point-in-time tabular features achieve stronger ROC-AUC and lower variance, so trees
remain the primary production model while LSTM serves as a sequential deep-learning benchmark.
"""

from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from src.features.labels import derive_product_category
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORY_TO_ID = {
    "HOME_DECOR": 1.0,
    "KITCHEN_DINING": 2.0,
    "BAGS_ACCESSORIES": 3.0,
    "STATIONERY_PAPER": 4.0,
    "SEASONAL_GIFTS": 5.0,
    "GARDEN_OUTDOOR": 6.0,
}


def build_customer_sequences(
    transactions_df: pd.DataFrame,
    customer_ids: list[str],
    cutoff_date: str | pd.Timestamp,
    max_len: int = 20,
) -> np.ndarray:
    """Build strictly pre-cutoff pre-padded (N, max_len, 5) order sequences per customer.

    Channels per order event:
    0: log1p(abs(order_value))
    1: log1p(gap_days_since_previous_order)
    2: log1p(abs(number_of_items))
    3: return_flag (1.0 if net order_value < 0 or any return line, else 0.0)
    4: dominant derived category numeric ID (1.0..7.0)
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    tx = transactions_df.copy()
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])

    # Strict temporal guard: never touch transactions on or after cutoff_date
    hist = tx[tx["invoice_date"] < cutoff_ts].copy()
    if not hist.empty:
        assert hist["invoice_date"].max() < cutoff_ts, (
            "Temporal leakage in build_customer_sequences!"
        )

    hist = hist[hist["customer_id"].notna()].copy()
    hist["customer_id"] = hist["customer_id"].astype(str).str.strip()
    target_set = {str(cid).strip() for cid in customer_ids}
    hist = hist[hist["customer_id"].isin(target_set)].copy()

    n_customers = len(customer_ids)
    seq_tensor = np.zeros((n_customers, max_len, 5), dtype=np.float32)
    if hist.empty:
        return seq_tensor

    if "line_total" not in hist.columns:
        hist["line_total"] = hist["quantity"] * hist["price"]
    is_ret = hist["is_return"] if "is_return" in hist.columns else (hist["quantity"] < 0)
    hist["ret_num"] = is_ret.astype(int)

    desc_s = (
        hist["description"] if "description" in hist.columns else pd.Series("", index=hist.index)
    )
    hist["cat_id"] = [
        CATEGORY_TO_ID.get(derive_product_category(c, d), 7.0)
        for c, d in zip(hist["stock_code"], desc_s, strict=True)
    ]

    orders = (
        hist.groupby(["customer_id", "invoice"], as_index=False)
        .agg(
            invoice_date=("invoice_date", "first"),
            order_value=("line_total", "sum"),
            num_items=("quantity", lambda s: float(np.sum(np.abs(s)))),
            has_return=("ret_num", "max"),
            dom_cat=("cat_id", "first"),
        )
        .sort_values(["customer_id", "invoice_date"])
    )
    orders["gap_days"] = (
        orders.groupby("customer_id")["invoice_date"].diff().dt.total_seconds().fillna(0.0)
        / 86400.0
    ).clip(lower=0.0)

    grouped = {str(cid): grp for cid, grp in orders.groupby("customer_id")}
    for idx, cid in enumerate(customer_ids):
        grp = grouped.get(str(cid).strip())
        if grp is None or grp.empty:
            continue
        tail = grp.tail(max_len)
        steps = np.column_stack(
            [
                np.log1p(np.abs(tail["order_value"].to_numpy(dtype=np.float32))),
                np.log1p(tail["gap_days"].to_numpy(dtype=np.float32)),
                np.log1p(tail["num_items"].to_numpy(dtype=np.float32)),
                tail["has_return"].to_numpy(dtype=np.float32),
                tail["dom_cat"].to_numpy(dtype=np.float32),
            ]
        )
        seq_tensor[idx, max_len - len(steps) :, :] = steps

    return seq_tensor


class KerasLSTMChurnModel:
    """Masking + LSTM(64) + Dropout binary churn classifier with EarlyStopping."""

    def __init__(
        self,
        lstm_units: int = 64,
        dropout_rate: float = 0.30,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 25,
        seed: int = 42,
    ) -> None:
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.model_: tf.keras.Model | None = None
        self.ch_mean_: np.ndarray | None = None
        self.ch_std_: np.ndarray | None = None

    def _scale_sequences(self, x_seq: np.ndarray, fit: bool = False) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float32).copy()
        non_pad = np.any(x != 0.0, axis=-1)
        if fit:
            valid_steps = x[non_pad] if np.any(non_pad) else x.reshape(-1, x.shape[-1])
            self.ch_mean_ = np.mean(valid_steps, axis=0)
            std = np.std(valid_steps, axis=0)
            self.ch_std_ = np.where(std > 1e-6, std, 1.0)
        assert self.ch_mean_ is not None and self.ch_std_ is not None
        x[non_pad] = (x[non_pad] - self.ch_mean_) / self.ch_std_
        return x

    def fit(self, x_seq: np.ndarray, y: np.ndarray) -> "KerasLSTMChurnModel":
        """Fit LSTM sequence classifier on training sequences only."""
        tf.keras.utils.set_random_seed(self.seed)
        x_scaled = self._scale_sequences(x_seq, fit=True)
        y_arr = np.asarray(y, dtype=np.float32)

        n_pos = max(float(np.sum(y_arr == 1)), 1.0)
        n_neg = max(float(np.sum(y_arr == 0)), 1.0)
        tot = n_pos + n_neg
        cw = {0: tot / (2.0 * n_neg), 1: (tot / (2.0 * n_pos)) * 1.1}

        seq_len, n_ch = x_scaled.shape[1], x_scaled.shape[2]
        inp = tf.keras.Input(shape=(seq_len, n_ch))
        x = tf.keras.layers.Masking(mask_value=0.0)(inp)
        x = tf.keras.layers.LSTM(self.lstm_units)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            x_scaled,
            y_arr,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=cw,
            callbacks=[es],
            verbose=0,
        )
        self.model_ = model
        return self

    def predict_proba(self, x_seq: np.ndarray) -> np.ndarray:
        """Predict 2-column probabilities [P(retained), P(churn)]."""
        if self.model_ is None:
            raise RuntimeError("KerasLSTMChurnModel must be fitted first.")
        x_scaled = self._scale_sequences(x_seq, fit=False)
        p1 = np.clip(self.model_.predict(x_scaled, verbose=0).reshape(-1), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])

    def get_params(self) -> dict[str, Any]:
        return {
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }
