"""Leakage-safe point-in-time snapshot builder separating history from future windows."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_snapshot(
    transactions_df: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    window_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into strictly disjoint historical and future observation windows."""
    if "invoice_date" not in transactions_df.columns:
        raise ValueError("transactions_df must contain an 'invoice_date' column.")

    cutoff_ts = pd.Timestamp(cutoff_date)
    window_end_ts = cutoff_ts + pd.Timedelta(days=int(window_days))

    tx = transactions_df.copy()
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])

    history_mask = tx["invoice_date"] < cutoff_ts
    future_mask = (tx["invoice_date"] >= cutoff_ts) & (tx["invoice_date"] < window_end_ts)

    assert not bool((history_mask & future_mask).any()), (
        "Overlap detected between history and future masks!"
    )

    history_df = tx.loc[history_mask].copy().reset_index(drop=True)
    future_df = tx.loc[future_mask].copy().reset_index(drop=True)

    if not history_df.empty:
        max_hist_ts = history_df["invoice_date"].max()
        assert max_hist_ts < cutoff_ts, (
            f"Leakage detected: max(history.invoice_date)={max_hist_ts} >= cutoff={cutoff_ts}"
        )

    if not future_df.empty:
        min_fut_ts = future_df["invoice_date"].min()
        max_fut_ts = future_df["invoice_date"].max()
        assert min_fut_ts >= cutoff_ts, (
            f"Boundary violation: min(future.invoice_date)={min_fut_ts} < cutoff={cutoff_ts}"
        )
        assert max_fut_ts < window_end_ts, (
            f"Window violation: max(future.invoice_date)={max_fut_ts} >= window_end={window_end_ts}"
        )

    if not history_df.empty and not future_df.empty:
        assert history_df["invoice_date"].max() < future_df["invoice_date"].min(), (
            "Temporal overlap between history_df and future_df!"
        )

    logger.info(
        "Built point-in-time snapshot | cutoff=%s | window_end=%s | history_rows=%d | future_rows=%d",
        cutoff_ts.date(),
        window_end_ts.date(),
        len(history_df),
        len(future_df),
    )
    return history_df, future_df
