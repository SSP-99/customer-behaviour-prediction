"""Strict point-in-time feature engineering, target labeling, and product associations."""

from collections import Counter
from itertools import combinations

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_COUNTRIES = {"United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"}


def build_point_in_time_features_and_targets(
    customer_tx: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """Construct leakage-free customer features prior to cutoff and 90-day future targets.

    Strict Point-in-Time Design:
    - Observation window: `invoice_date < cutoff_date` (2011-09-10).
    - Prediction window: `cutoff_date <= invoice_date < cutoff_date + churn_window_days` (90 days).
    - Eligible cohort: Customers with >= `min_purchases_before_cutoff` non-return invoices
      strictly before `cutoff_date`.
    - `churn_label`: 1 if customer made 0 non-return purchases in the future window, else 0.
    - `clv_90d_target`: Net spend (`sum(line_total)`, floored at 0.0) in the future window.

    Args:
        customer_tx: Cleaned customer-level transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        Customer-level modeling DataFrame with `customer_id`, features, and targets.
    """
    pit = config.point_in_time
    cutoff = pd.Timestamp(pit.cutoff_date)
    window_end = cutoff + pd.Timedelta(days=pit.churn_window_days)

    tx = customer_tx.copy()
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])

    hist = tx[tx["invoice_date"] < cutoff].copy()
    future = tx[(tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < window_end)].copy()

    logger.info(
        "Point-in-time split | cutoff=%s | window_end=%s | hist_rows=%d | future_rows=%d",
        cutoff.date(),
        window_end.date(),
        len(hist),
        len(future),
    )

    hist_purchases = hist[~hist["is_return"]].copy()
    hist_returns = hist[hist["is_return"]].copy()

    purchase_counts = hist_purchases.groupby("customer_id")["invoice"].nunique()
    eligible_ids = purchase_counts[purchase_counts >= pit.min_purchases_before_cutoff].index
    hist = hist[hist["customer_id"].isin(eligible_ids)].copy()
    hist_purchases = hist_purchases[hist_purchases["customer_id"].isin(eligible_ids)].copy()
    hist_returns = hist_returns[hist_returns["customer_id"].isin(eligible_ids)].copy()

    inv_totals = (
        hist_purchases.groupby(["customer_id", "invoice"], as_index=False)
        .agg(
            invoice_spend=("line_total", "sum"),
            invoice_items=("quantity", "sum"),
            invoice_date=("invoice_date", "first"),
        )
        .sort_values(["customer_id", "invoice_date"])
    )

    rfm = inv_totals.groupby("customer_id").agg(
        first_purchase_date=("invoice_date", "min"),
        last_purchase_date=("invoice_date", "max"),
        frequency=("invoice", "nunique"),
        monetary_gross=("invoice_spend", "sum"),
        avg_order_value=("invoice_spend", "mean"),
        std_order_value=("invoice_spend", "std"),
        max_order_value=("invoice_spend", "max"),
        avg_basket_size=("invoice_items", "mean"),
        total_items_purchased=("invoice_items", "sum"),
    )
    rfm["std_order_value"] = rfm["std_order_value"].fillna(0.0)
    rfm["recency_days"] = (cutoff - rfm["last_purchase_date"]).dt.total_seconds() / 86400.0
    rfm["tenure_days"] = (cutoff - rfm["first_purchase_date"]).dt.total_seconds() / 86400.0
    rfm["active_span_days"] = (
        rfm["last_purchase_date"] - rfm["first_purchase_date"]
    ).dt.total_seconds() / 86400.0

    inv_totals["prev_invoice_date"] = inv_totals.groupby("customer_id")["invoice_date"].shift(1)
    inv_totals["gap_days"] = (
        inv_totals["invoice_date"] - inv_totals["prev_invoice_date"]
    ).dt.total_seconds() / 86400.0
    cadence = inv_totals.groupby("customer_id")["gap_days"].agg(
        inter_purchase_days_mean="mean",
        inter_purchase_days_std="std",
    )
    cadence["inter_purchase_days_mean"] = cadence["inter_purchase_days_mean"].fillna(
        rfm["tenure_days"]
    )
    cadence["inter_purchase_days_std"] = cadence["inter_purchase_days_std"].fillna(0.0)

    ret_agg = hist_returns.groupby("customer_id").agg(
        return_invoices=("invoice", "nunique"),
        monetary_returns=("returns", "sum"),
    )
    prod_diversity = hist_purchases.groupby("customer_id").agg(
        unique_products=("stock_code", "nunique"),
        bulk_order_lines=("is_bulk_order", "sum"),
        active_months=("invoice_date", lambda s: s.dt.to_period("M").nunique()),
    )

    for days in (30, 60, 90):
        win_start = cutoff - pd.Timedelta(days=days)
        sub = hist_purchases[hist_purchases["invoice_date"] >= win_start]
        spend_s = sub.groupby("customer_id")["line_total"].sum().rename(f"spend_last_{days}d")
        orders_s = sub.groupby("customer_id")["invoice"].nunique().rename(f"orders_last_{days}d")
        rfm = rfm.join(spend_s).join(orders_s)
        rfm[f"spend_last_{days}d"] = rfm[f"spend_last_{days}d"].fillna(0.0)
        rfm[f"orders_last_{days}d"] = rfm[f"orders_last_{days}d"].fillna(0.0)

    hist_purchases["is_weekend"] = hist_purchases["invoice_date"].dt.dayofweek >= 5
    hist_purchases["hour"] = hist_purchases["invoice_date"].dt.hour
    habits = hist_purchases.groupby("customer_id").agg(
        weekend_order_ratio=("is_weekend", "mean"),
        preferred_hour=("hour", "mean"),
        country=("country", "first"),
    )
    habits["country_grouped"] = habits["country"].where(
        habits["country"].isin(TOP_COUNTRIES), "Other"
    )
    habits["is_uk"] = (habits["country"] == "United Kingdom").astype(int)

    features = (
        rfm.join(cadence).join(ret_agg).join(prod_diversity).join(habits.drop(columns=["country"]))
    )
    features["return_invoices"] = features["return_invoices"].fillna(0.0)
    features["monetary_returns"] = features["monetary_returns"].fillna(0.0)
    features["monetary_net"] = (features["monetary_gross"] - features["monetary_returns"]).round(4)
    features["historical_clv"] = features["monetary_net"]

    features["return_rate_orders"] = (
        features["return_invoices"] / (features["frequency"] + features["return_invoices"])
    ).fillna(0.0)
    features["return_rate_value"] = (
        features["monetary_returns"] / features["monetary_gross"].clip(lower=1.0)
    ).clip(0.0, 1.0)

    features["purchase_frequency_per_month"] = features["frequency"] / (
        (features["tenure_days"] / 30.0).clip(lower=1.0)
    )
    features["recency_to_cadence_ratio"] = features["recency_days"] / (
        features["inter_purchase_days_mean"].clip(lower=1.0)
    )
    avg_30d_baseline = features["monetary_gross"] / (
        (features["tenure_days"] / 30.0).clip(lower=1.0)
    )
    features["spend_trend_ratio"] = features["spend_last_30d"] / avg_30d_baseline.clip(lower=1.0)

    features = features.drop(columns=["first_purchase_date", "last_purchase_date"])

    future_eligible = future[future["customer_id"].isin(eligible_ids)].copy()
    future_purchases = future_eligible[~future_eligible["is_return"]]

    future_order_counts = future_purchases.groupby("customer_id")["invoice"].nunique()
    future_net_spend = future_eligible.groupby("customer_id")["line_total"].sum()

    features["future_orders_90d"] = (
        future_order_counts.reindex(features.index).fillna(0).astype(int)
    )
    features["churn_label"] = (features["future_orders_90d"] == 0).astype(int)
    features["clv_90d_target"] = (
        future_net_spend.reindex(features.index).fillna(0.0).clip(lower=0.0).round(4)
    )

    features = features.drop(columns=["future_orders_90d"]).reset_index()

    logger.info(
        "Built customer dataset | customers=%d | features=%d | churn_rate=%.2f%% | mean_clv_90d=%.2f",
        len(features),
        len(features.columns) - 3,
        100.0 * features["churn_label"].mean(),
        features["clv_90d_target"].mean(),
    )
    return features


def build_product_associations(
    product_tx: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """Compute item-to-item basket co-occurrence support, confidence, and lift pre-cutoff.

    Uses all valid non-return product baskets (including guest checkouts) strictly
    before `cutoff_date` so recommendations have zero future leakage.

    Args:
        product_tx: Cleaned product-level transactions DataFrame.
        config: Application configuration dataclass.

    Returns:
        DataFrame of ranked product association pairs (item_a -> item_b).
    """
    cutoff = pd.Timestamp(config.point_in_time.cutoff_date)
    min_support = int(config.models["recommender"]["min_support_count"])

    hist_prod = product_tx[
        (pd.to_datetime(product_tx["invoice_date"]) < cutoff) & (~product_tx["is_return"])
    ].copy()

    item_counts = hist_prod.groupby("stock_code")["invoice"].nunique()
    frequent_items = set(item_counts[item_counts >= min_support].nlargest(500).index)
    hist_prod = hist_prod[hist_prod["stock_code"].isin(frequent_items)]

    desc_lookup = (
        hist_prod.drop_duplicates("stock_code").set_index("stock_code")["description"].to_dict()
    )

    baskets = hist_prod.groupby("invoice")["stock_code"].apply(lambda s: sorted(set(s)))
    total_baskets = max(len(baskets), 1)

    single_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()

    for items in baskets:
        for item in items:
            single_counts[item] += 1
        if 2 <= len(items) <= 40:
            for a, b in combinations(items, 2):
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1

    rows: list[dict[str, object]] = []
    for (item_a, item_b), co_count in pair_counts.items():
        if co_count < min_support:
            continue
        supp_ab = co_count / total_baskets
        supp_a = single_counts[item_a] / total_baskets
        supp_b = single_counts[item_b] / total_baskets
        confidence = supp_ab / max(supp_a, 1e-9)
        lift = confidence / max(supp_b, 1e-9)
        rows.append(
            {
                "item_a": item_a,
                "item_a_desc": desc_lookup.get(item_a, item_a),
                "item_b": item_b,
                "item_b_desc": desc_lookup.get(item_b, item_b),
                "co_occurrence_count": int(co_count),
                "support": round(float(supp_ab), 6),
                "confidence": round(float(confidence), 6),
                "lift": round(float(lift), 6),
            }
        )

    assoc_df = pd.DataFrame(rows)
    if not assoc_df.empty:
        assoc_df = assoc_df.sort_values(
            ["item_a", "lift", "confidence"], ascending=[True, False, False]
        ).reset_index(drop=True)

    logger.info("Built product association rules | pairs=%d", len(assoc_df))
    return assoc_df
