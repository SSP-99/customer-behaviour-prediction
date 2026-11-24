"""Pure, unit-testable customer behavioral, seasonal, affinity, and engagement features."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.labels import derive_product_category
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_rfm_and_clv(history_df: pd.DataFrame, cutoff_date: str | pd.Timestamp) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary, and Historical/Projected CLV strictly pre-cutoff."""
    cutoff = pd.Timestamp(cutoff_date)
    tx = history_df[history_df["invoice_date"] < cutoff].copy()
    is_ret = tx["is_return"] if "is_return" in tx.columns else (tx["quantity"] < 0)
    purchases = tx[~is_ret]

    inv = purchases.groupby(["customer_id", "invoice"], as_index=False).agg(
        order_spend=("line_total", "sum"),
        order_items=("quantity", "sum"),
        order_date=("invoice_date", "first"),
    )
    rfm = inv.groupby("customer_id").agg(
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
        frequency=("invoice", "nunique"),
        monetary_total=("order_spend", "sum"),
        monetary_avg=("order_spend", "mean"),
        avg_basket_items=("order_items", "mean"),
    )
    rfm["recency_days"] = (cutoff - rfm["last_purchase"]).dt.total_seconds() / 86400.0
    rfm["tenure_days"] = (cutoff - rfm["first_purchase"]).dt.total_seconds() / 86400.0
    rfm["avg_days_per_order"] = rfm["tenure_days"] / rfm["frequency"].clip(lower=1)

    net_rev = tx.groupby("customer_id")["line_total"].sum().rename("historical_clv")
    rfm = rfm.join(net_rev)
    rfm["historical_clv"] = rfm["historical_clv"].fillna(0.0)
    active_months = (rfm["tenure_days"] / 30.0).clip(lower=1.0)
    rfm["projected_clv_90d"] = (rfm["historical_clv"] / active_months) * 3.0
    return rfm.drop(columns=["first_purchase", "last_purchase"])


def compute_frequency_trend(
    history_df: pd.DataFrame, cutoff_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Compute linear regression slope of monthly order frequency per customer."""
    cutoff = pd.Timestamp(cutoff_date)
    tx = history_df[history_df["invoice_date"] < cutoff].copy()
    is_ret = tx["is_return"] if "is_return" in tx.columns else (tx["quantity"] < 0)
    purchases = tx[~is_ret].copy()
    purchases["ym"] = purchases["invoice_date"].dt.to_period("M")

    monthly = (
        purchases.groupby(["customer_id", "ym"], as_index=False)["invoice"]
        .nunique()
        .sort_values(["customer_id", "ym"])
    )

    slopes: dict[str, float] = {}
    for cid, grp in monthly.groupby("customer_id"):
        y_vals = grp["invoice"].to_numpy(dtype=np.float64)
        if len(y_vals) < 2:
            slopes[str(cid)] = 0.0
        else:
            x_vals = np.arange(len(y_vals), dtype=np.float64)
            slope = float(np.polyfit(x_vals, y_vals, 1))
            slopes[str(cid)] = round(slope, 6)

    return pd.DataFrame({"frequency_trend_slope": pd.Series(slopes, dtype=np.float64)})


def compute_inter_purchase_cadence(
    history_df: pd.DataFrame, cutoff_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Compute mean and variance of days between consecutive orders (0 variance for single-order)."""
    cutoff = pd.Timestamp(cutoff_date)
    tx = history_df[history_df["invoice_date"] < cutoff].copy()
    is_ret = tx["is_return"] if "is_return" in tx.columns else (tx["quantity"] < 0)
    inv = (
        tx[~is_ret]
        .groupby(["customer_id", "invoice"], as_index=False)["invoice_date"]
        .first()
        .sort_values(["customer_id", "invoice_date"])
    )
    inv["gap_days"] = inv.groupby("customer_id")["invoice_date"].diff().dt.total_seconds() / 86400.0
    cadence = inv.groupby("customer_id")["gap_days"].agg(
        inter_purchase_mean="mean",
        inter_purchase_var="var",
    )
    tenure = (
        cutoff - inv.groupby("customer_id")["invoice_date"].min()
    ).dt.total_seconds() / 86400.0
    cadence["inter_purchase_mean"] = cadence["inter_purchase_mean"].fillna(tenure)
    cadence["inter_purchase_var"] = cadence["inter_purchase_var"].fillna(0.0)
    return cadence


def compute_seasonal_concentration(
    history_df: pd.DataFrame, cutoff_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Compute Nov-Dec holiday spend share and monthly spend Herfindahl concentration index."""
    cutoff = pd.Timestamp(cutoff_date)
    tx = history_df[history_df["invoice_date"] < cutoff].copy()
    is_ret = tx["is_return"] if "is_return" in tx.columns else (tx["quantity"] < 0)
    purchases = tx[~is_ret].copy()
    purchases["month"] = purchases["invoice_date"].dt.month
    purchases["is_holiday"] = purchases["month"].isin([11, 12])

    total_spend = purchases.groupby("customer_id")["line_total"].sum().clip(lower=1e-6)
    holiday_spend = purchases[purchases["is_holiday"]].groupby("customer_id")["line_total"].sum()
    holiday_share = (holiday_spend / total_spend).fillna(0.0).clip(0.0, 1.0)

    m_spend = purchases.groupby(["customer_id", "month"], as_index=False)["line_total"].sum()
    m_spend["tot"] = m_spend["customer_id"].map(total_spend)
    m_spend["share_sq"] = (m_spend["line_total"] / m_spend["tot"]) ** 2
    hhi = m_spend.groupby("customer_id")["share_sq"].sum().clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "holiday_spend_share": holiday_share,
            "monthly_spend_hhi": hhi,
        }
    ).fillna(0.0)


def compute_category_affinity(
    history_df: pd.DataFrame, cutoff_date: str | pd.Timestamp, top_n: int = 5
) -> pd.DataFrame:
    """Compute spend share vector across top-N product categories plus unique SKU diversity."""
    cutoff = pd.Timestamp(cutoff_date)
    tx = history_df[history_df["invoice_date"] < cutoff].copy()
    is_ret = tx["is_return"] if "is_return" in tx.columns else (tx["quantity"] < 0)
    purchases = tx[~is_ret].copy()

    desc_s = (
        purchases["description"]
        if "description" in purchases.columns
        else pd.Series("", index=purchases.index)
    )
    purchases["category"] = [
        derive_product_category(c, d) for c, d in zip(purchases["stock_code"], desc_s, strict=True)
    ]

    top_cats = purchases.groupby("category")["line_total"].sum().nlargest(top_n).index.tolist()
    tot_spend = purchases.groupby("customer_id")["line_total"].sum().clip(lower=1e-6)
    diversity = purchases.groupby("customer_id")["stock_code"].nunique().rename("unique_skus")

    out = pd.DataFrame({"unique_skus": diversity})
    for cat in top_cats:
        cat_spend = (
            purchases[purchases["category"] == cat].groupby("customer_id")["line_total"].sum()
        )
        out[f"cat_share_{cat.lower()}"] = (cat_spend / tot_spend).fillna(0.0).clip(0.0, 1.0)

    return out.fillna(0.0)


def compute_returns_and_markdown_sensitivity(
    history_df: pd.DataFrame, cutoff_date: str | pd.Timestamp
) -> pd.DataFrame:
    """Compute return rate, cancellation count, markdown sensitivity proxy, countries, and bulk share."""
    cutoff = pd.Timestamp(cutoff_date)
    tx = history_df[history_df["invoice_date"] < cutoff].copy()
    is_ret = tx["is_return"] if "is_return" in tx.columns else (tx["quantity"] < 0)
    purchases = tx[~is_ret].copy()
    returns = tx[is_ret].copy()

    # Markdown sensitivity proxy: share of purchase lines priced < 90% of item's median historical price
    median_prices = purchases.groupby("stock_code")["price"].median()
    purchases["item_med_price"] = purchases["stock_code"].map(median_prices)
    purchases["is_markdown"] = purchases["price"] < (0.90 * purchases["item_med_price"])

    gross = purchases.groupby("customer_id")["line_total"].sum().clip(lower=1e-6)
    ret_val = returns.groupby("customer_id")["line_total"].sum().abs()
    cancel_cnt = returns.groupby("customer_id")["invoice"].nunique()
    md_share = purchases.groupby("customer_id")["is_markdown"].mean()
    n_countries = purchases.groupby("customer_id")["country"].nunique()
    bulk_col = (
        purchases["is_bulk_order"]
        if "is_bulk_order" in purchases.columns
        else (purchases["quantity"].abs() >= 100)
    )
    purchases["bulk_flag"] = bulk_col
    bulk_share = purchases.groupby("customer_id")["bulk_flag"].mean()

    idx = purchases["customer_id"].unique()
    out = pd.DataFrame(index=idx)
    out.index.name = "customer_id"
    out["return_value_rate"] = (ret_val.reindex(idx).fillna(0.0) / gross.reindex(idx)).clip(
        0.0, 1.0
    )
    out["cancellation_count"] = cancel_cnt.reindex(idx).fillna(0.0)
    out["markdown_sensitivity"] = md_share.reindex(idx).fillna(0.0)
    out["country_count"] = n_countries.reindex(idx).fillna(1.0)
    out["bulk_buyer_share"] = bulk_share.reindex(idx).fillna(0.0)
    return out


def compute_engagement_score(rfm_df: pd.DataFrame) -> pd.Series:
    """Compute composite 0-100 RFM engagement score using weighted 1-5 quintiles."""
    r_score = pd.qcut(
        rfm_df["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
    ).astype(float)
    f_score = pd.qcut(rfm_df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(
        float
    )
    m_score = pd.qcut(
        rfm_df["monetary_total"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(float)
    raw = 0.35 * r_score + 0.35 * f_score + 0.30 * m_score
    return (((raw - 1.0) / 4.0) * 100.0).round(2).rename("engagement_score")


def write_feature_dictionary(output_path: str | Path = "docs/feature_dictionary.md") -> Path:
    """Generate docs/feature_dictionary.md documenting every engineered feature."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Customer Feature Dictionary (Point-in-Time Safe)",
        "",
        "All features are computed strictly from historical transactions (`invoice_date < cutoff_date`).",
        "",
        "| Feature Name | Formula / Definition | Business & Predictive Signal |",
        "|---|---|---|",
        "| `recency_days` | `(cutoff_date - last_purchase_date) / 86400s` | Primary churn indicator; longer inactivity strongly increases churn probability. |",
        "| `frequency` | Count of distinct non-return invoices pre-cutoff | Habitual repeat buyers exhibit lower churn and higher future CLV. |",
        "| `monetary_total` | Sum of positive purchase line totals | Total gross historical revenue generated by the customer. |",
        "| `monetary_avg` | `monetary_total / frequency` | Average order value (AOV) distinguishing bulk vs. small-basket buyers. |",
        "| `historical_clv` | Sum of signed `line_total` (gross sales minus returns) | True net historical profitability per customer. |",
        "| `projected_clv_90d` | `(historical_clv / max(tenure_months, 1)) * 3` | Baseline run-rate 90-day spend expectation. |",
        "| `frequency_trend_slope` | OLS slope of monthly invoice counts over active months | Captures acceleration vs. decay in ordering cadence prior to cutoff. |",
        "| `inter_purchase_mean` | Mean days between consecutive purchase invoices | Personal replenishment cycle length. |",
        "| `inter_purchase_var` | Variance of inter-purchase gaps (0 for 1-order customers) | Order regularity vs. erratic purchasing behavior. |",
        "| `holiday_spend_share` | Share of historical spend occurring in Nov-Dec | Identifies seasonal gift buyers who churn outside Q4. |",
        "| `monthly_spend_hhi` | Sum of squared monthly spend shares (Herfindahl index) | Measures calendar concentration (1.0 = single-month buyer). |",
        "| `cat_share_<category>` | Spend share in top-5 derived product categories | Captures category affinity for cross-sell and retention stability. |",
        "| `unique_skus` | Count of distinct StockCodes purchased | Catalog breadth reduces single-item churn risk. |",
        "| `return_value_rate` | `abs(return_spend) / max(gross_spend, 1e-6)` | High return friction precedes customer defection. |",
        "| `cancellation_count` | Distinct cancellation invoices (`C*` or `quantity < 0`) | Direct operational dissatisfaction signal. |",
        "| `markdown_sensitivity` | Share of lines bought at `< 90%` of SKU median unit price | Proxy for promotion/discount-driven purchasing behavior. |",
        "| `bulk_buyer_share` | Share of lines with `abs(quantity) >= 100` | Distinguishes B2B wholesale accounts from retail consumers. |",
        "| `engagement_score` | Weighted 1-5 RFM quintile score scaled to `0-100` | Executive summary index combining recency, frequency, and spend. |",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote feature dictionary to %s", out)
    return out
