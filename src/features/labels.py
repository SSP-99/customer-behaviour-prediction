"""Leakage-free customer target label generation (churn, future_spend, next_purchase_category)."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("HOME_DECOR", ("HEART", "HOLDER", "CANDLE", "LANTERN", "FRAME", "CLOCK", "SIGN", "WREATH")),
    ("KITCHEN_DINING", ("MUG", "CAKE", "TEA", "PLATE", "BOWL", "CUP", "LUNCH", "BOTTLE", "JAR")),
    ("BAGS_ACCESSORIES", ("BAG", "PURSE", "TOTE", "POUCH", "UMBRELLA", "CHARM", "NECKLACE")),
    ("STATIONERY_PAPER", ("CARD", "WRAP", "PAPER", "NOTEBOOK", "PEN", "STICKER", "RIBBON", "TAG")),
    ("SEASONAL_GIFTS", ("CHRISTMAS", "XMAS", "EASTER", "PARTY", "BUNTING", "GIFT", "VINTAGE")),
    ("GARDEN_OUTDOOR", ("GARDEN", "BIRD", "FLOWER", "PLANT", "POT", "WATERING")),
]


def derive_product_category(stock_code: str, description: str | None = None) -> str:
    """Derive a clean retail product category from Description keywords with StockCode prefix fallback."""
    desc_upper = str(description or "").strip().upper()
    for category_name, keywords in CATEGORY_KEYWORDS:
        if any(kw in desc_upper for kw in keywords):
            return category_name

    code_clean = "".join(ch for ch in str(stock_code).strip().upper() if ch.isalnum())
    prefix = code_clean[:2] if len(code_clean) >= 2 else (code_clean or "MISC")
    return f"SKU_{prefix}"


def build_labels(history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level churn, future_spend, next_purchase_flag, and next_purchase_category."""
    hist = history_df.copy()
    fut = future_df.copy()

    hist = hist[hist["customer_id"].notna()].copy()
    hist["customer_id"] = hist["customer_id"].astype(str).str.strip()
    hist = hist[hist["customer_id"] != ""].copy()

    is_ret_hist = hist["is_return"] if "is_return" in hist.columns else (hist["quantity"] < 0)
    hist_purchases = hist[~is_ret_hist]

    purchase_invoices = hist_purchases.groupby("customer_id")["invoice"].nunique()
    eligible_customers = sorted(purchase_invoices[purchase_invoices >= 1].index.tolist())

    base = pd.DataFrame({"customer_id": eligible_customers})
    if base.empty:
        base["churn"] = pd.Series(dtype="int64")
        base["future_spend"] = pd.Series(dtype="float64")
        base["next_purchase_flag"] = pd.Series(dtype="int64")
        base["next_purchase_category"] = pd.Series(dtype="object")
        return base

    if "line_total" not in fut.columns:
        fut["line_total"] = fut["quantity"] * fut["price"]

    fut = fut[fut["customer_id"].notna()].copy()
    fut["customer_id"] = fut["customer_id"].astype(str).str.strip()
    fut_eligible = fut[fut["customer_id"].isin(set(eligible_customers))].copy()

    is_ret_fut = (
        fut_eligible["is_return"]
        if "is_return" in fut_eligible.columns
        else (fut_eligible["quantity"] < 0)
    )
    fut_purchases = fut_eligible[~is_ret_fut].copy()

    fut_order_counts = fut_purchases.groupby("customer_id")["invoice"].nunique()
    orders_series = base["customer_id"].map(fut_order_counts).fillna(0).astype(int)
    base["churn"] = (orders_series == 0).astype(int)
    base["next_purchase_flag"] = (1 - base["churn"]).astype(int)

    fut_net_spend = fut_eligible.groupby("customer_id")["line_total"].sum()
    base["future_spend"] = (
        base["customer_id"].map(fut_net_spend).fillna(0.0).clip(lower=0.0).round(4)
    )

    if not fut_purchases.empty:
        desc_series = (
            fut_purchases["description"]
            if "description" in fut_purchases.columns
            else pd.Series("", index=fut_purchases.index)
        )
        fut_purchases["derived_category"] = [
            derive_product_category(code, desc)
            for code, desc in zip(fut_purchases["stock_code"], desc_series, strict=True)
        ]
        cat_qty = (
            fut_purchases.groupby(["customer_id", "derived_category"], as_index=False)["quantity"]
            .sum()
            .sort_values(
                ["customer_id", "quantity", "derived_category"], ascending=[True, False, True]
            )
        )
        top_cat_map = (
            cat_qty.drop_duplicates(subset=["customer_id"], keep="first")
            .set_index("customer_id")["derived_category"]
            .to_dict()
        )
        base["next_purchase_category"] = base["customer_id"].map(top_cat_map).fillna("NONE")
        base.loc[base["churn"] == 1, "next_purchase_category"] = "NONE"
    else:
        base["next_purchase_category"] = "NONE"

    n_total = len(base)
    n_churned = int((base["churn"] == 1).sum())
    n_active = int((base["churn"] == 0).sum())
    churn_rate = 100.0 * n_churned / max(n_total, 1)

    summary_msg = (
        f"Labels summary | customers={n_total} | churned={n_churned} ({churn_rate:.2f}%) | "
        f"retained={n_active} ({100.0 - churn_rate:.2f}%) | mean_future_spend={base['future_spend'].mean():.2f}"
    )
    print(summary_msg)
    logger.info(summary_msg)
    return base
