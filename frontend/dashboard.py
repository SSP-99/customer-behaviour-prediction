"""Vantara Retail Solutions â€” Multi-Page Executive & Retention Intelligence Dashboard."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Vantara Retail Solutions â€” Retention AI",
    page_icon="ðŸ“Š",
    layout="wide",
)

st.sidebar.title("Vantara Retail Solutions")
st.sidebar.caption("Customer Churn, CLV & Cross-Sell Intelligence")


@st.cache_data(ttl=60)
def fetch_json(endpoint: str) -> dict[str, Any]:
    """Fetch JSON payload from FastAPI service with caching."""
    resp = requests.get(f"{API_URL}{endpoint}", timeout=10)
    resp.raise_for_status()
    return dict(resp.json())


try:
    health = fetch_json("/health")
    st.sidebar.success(f"API Connected ({health.get('customers_indexed', 0):,} customers)")
except Exception as exc:
    st.error(
        f"Cannot connect to FastAPI backend at `{API_URL}`. "
        f"Start it with `uvicorn api.main:app --port 8000`. Details: {exc}"
    )
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Executive Overview",
        "2. RFM Segmentation",
        "3. Churn Risk Leaderboard",
        "4. Product Cross-Sell Recommender",
        "5. Customer SHAP & LIME Explainer",
        "6. Batch CSV Scoring",
        "7. Model Governance & KPIs",
    ],
)

leaderboard_data = fetch_json("/leaderboard?top=500")
df_cohort = pd.DataFrame(leaderboard_data.get("customers", []))

if page == "1. Executive Overview":
    st.title("Executive Retention & CLV Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Eligible Customers", f"{leaderboard_data['total_customers']:,}")
    c2.metric("Predicted Churn Rate", f"{leaderboard_data['churn_rate'] * 100:.1f}%")
    c3.metric("Total 90d CLV", f"${leaderboard_data['total_predicted_clv'] / 1e6:.2f}M")
    c4.metric("High-Risk Customers", f"{leaderboard_data['high_risk_count']:,}")
    c5.metric("90d Revenue at Risk", f"${leaderboard_data['total_revenue_at_risk'] / 1e3:.1f}K")

    if not df_cohort.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_tier = px.pie(
                df_cohort,
                names="risk_tier",
                title="Top-500 At-Risk Cohort by Risk Tier",
                color="risk_tier",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            )
            st.plotly_chart(fig_tier, use_container_width=True)
        with col_b:
            fig_scatter = px.scatter(
                df_cohort.head(250),
                x="recency_days",
                y="predicted_clv",
                color="risk_tier",
                size="revenue_at_risk",
                hover_data=["customer_id", "segment_name", "churn_probability"],
                title="Recency vs. Predicted 90d CLV (Sized by Revenue at Risk)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "2. RFM Segmentation":
    st.title("RFM Customer Personas & Cluster Profiles")
    seg_data = fetch_json("/segments/summary")
    df_prof = pd.DataFrame(seg_data.get("profiles", []))
    st.dataframe(df_prof, use_container_width=True)
    if not df_prof.empty:
        fig_seg = px.bar(
            df_prof,
            x="segment_name",
            y="customer_count",
            color="avg_monetary_net",
            title="Customer Count & Average Net Spend by Segment",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

elif page == "3. Churn Risk Leaderboard":
    st.title("High-Value Churn Risk Leaderboard")
    if not df_cohort.empty:
        tiers = st.multiselect(
            "Filter Risk Tier", ["High", "Medium", "Low"], default=["High", "Medium"]
        )
        filtered = df_cohort[df_cohort["risk_tier"].isin(tiers)]
        st.dataframe(filtered, use_container_width=True)
        st.download_button(
            "Download Filtered Leaderboard CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="vantara_churn_leaderboard.csv",
            mime="text/csv",
        )

elif page == "4. Product Cross-Sell Recommender":
    st.title("Market-Basket Cross-Sell Recommender")
    sku = st.text_input("Enter Product StockCode", value="85123A")
    top_n = st.slider("Number of Recommendations", 1, 15, 5)
    if st.button("Get Recommendations"):
        rec_res = fetch_json(f"/recommend/{sku}?top_n={top_n}")
        st.dataframe(pd.DataFrame(rec_res.get("recommendations", [])), use_container_width=True)

elif page == "5. Customer SHAP & LIME Explainer":
    st.title("Individual Customer Churn Explainability (SHAP + LIME)")
    default_cid = str(df_cohort["customer_id"].iloc) if not df_cohort.empty else "12347"
    cid_input = st.text_input("Customer ID", value=default_cid)
    if st.button("Explain Customer"):
        try:
            exp = fetch_json(f"/explain/{cid_input}")
            m1, m2 = st.columns(2)
            m1.metric("Churn Probability", f"{exp['churn_probability'] * 100:.1f}%")
            m2.metric("Risk Tier", exp["risk_tier"])
            st.info(exp["narrative"])
            st.success(f"**Recommended Action:** {exp['recommended_action']}")

            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("SHAP Feature Attributions")
                df_shap = pd.DataFrame(exp.get("shap_top_drivers", []))
                if not df_shap.empty:
                    fig_s = px.bar(
                        df_shap, x="shap_value", y="feature", orientation="h", color="direction"
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
            with c_right:
                st.subheader("LIME Local Rule Weights")
                st.dataframe(
                    pd.DataFrame(exp.get("lime_top_drivers", [])), use_container_width=True
                )
        except Exception as exc:
            st.error(f"Could not explain customer '{cid_input}': {exc}")

elif page == "6. Batch CSV Scoring":
    st.title("Batch Customer Scoring (CSV Upload)")
    uploaded = st.file_uploader("Upload CSV with customer_id and RFM features", type=["csv"])
    if uploaded is not None:
        resp = requests.post(
            f"{API_URL}/predict/batch",
            files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            timeout=30,
        )
        if resp.status_code == 200:
            batch_res = resp.json()
            b1, b2, b3 = st.columns(3)
            b1.metric("Records Scored", batch_res["records_scored"])
            b2.metric("High Risk Flagged", batch_res["high_risk_count"])
            b3.metric("Total Predicted 90d CLV", f"${batch_res['total_predicted_clv']:,.2f}")
            out_df = pd.DataFrame(batch_res["predictions"])
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download Scored Results CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="vantara_batch_predictions.csv",
                mime="text/csv",
            )
        else:
            st.error(f"Batch scoring failed: {resp.text}")

elif page == "7. Model Governance & KPIs":
    st.title("Model Governance, Validation & Test KPIs")
    meta = fetch_json("/model/metadata")
    st.json(meta)
    fig_path = Path("docs/figures/global_feature_importance.png")
    if fig_path.exists():
        st.image(str(fig_path), caption="Global SHAP Feature Importance")
