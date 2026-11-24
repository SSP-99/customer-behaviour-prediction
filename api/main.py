"""FastAPI REST service for Vantara Customer Behavior Prediction Platform."""

import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from database.db import init_db, save_prediction_record
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import BatchPredictResponse, PredictRequest, PredictResponse, ReasonItem
from src.explainability.narrative import (
    build_plain_language_narrative,
    classify_risk_tier,
    explain_with_lime_or_surrogate,
)
from src.features.preprocessing import TabularPreprocessor, get_feature_columns, transform_features
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained models, preprocessor, explainer, and lookup tables once at startup."""
    cfg = load_config("config/config.yaml")
    STATE["config"] = cfg
    STATE["db_engine"] = init_db(cfg)

    assert cfg.artifacts is not None
    STATE["preprocessor"] = joblib.load(cfg.artifacts.preprocessor_path)
    STATE["churn_bundle"] = joblib.load(cfg.artifacts.best_churn_model_path)
    STATE["clv_bundle"] = joblib.load(cfg.artifacts.best_clv_model_path)
    STATE["kmeans_bundle"] = joblib.load(cfg.artifacts.kmeans_model_path)
    STATE["recommender"] = joblib.load(cfg.artifacts.recommender_rules_path)
    STATE["explainer"] = joblib.load(cfg.artifacts.shap_explainer_path)

    meta_path = Path("models_artifacts/model_metadata.json")
    STATE["metadata"] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )

    df_feat = pd.read_parquet(cfg.data.processed_features_path)
    df_feat["customer_id"] = df_feat["customer_id"].astype(str)
    STATE["features_df"] = df_feat
    STATE["features_by_id"] = df_feat.set_index("customer_id", drop=False)

    seg_path = Path("data/processed/customer_segments.parquet")
    if seg_path.exists():
        seg_df = pd.read_parquet(seg_path)
        seg_df["customer_id"] = seg_df["customer_id"].astype(str)
        STATE["segments_by_id"] = seg_df.set_index("customer_id")["segment_name"].to_dict()
    else:
        STATE["segments_by_id"] = {}

    shap_path = Path("data/processed/shap_values.parquet")
    if shap_path.exists():
        shap_df = pd.read_parquet(shap_path)
        shap_df["customer_id"] = shap_df["customer_id"].astype(str)
        STATE["shap_by_id"] = shap_df.set_index("customer_id").to_dict(orient="index")
    else:
        STATE["shap_by_id"] = {}

    prep: TabularPreprocessor = STATE["preprocessor"]
    x_all = transform_features(prep, df_feat, cfg)
    STATE["x_train"] = x_all[(df_feat["split"] == "train").to_numpy()]

    churn_probs = STATE["churn_bundle"]["model"].predict_proba(x_all)[:, 1]
    clv_preds = STATE["clv_bundle"]["model"].predict(x_all)
    thresh = float(STATE["churn_bundle"].get("threshold", 0.45))

    cohort = pd.DataFrame(
        {
            "customer_id": df_feat["customer_id"].values,
            "country_grouped": df_feat["country_grouped"].values,
            "recency_days": np.round(df_feat["recency_days"].values, 1),
            "frequency": df_feat["frequency"].values,
            "monetary_net": np.round(df_feat["monetary_net"].values, 2),
            "churn_probability": np.round(churn_probs, 4),
            "churn_flag": (churn_probs >= thresh).astype(int),
            "predicted_clv": np.round(clv_preds, 2),
            "segment_name": [
                STATE["segments_by_id"].get(c, "Unassigned") for c in df_feat["customer_id"]
            ],
        }
    )
    cohort["risk_tier"] = [classify_risk_tier(float(p))[0] for p in cohort["churn_probability"]]
    cohort["revenue_at_risk"] = np.round(cohort["churn_probability"] * cohort["predicted_clv"], 2)
    STATE["cohort_df"] = cohort
    yield
    STATE.clear()


app = FastAPI(
    title="Vantara Retail Solutions â€” Customer Behavior Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "models_loaded": "churn_bundle" in STATE and "clv_bundle" in STATE,
        "customers_indexed": len(STATE.get("cohort_df", [])),
    }


@app.get("/model/metadata")
def get_model_metadata() -> dict[str, Any]:
    return STATE.get("metadata", {})


@app.get("/segments/summary")
def get_segments_summary() -> dict[str, Any]:
    km = STATE["kmeans_bundle"]
    return {
        "persona_map": km["persona_map"],
        "k_diagnostics": km["k_diagnostics"],
        "profiles": km["segment_profiles"],
    }


@app.get("/leaderboard")
def get_leaderboard(top: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    cohort: pd.DataFrame = STATE["cohort_df"]
    ranked = cohort.sort_values("revenue_at_risk", ascending=False).head(top)
    return {
        "total_customers": len(cohort),
        "churn_rate": round(float(cohort["churn_flag"].mean()), 4),
        "total_predicted_clv": round(float(cohort["predicted_clv"].sum()), 2),
        "high_risk_count": int((cohort["risk_tier"] == "High").sum()),
        "total_revenue_at_risk": round(float(cohort["revenue_at_risk"].sum()), 2),
        "customers": ranked.to_dict(orient="records"),
    }


@app.get("/customers/{customer_id}")
def get_customer_profile(customer_id: str) -> dict[str, Any]:
    cid = str(customer_id).strip()
    cohort: pd.DataFrame = STATE["cohort_df"]
    match = cohort[cohort["customer_id"] == cid]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{cid}' not found.")
    return next(iter(match.to_dict(orient="records")))


@app.get("/recommend/{stock_code}")
def recommend_products(
    stock_code: str, top_n: int = Query(default=5, ge=1, le=20)
) -> dict[str, Any]:
    items = STATE["recommender"].recommend_for_item(stock_code, top_n=top_n)
    return {"stock_code": stock_code.strip().upper(), "recommendations": items}


@app.get("/explain/{customer_id}")
def explain_customer(customer_id: str) -> dict[str, Any]:
    cid = str(customer_id).strip()
    by_id: pd.DataFrame = STATE["features_by_id"]
    if cid not in by_id.index:
        raise HTTPException(status_code=404, detail=f"Customer '{cid}' not found.")

    cfg = STATE["config"]
    prep: TabularPreprocessor = STATE["preprocessor"]
    row_df = by_id.loc[[cid]]
    x_row = transform_features(prep, row_df, cfg)
    shap_res = STATE["explainer"].explain_instance(x_row, top_k=6)

    pre = STATE["shap_by_id"].get(cid)
    if pre:
        narr_info = {
            "risk_tier": pre["risk_tier"],
            "churn_probability": pre["churn_probability"],
            "narrative": pre["narrative"],
            "recommended_action": pre["recommended_action"],
        }
    else:
        narr_info = build_plain_language_narrative(
            row_df.iloc[0],
            STATE["features_df"],
            shap_res["top_drivers"],
            shap_res["predicted_probability"],
        )

    lime_drivers = explain_with_lime_or_surrogate(
        STATE["x_train"],
        x_row,
        STATE["churn_bundle"]["model"],
        prep.get_feature_names_out().tolist(),
        top_k=5,
    )
    return {
        "customer_id": cid,
        **narr_info,
        "shap_top_drivers": shap_res["top_drivers"],
        "lime_top_drivers": lime_drivers,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_single(req: PredictRequest) -> PredictResponse:
    cfg = STATE["config"]
    prep: TabularPreprocessor = STATE["preprocessor"]
    by_id: pd.DataFrame = STATE["features_by_id"]
    df_all: pd.DataFrame = STATE["features_df"]

    if req.customer_id and str(req.customer_id).strip() in by_id.index:
        cid = str(req.customer_id).strip()
        row_df = by_id.loc[[cid]].copy()
    else:
        cid = str(req.customer_id or "ADHOC_CUSTOMER").strip()
        num_cols, _ = get_feature_columns(df_all)
        payload = {**df_all[num_cols].median().to_dict(), "country_grouped": req.country_grouped}
        for field in (
            "recency_days",
            "frequency",
            "monetary_gross",
            "monetary_net",
            "avg_order_value",
            "unique_products",
            "tenure_days",
        ):
            val = getattr(req, field)
            if val is not None:
                payload[field] = float(val)
        if req.monetary_net is None and req.monetary_gross is not None:
            payload["monetary_net"] = float(req.monetary_gross)
        row_df = pd.DataFrame([payload])

    x_vec = transform_features(prep, row_df, cfg)
    prob = float(STATE["churn_bundle"]["model"].predict_proba(x_vec)[0, 1])
    thresh = float(STATE["churn_bundle"].get("threshold", 0.45))
    flag = int(prob >= thresh)
    clv = float(STATE["clv_bundle"]["model"].predict(x_vec)[0])

    km_b = STATE["kmeans_bundle"]
    rfm_vals = np.log1p(np.maximum(row_df[km_b["rfm_cols"]].to_numpy(dtype=np.float64), 0.0))
    seg_id = int(km_b["model"].predict((rfm_vals - km_b["means"]) / km_b["stds"])[0])
    seg_name = str(km_b["persona_map"].get(seg_id, "Loyal Customers"))

    shap_res = STATE["explainer"].explain_instance(x_vec, top_k=5)
    narr = build_plain_language_narrative(row_df.iloc[0], df_all, shap_res["top_drivers"], prob)
    model_ver = f"{STATE['churn_bundle']['model_name']}_v1"

    save_prediction_record(
        STATE["db_engine"],
        {
            "customer_id": cid,
            "churn_probability": round(prob, 4),
            "churn_prediction": flag,
            "risk_tier": narr["risk_tier"],
            "predicted_clv_90d": round(clv, 2),
            "segment_name": seg_name,
            "model_version": model_ver,
        },
    )

    return PredictResponse(
        customer_id=cid,
        churn_probability=round(prob, 4),
        churn_flag=flag,
        risk_tier=narr["risk_tier"],
        predicted_clv=round(clv, 2),
        segment_name=seg_name,
        is_anomaly=int(float(row_df["monetary_gross"].iloc[0]) > 10000.0),
        top_reasons=[ReasonItem(**d) for d in shap_res["top_drivers"]],
        narrative=narr["narrative"],
        recommended_action=narr["recommended_action"],
        model_version=model_ver,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(file: UploadFile) -> BatchPredictResponse:
    raw_bytes = await file.read()
    try:
        upload_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV upload: {exc}") from exc

    if upload_df.empty or len(upload_df) > 10000:
        raise HTTPException(status_code=422, detail="CSV must contain between 1 and 10,000 rows.")

    cfg = STATE["config"]
    prep: TabularPreprocessor = STATE["preprocessor"]
    df_all: pd.DataFrame = STATE["features_df"]
    num_cols, _ = get_feature_columns(df_all)
    medians = df_all[num_cols].median()

    scored_input = upload_df.copy()
    for col in num_cols:
        scored_input[col] = (
            medians[col]
            if col not in scored_input.columns
            else pd.to_numeric(scored_input[col], errors="coerce").fillna(medians[col])
        )
    if "country_grouped" not in scored_input.columns:
        scored_input["country_grouped"] = "United Kingdom"

    x_mat = transform_features(prep, scored_input, cfg)
    probs = STATE["churn_bundle"]["model"].predict_proba(x_mat)[:, 1]
    clvs = STATE["clv_bundle"]["model"].predict(x_mat)
    thresh = float(STATE["churn_bundle"].get("threshold", 0.45))

    cids = (
        scored_input["customer_id"].astype(str).tolist()
        if "customer_id" in scored_input.columns
        else [f"ROW_{i + 1}" for i in range(len(scored_input))]
    )
    rows_out: list[dict[str, Any]] = []
    high_cnt = 0
    for i, cid in enumerate(cids):
        p, c = float(probs[i]), float(clvs[i])
        tier, action = classify_risk_tier(p)
        if tier == "High":
            high_cnt += 1
        rows_out.append(
            {
                "customer_id": cid,
                "churn_probability": round(p, 4),
                "churn_flag": int(p >= thresh),
                "risk_tier": tier,
                "predicted_clv": round(c, 2),
                "revenue_at_risk": round(p * c, 2),
                "recommended_action": action,
            }
        )

    return BatchPredictResponse(
        records_scored=len(rows_out),
        high_risk_count=high_cnt,
        total_predicted_clv=round(float(np.sum(clvs)), 2),
        predictions=rows_out,
    )
