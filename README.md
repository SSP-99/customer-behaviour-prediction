# Vantara Retail Solutions — Customer Behavior Prediction & Retention Intelligence Platform

End-to-end production machine learning platform predicting **90-day customer churn**, **90-day Customer Lifetime Value (CLV)**, **RFM customer personas**, **market-basket cross-sell recommendations**, and **SHAP + LIME plain-language explanations** on the UCI Online Retail II dataset.

## Quickstart (Local)

```bash
# 1. Run full end-to-end ML pipeline (data -> features -> train -> segment -> explain)
python -m src.pipeline --stage all

# 2. Generate model comparison report & production metadata
python -m src.models.compare

# 3. Run unit & integration tests + API latency benchmark
pytest
python -m scripts.benchmark_api

# 4. Launch FastAPI backend (Terminal 1)
uvicorn api.main:app --host 127.0.0.1 --port 8000

# 5. Launch Streamlit dashboard (Terminal 2)
streamlit run frontend/dashboard.py --server.port 8501
```

## Quickstart (Docker Compose)

```bash
docker compose up --build
```
- **Swagger API Docs**: `http://localhost:8000/docs`
- **Streamlit Dashboard**: `http://localhost:8501`

## Key Verified Results
- **Churn Classification Winner (`XGBoost`)**: Test ROC-AUC = **0.8214** (Target >= 0.80: **PASS**), Test Recall = **0.8285** (Target >= 0.70: **PASS**), Test F1 = **0.7815**.
- **90-Day CLV Regression Winner (`XGBoost`)**: Validation R² = **0.7112** (MAE = **£309.32**); Test R² = **0.1997** (heavy Q4 wholesale tail skew documented in `docs/final_report.md`).
- **API Latency (500 requests)**: p50 = **70.24 ms**, p95 = **79.39 ms**, p99 = **217.51 ms** (SLA < 400 ms: **PASS**).

## Dataset Attribution
Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository (CC BY 4.0).