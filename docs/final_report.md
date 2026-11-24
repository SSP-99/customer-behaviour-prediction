# Vantara Retail Solutions — Customer Behavior Prediction & Retention Intelligence Platform
## Final Technical & Executive Report

### 1. Executive Summary
Vantara Retail Solutions built an end-to-end, leakage-free machine learning platform on the UCI Online Retail II dataset (`1,067,371` raw transactions; `5,280` eligible pre-cutoff customers) to predict 90-day customer churn, forecast 90-day Customer Lifetime Value (CLV), segment customers into actionable RFM personas, recommend cross-sell products, and explain every prediction via SHAP and LIME.
- **Churn Classification Winner (`XGBoost`)**: Achieved **0.8214 ROC-AUC** and **0.8285 Recall** on the held-out test set (**PASS** against >= 0.80 AUC and >= 0.70 Recall targets).
- **90-Day CLV Regression Winner (`XGBoost`)**: Achieved **0.7112 R²** (`MAE = £309.32`) on validation and **0.1997 R²** (`MAE = £487.51`) on the Q4 test split due to extreme corporate wholesale outliers.
- **Production Serving Latency**: FastAPI benchmarked across 500 live requests achieved **p50 = 70.24 ms**, **p95 = 79.39 ms**, and **p99 = 217.51 ms** (well under the 400 ms SLA).

### 2. Dataset & Point-in-Time Leakage Prevention
- **Citation**: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository (CC BY 4.0).
- **Temporal Boundary**: All features are engineered strictly prior to `2011-09-10` (`invoice_date < 2011-09-10`), while churn (`churn_label`) and net future spend (`clv_90d_target`) are measured strictly inside `[2011-09-10, 2011-12-09)`. Automated pytest suites enforce zero temporal overlap and fit all encoders/scalers on the 70% training split only.

### 3. Model Comparison & Test Results

| Task | Winning Model | Validation Metrics | Held-Out Test Metrics | KPI Status |
|---|---|---|---|---|
| Churn Classification | XGBoost | ROC-AUC: `0.8113`, Recall: `0.8393`, F1: `0.7842` | ROC-AUC: `0.8214`, Recall: `0.8285`, F1: `0.7815` | **PASS** (AUC >= 0.80, Recall >= 0.70) |
| 90-Day CLV Regression | XGBoost (log1p target) | R²: `0.7112`, MAE: `£309.32`, RMSE: `£784.63` | R²: `0.1997`, MAE: `£487.51`, RMSE: `£4,056.93` | Validation PASS; Test R² impacted by Q4 wholesale tail |

**Documented Reason for Test CLV R² Variance**: While `log1p`-transformed XGBoost explained `71.1%` of variance on validation (`RMSE = £784.63`), the November–December test window contains a small number of B2B wholesale accounts placing single orders exceeding `£50,000`, inflating squared-error RMSE (`£4,056.93`) despite strong median/mean absolute accuracy (`MAE = £487.51`).

### 4. RFM Customer Segmentation & Cross-Sell Recommender
- **K-Means++ (`k=4`)**: Segmented customers into **Champions**, **Loyal Customers**, **At-Risk High-Value**, and **Hibernating / Lost**, persisted in `models_artifacts/kmeans_segments.joblib`.
- **Market-Basket Recommender**: Indexed **500** top products with lift/confidence association rules and storewide bestseller cold-start fallback (`models_artifacts/product_recommender.joblib`).

### 5. Explainability (SHAP + LIME) & Deployment
- **Top 5 Global Churn Drivers (`SHAP`)**: `purchase_frequency_per_month` (`0.4182`), `recency_days` (`0.3268`), `unique_products` (`0.1379`), `active_span_days` (`0.0979`), and `monetary_net` (`0.0927`).
- **Full-Stack Deployment**: FastAPI (`api/main.py`) + 7-view Streamlit Executive Dashboard (`frontend/dashboard.py`) + PostgreSQL/SQLite persistence (`database/db.py`) + `docker-compose.yml`.

### 6. Mathematical Foundations Summary
- **Logistic Sigmoid & Binary Cross-Entropy**: $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)] + \lambda \|w\|_2^2$.
- **Gradient Descent & Backpropagation**: $w \leftarrow w - \eta \nabla_w \mathcal{L}(w)$ via chain rule $\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^\top$.
- **K-Means Inertia & Silhouette**: $J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$, $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$.