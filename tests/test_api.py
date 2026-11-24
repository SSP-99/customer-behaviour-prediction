"""Integration tests for FastAPI endpoints using FastAPI TestClient."""

from fastapi.testclient import TestClient

from api.main import app


def test_api_endpoints_end_to_end() -> None:
    """Verify /health, /model/metadata, /predict, 422 validation, /predict/batch, and /explain."""
    with TestClient(app) as client:
        # 1. Health check
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "healthy"
        assert r_health.json()["models_loaded"] is True

        # 2. Metadata
        r_meta = client.get("/model/metadata")
        assert r_meta.status_code == 200
        assert "churn_model" in r_meta.json()

        # 3. Leaderboard & single customer lookup
        r_lead = client.get("/leaderboard?top=5")
        assert r_lead.status_code == 200
        top_customers = r_lead.json()["customers"]
        assert len(top_customers) > 0
        sample_cid = str(next(iter(top_customers))["customer_id"])

        # 4. Valid single prediction (by customer_id)
        r_pred = client.post("/predict", json={"customer_id": sample_cid})
        assert r_pred.status_code == 200
        pred_body = r_pred.json()
        assert 0.0 <= pred_body["churn_probability"] <= 1.0
        assert pred_body["risk_tier"] in {"Low", "Medium", "High"}

        # 5. Valid single prediction (by raw features)
        r_raw = client.post(
            "/predict",
            json={"recency_days": 120.0, "frequency": 2.0, "monetary_gross": 180.0},
        )
        assert r_raw.status_code == 200

        # 6. Invalid single prediction -> 422 Unprocessable Entity
        r_bad = client.post("/predict", json={})
        assert r_bad.status_code == 422

        # 7. Explain endpoint (SHAP + LIME + narrative)
        r_exp = client.get(f"/explain/{sample_cid}")
        assert r_exp.status_code == 200
        exp_body = r_exp.json()
        assert len(exp_body["shap_top_drivers"]) > 0
        assert len(exp_body["lime_top_drivers"]) > 0

        # 8. Batch CSV upload
        csv_content = "customer_id,recency_days,frequency,monetary_gross\nC_TEST_1,15,10,2500\nC_TEST_2,200,1,45\n"
        r_batch = client.post(
            "/predict/batch",
            files={"file": ("batch.csv", csv_content.encode("utf-8"), "text/csv")},
        )
        assert r_batch.status_code == 200
        assert r_batch.json()["records_scored"] == 2
