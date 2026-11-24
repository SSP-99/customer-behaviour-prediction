CREATE TABLE IF NOT EXISTS customer_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(64) NOT NULL,
    churn_probability REAL NOT NULL,
    churn_prediction INTEGER NOT NULL,
    risk_tier VARCHAR(32) NOT NULL,
    predicted_clv_90d REAL NOT NULL,
    segment_name VARCHAR(64),
    model_version VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS batch_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    records_scored INTEGER NOT NULL,
    high_risk_count INTEGER NOT NULL,
    total_predicted_clv REAL NOT NULL,
    notes TEXT
);