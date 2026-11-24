"""Pydantic v2 request and response schemas for Vantara Customer Behavior Prediction API."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """Single-customer prediction request accepting either customer_id or raw features."""

    customer_id: str | None = Field(default=None, examples=["12347"])
    recency_days: float | None = Field(default=None, ge=0.0, examples=[45.0])
    frequency: float | None = Field(default=None, ge=1.0, examples=[6.0])
    monetary_gross: float | None = Field(default=None, ge=0.0, examples=[1450.0])
    monetary_net: float | None = Field(default=None, examples=[1420.0])
    avg_order_value: float | None = Field(default=None, ge=0.0, examples=[240.0])
    unique_products: float | None = Field(default=None, ge=1.0, examples=[32.0])
    tenure_days: float | None = Field(default=None, ge=0.0, examples=[220.0])
    country_grouped: str = Field(default="United Kingdom", examples=["United Kingdom"])

    @model_validator(mode="after")
    def check_id_or_features(self) -> "PredictRequest":
        """Ensure either customer_id or core RFM features are provided."""
        has_id = self.customer_id is not None and str(self.customer_id).strip() != ""
        has_rfm = (
            self.recency_days is not None
            and self.frequency is not None
            and self.monetary_gross is not None
        )
        if not has_id and not has_rfm:
            raise ValueError(
                "Provide either 'customer_id' or raw features ('recency_days', 'frequency', 'monetary_gross')."
            )
        return self


class ReasonItem(BaseModel):
    """Single SHAP feature attribution driver."""

    feature: str
    shap_value: float
    abs_impact: float
    direction: str


class PredictResponse(BaseModel):
    """Structured prediction response for a single customer."""

    customer_id: str
    churn_probability: float
    churn_flag: int
    risk_tier: str
    predicted_clv: float
    segment_name: str
    is_anomaly: int
    top_reasons: list[ReasonItem]
    narrative: str
    recommended_action: str
    model_version: str


class BatchPredictResponse(BaseModel):
    """Summary and scored rows returned from CSV batch upload."""

    records_scored: int
    high_risk_count: int
    total_predicted_clv: float
    predictions: list[dict[str, Any]]
