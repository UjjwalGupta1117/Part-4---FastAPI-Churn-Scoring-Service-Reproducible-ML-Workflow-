"""
app/main.py — D2C Churn Scoring Service
Run: uvicorn app.main:app --reload
"""

import pickle
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "model" / "model.pkl"
FEATS_PATH  = BASE_DIR / "model" / "feature_cols.json"
THRESHOLD   = 0.40

# ── Load model once at startup ─────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(FEATS_PATH, "r") as f:
    FEATURE_COLS = json.load(f)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="D2C Churn Scoring Service",
    description="Scores customers on 60-day churn probability for the retention team.",
    version="1.0.0",
)


# ── Pydantic input schema ──────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Input schema for a single customer scoring request.
    All fields correspond to features in rfm_modeling_snapshot.csv.
    Categorical fields are label-encoded integers (same encoding used during training).
    """
    # Demographic (label-encoded)
    city_tier:            int   = Field(..., ge=0, le=2,  description="0=Tier1, 1=Tier2, 2=Tier3")
    age_group:            int   = Field(..., ge=0, le=3,  description="0=18-24, 1=25-34, 2=35-44, 3=45+")
    acquisition_channel:  int   = Field(..., ge=0, le=5,  description="0-5 encoded channels")
    loyalty_tier:         int   = Field(..., ge=0, le=3,  description="0=Gold, 1=None, 2=Platinum, 3=Silver")
    preferred_category:   int   = Field(..., ge=0, le=5,  description="0-5 encoded categories")
    marketing_consent:    int   = Field(..., ge=0, le=1,  description="0=No, 1=Yes")
    is_loyalty_member:    int   = Field(..., ge=0, le=1,  description="1 if loyalty_tier is not null")
    has_rated:            int   = Field(..., ge=0, le=1,  description="1 if customer has given at least one rating")

    # Transactional RFM
    recency_days:         int   = Field(..., ge=0,        description="Days since last order")
    frequency_180d:       int   = Field(..., ge=0,        description="Orders in last 180 days")
    monetary_180d:        float = Field(..., ge=0.0,      description="Total spend (INR) in last 180 days")

    # Behavioural
    return_rate_180d:     float = Field(..., ge=0.0, le=1.0)
    avg_discount_pct_180d:float = Field(..., ge=0.0, le=1.0)
    avg_rating_180d:      float = Field(..., ge=1.0, le=5.0)
    category_diversity_180d: int = Field(..., ge=0)

    # Support
    ticket_count_90d:          int   = Field(..., ge=0)
    negative_ticket_rate_90d:  float = Field(..., ge=0.0, le=1.0)
    avg_resolution_hours_90d:  float = Field(..., ge=0.0)

    # Customer age
    days_since_signup: int = Field(..., ge=0)

    # Web engagement
    sessions_30d:         int   = Field(..., ge=0)
    product_views_30d:    int   = Field(..., ge=0)
    cart_adds_30d:        int   = Field(..., ge=0)
    wishlist_adds_30d:    int   = Field(..., ge=0)
    abandoned_carts_30d:  int   = Field(..., ge=0)
    email_opens_30d:      int   = Field(..., ge=0)
    campaign_clicks_30d:  int   = Field(..., ge=0)
    last_visit_days_ago:  int   = Field(..., ge=0)

    @field_validator("monetary_180d")
    def monetary_not_negative(cls, v):
        if v < 0:
            raise ValueError("monetary_180d must be >= 0")
        return v


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=500)


# ── Response helpers ───────────────────────────────────────────────────────────
def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def risk_explanation(features: CustomerFeatures, prob: float) -> str:
    reasons = []
    if features.recency_days > 90:
        reasons.append(f"last order was {features.recency_days} days ago")
    if features.sessions_30d == 0:
        reasons.append("no app/web sessions in the last 30 days")
    if features.frequency_180d <= 1:
        reasons.append("only 1 or fewer orders in the last 180 days")
    if features.negative_ticket_rate_90d >= 0.5:
        reasons.append("majority of recent support tickets had negative sentiment")
    if features.last_visit_days_ago > 20:
        reasons.append(f"last site visit was {features.last_visit_days_ago} days ago")
    if not reasons:
        reasons.append("moderate risk based on combined purchase and engagement signals")
    return "Churn risk drivers: " + "; ".join(reasons) + "."


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    """
    Returns service status and confirms model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_count": len(FEATURE_COLS),
        "threshold": THRESHOLD,
    }


@app.post("/predict", tags=["Scoring"])
def predict(customer: CustomerFeatures):
    """
    Score a single customer.

    Returns:
    - churn_probability: float [0, 1]
    - predicted_churn: bool (True if prob >= threshold)
    - risk_label: LOW / MEDIUM / HIGH
    - explanation: human-readable reason string
    """
    try:
        row = np.array([[getattr(customer, col) for col in FEATURE_COLS]])
        prob = float(MODEL.predict_proba(row)[0, 1])
        pred = prob >= THRESHOLD
        return {
            "churn_probability": round(prob, 4),
            "predicted_churn":   bool(pred),
            "risk_label":        risk_label(prob),
            "explanation":       risk_explanation(customer, prob),
            "threshold_used":    THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", tags=["Scoring"])
def batch_predict(request: BatchRequest):
    """
    Score a batch of customers (max 500 per request).

    Returns a list of results in the same order as the input.
    """
    try:
        rows = np.array([
            [getattr(c, col) for col in FEATURE_COLS]
            for c in request.customers
        ])
        probas = MODEL.predict_proba(rows)[:, 1]
        results = []
        for i, (customer, prob) in enumerate(zip(request.customers, probas)):
            prob = float(prob)
            results.append({
                "index":            i,
                "churn_probability": round(prob, 4),
                "predicted_churn":   bool(prob >= THRESHOLD),
                "risk_label":        risk_label(prob),
                "explanation":       risk_explanation(customer, prob),
            })
        return {
            "count":    len(results),
            "threshold": THRESHOLD,
            "results":  results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
