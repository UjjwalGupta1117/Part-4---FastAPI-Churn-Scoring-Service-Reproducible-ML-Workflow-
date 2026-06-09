"""
tests/test_api.py

Run: pytest tests/ -v
Requires the API to be running: uvicorn app.main:app --reload
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ── Shared valid payload ───────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "city_tier": 0,
    "age_group": 1,
    "acquisition_channel": 2,
    "loyalty_tier": 3,
    "preferred_category": 0,
    "marketing_consent": 1,
    "is_loyalty_member": 1,
    "has_rated": 1,
    "recency_days": 45,
    "frequency_180d": 3,
    "monetary_180d": 2500.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.2,
    "avg_rating_180d": 4.0,
    "category_diversity_180d": 2,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 300,
    "sessions_30d": 5,
    "product_views_30d": 10,
    "cart_adds_30d": 2,
    "wishlist_adds_30d": 1,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 3,
    "campaign_clicks_30d": 1,
    "last_visit_days_ago": 4,
}

HIGH_RISK_PAYLOAD = {
    **VALID_PAYLOAD,
    "recency_days": 250,
    "sessions_30d": 0,
    "frequency_180d": 0,
    "last_visit_days_ago": 55,
    "negative_ticket_rate_90d": 0.8,
    "monetary_180d": 0.0,
}

LOW_RISK_PAYLOAD = {
    **VALID_PAYLOAD,
    "recency_days": 5,
    "sessions_30d": 20,
    "frequency_180d": 8,
    "monetary_180d": 8000.0,
    "last_visit_days_ago": 1,
}


# ── Test 1: Health endpoint ────────────────────────────────────────────────────
class TestHealth:
    def test_health_status_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True

    def test_health_threshold_present(self):
        resp = client.get("/health")
        data = resp.json()
        assert "threshold" in data
        assert 0 < data["threshold"] < 1


# ── Test 2: Single predict ─────────────────────────────────────────────────────
class TestPredict:
    def test_predict_valid_payload_returns_200(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_has_required_fields(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "churn_probability" in data
        assert "predicted_churn" in data
        assert "risk_label" in data
        assert "explanation" in data

    def test_predict_probability_in_range(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        prob = resp.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_label_valid(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        label = resp.json()["risk_label"]
        assert label in {"LOW", "MEDIUM", "HIGH"}

    def test_predict_high_risk_customer(self):
        resp = client.post("/predict", json=HIGH_RISK_PAYLOAD)
        assert resp.status_code == 200
        prob = resp.json()["churn_probability"]
        # High-risk profile should have elevated probability
        assert prob > 0.35

    def test_predict_low_risk_customer(self):
        resp = client.post("/predict", json=LOW_RISK_PAYLOAD)
        assert resp.status_code == 200
        prob = resp.json()["churn_probability"]
        # Low-risk profile should have lower probability
        assert prob < 0.65

    def test_predict_invalid_discount_pct(self):
        bad = {**VALID_PAYLOAD, "avg_discount_pct_180d": 1.5}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422   # Pydantic validation error

    def test_predict_negative_recency(self):
        bad = {**VALID_PAYLOAD, "recency_days": -1}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_missing_field(self):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "recency_days"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_predicted_churn_consistent_with_threshold(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        threshold = data["threshold_used"]
        assert data["predicted_churn"] == (data["churn_probability"] >= threshold)


# ── Test 3: Batch predict ──────────────────────────────────────────────────────
class TestBatchPredict:
    def test_batch_single_customer(self):
        resp = client.post("/batch_predict", json={"customers": [VALID_PAYLOAD]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["results"]) == 1

    def test_batch_multiple_customers(self):
        payload = {"customers": [VALID_PAYLOAD, HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]}
        resp = client.post("/batch_predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["results"]) == 3

    def test_batch_preserves_order(self):
        payload = {"customers": [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]}
        resp = client.post("/batch_predict", json=payload)
        data = resp.json()
        results = data["results"]
        assert results[0]["index"] == 0
        assert results[1]["index"] == 1

    def test_batch_empty_list_rejected(self):
        resp = client.post("/batch_predict", json={"customers": []})
        assert resp.status_code == 422

    def test_batch_result_has_required_fields(self):
        resp = client.post("/batch_predict", json={"customers": [VALID_PAYLOAD]})
        result = resp.json()["results"][0]
        for field in ["index", "churn_probability", "predicted_churn", "risk_label", "explanation"]:
            assert field in result

    def test_batch_probabilities_in_range(self):
        payload = {"customers": [VALID_PAYLOAD, HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]}
        resp = client.post("/batch_predict", json=payload)
        for r in resp.json()["results"]:
            assert 0.0 <= r["churn_probability"] <= 1.0
