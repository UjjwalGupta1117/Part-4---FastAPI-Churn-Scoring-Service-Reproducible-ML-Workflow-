# Part 4 — D2C Churn Scoring Service

FastAPI scoring API for the churn prediction model.

---

## Folder Structure

```
part4/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI app — /health, /predict, /batch_predict
├── model/                   # Created by train_and_save_model.py (gitignored if large)
│   ├── model.pkl
│   ├── feature_cols.json
│   └── metrics.json
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── train_and_save_model.py  # Run once to train and save the model
├── monitoring_plan.md
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and save the model (requires ../data/rfm_modeling_snapshot.csv)
python train_and_save_model.py

# 3. Start the API
uvicorn app.main:app --reload

# 4. Run tests (in a separate terminal while the server is running)
pytest tests/ -v
```

---

## Endpoints

### `GET /health`
Returns service status and confirms the model is loaded.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "feature_count": 27,
  "threshold": 0.4
}
```

---

### `POST /predict`
Score a single customer.

**Request body** — all 27 model features (see `CustomerFeatures` schema in `app/main.py`).

**Sample request**
```json
{
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
  "last_visit_days_ago": 4
}
```

**Sample response**
```json
{
  "churn_probability": 0.3124,
  "predicted_churn": false,
  "risk_label": "LOW",
  "explanation": "Churn risk drivers: moderate risk based on combined purchase and engagement signals.",
  "threshold_used": 0.4
}
```

---

### `POST /batch_predict`
Score up to 500 customers in one request.

**Request body**
```json
{
  "customers": [
    { ...customer_1_features... },
    { ...customer_2_features... }
  ]
}
```

**Sample response**
```json
{
  "count": 2,
  "threshold": 0.4,
  "results": [
    {
      "index": 0,
      "churn_probability": 0.3124,
      "predicted_churn": false,
      "risk_label": "LOW",
      "explanation": "..."
    },
    {
      "index": 1,
      "churn_probability": 0.7891,
      "predicted_churn": true,
      "risk_label": "HIGH",
      "explanation": "Churn risk drivers: last order was 250 days ago; no app/web sessions in the last 30 days."
    }
  ]
}
```

---

## Risk Label Thresholds

| Label | Churn Probability | Recommended Action |
|---|---|---|
| LOW | < 0.40 | No immediate action needed |
| MEDIUM | 0.40 – 0.69 | Include in standard retention campaign |
| HIGH | ≥ 0.70 | Priority outreach — personal CRM contact |

---

## Categorical Encoding Reference

The API expects label-encoded integers matching the training encoding:

| Feature | Encoding |
|---|---|
| `city_tier` | 0=Tier 1, 1=Tier 2, 2=Tier 3 |
| `age_group` | 0=18-24, 1=25-34, 2=35-44, 3=45+ |
| `acquisition_channel` | 0=Google Search, 1=Influencer, 2=Instagram, 3=Marketplace, 4=Organic, 5=Referral |
| `loyalty_tier` | 0=Gold, 1=None, 2=Platinum, 3=Silver |
| `preferred_category` | 0=Baby Care, 1=Fragrance, 2=Hair Care, 3=Makeup, 4=Skin Care, 5=Wellness |
| `marketing_consent` | 0=No, 1=Yes |

> **Note**: Exact label encoding order depends on the alphabetical sort applied by `sklearn.LabelEncoder`.  
> Run `train_and_save_model.py` with `verbose=True` to print the exact mappings for your dataset.

---

## Interactive Docs

Once the server is running, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc:       `http://127.0.0.1:8000/redoc`
