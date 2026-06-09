"""
train_and_save_model.py

Run this once before starting the API:
    python train_and_save_model.py

Trains XGBoost on rfm_modeling_snapshot.csv and saves:
    model/model.pkl
    model/feature_cols.json
    model/metrics.json
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("../data/rfm_modeling_snapshot.csv")
MODEL_DIR  = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
THRESHOLD  = 0.40

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

# ── Feature engineering ───────────────────────────────────────────────────────
df["is_loyalty_member"] = df["loyalty_tier"].notna().astype(int)
df["loyalty_tier"]      = df["loyalty_tier"].fillna("None")
df["has_rated"]         = df["avg_rating_180d"].notna().astype(int)
df["avg_rating_180d"]   = df["avg_rating_180d"].fillna(df["avg_rating_180d"].median())

cat_cols = ["city_tier", "age_group", "acquisition_channel", "loyalty_tier",
            "preferred_category", "marketing_consent"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

FEATURE_COLS = [
    "city_tier", "age_group", "acquisition_channel", "loyalty_tier",
    "preferred_category", "marketing_consent", "is_loyalty_member", "has_rated",
    "recency_days", "frequency_180d", "monetary_180d", "return_rate_180d",
    "avg_discount_pct_180d", "avg_rating_180d", "category_diversity_180d",
    "ticket_count_90d", "negative_ticket_rate_90d", "avg_resolution_hours_90d",
    "days_since_signup", "sessions_30d", "product_views_30d", "cart_adds_30d",
    "wishlist_adds_30d", "abandoned_carts_30d", "email_opens_30d",
    "campaign_clicks_30d", "last_visit_days_ago",
]
TARGET = "churn_next_60d"

train = df[df["split"] == "train"]
val   = df[df["split"] == "validation"]
test  = df[df["split"] == "test"]

X_train, y_train = train[FEATURE_COLS], train[TARGET]
X_val,   y_val   = val[FEATURE_COLS],   val[TARGET]
X_test,  y_test  = test[FEATURE_COLS],  test[TARGET]

# ── Train ─────────────────────────────────────────────────────────────────────
scale_pw = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pw,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# ── Evaluate ──────────────────────────────────────────────────────────────────
test_proba = model.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= THRESHOLD).astype(int)

metrics = {
    "model": "XGBoost",
    "split": "test",
    "threshold": THRESHOLD,
    "AUC":       round(roc_auc_score(y_test, test_proba), 4),
    "F1":        round(f1_score(y_test, test_pred), 4),
    "Recall":    round(recall_score(y_test, test_pred), 4),
    "Precision": round(precision_score(y_test, test_pred), 4),
}
print("\nTest metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# ── Save artefacts ────────────────────────────────────────────────────────────
with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(MODEL_DIR / "feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODEL_DIR / 'model.pkl'}")
print(f"  {MODEL_DIR / 'feature_cols.json'}")
print(f"  {MODEL_DIR / 'metrics.json'}")
