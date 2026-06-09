# Monitoring Plan — D2C Churn Scoring API
Version: 1.0 | Snapshot: 2025-09-30

---

## 1. What to Monitor and Why

### 1.1 Data Drift (Input Features)
**What**: Track whether the distribution of incoming features has shifted away from the training distribution.  
**Why**: If customer behaviour changes (seasonal purchase patterns, new product launches, price changes), the model's training data becomes stale and predictions degrade silently.  
**How**: Compute the **Population Stability Index (PSI)** monthly for each feature:
- `PSI < 0.10` → stable, no action
- `0.10 ≤ PSI < 0.20` → slight drift, investigate
- `PSI ≥ 0.20` → significant drift, **trigger retraining**

Priority features to monitor (by importance):
1. `recency_days`
2. `last_visit_days_ago`
3. `sessions_30d`
4. `monetary_180d`
5. `frequency_180d`

### 1.2 Prediction Distribution
**What**: Track the distribution of `churn_probability` scores returned by the API daily and weekly.  
**Why**: A shift in the predicted churn rate (e.g., from 47% to 65%) indicates either real behaviour change or model degradation.  
**Metrics to track**:
- Mean predicted probability (rolling 7-day)
- % of customers scored HIGH / MEDIUM / LOW
- Predicted churn rate vs historical baseline (47%)

**Alert threshold**: If predicted churn rate deviates more than ±10 percentage points from baseline for 2 consecutive weeks, escalate for investigation.

### 1.3 Model Performance (Business Outcomes)
**What**: Compare model predictions against actual churn outcomes (known 60 days after scoring).  
**Why**: AUC on test data measures performance at training time. Real performance can drift.  
**Cadence**: Monthly, using a rolling 90-day evaluation window.  
**Metrics to track**:
- AUC-ROC on recent cohort
- Recall on recent cohort (missed churners = lost LTV)
- Precision on recent cohort (false alarms = wasted spend)

**Retrain trigger**: AUC drops below 0.70 OR recall drops below 0.60 on any monthly cohort.

### 1.4 API Health & Errors
**What**: Track API-level reliability and error rates.  
**Metrics**:
- Response time (p50, p95, p99) — alert if p95 > 500ms
- HTTP 4xx rate — indicates bad input from calling system; review schema
- HTTP 5xx rate — indicates model/server failure; alert immediately
- Requests per minute — unexpected spikes may indicate misuse

**Tooling**: Standard APM (Datadog, New Relic, or CloudWatch). Log all 5xx errors with full payload for debugging.

### 1.5 Fairness Monitoring
**What**: Track predicted churn rates and intervention rates across demographic slices.  
**Why**: If the model systematically scores one city tier or age group as higher risk without corresponding actual churn, the retention team will disproportionately target that group.  
**Cadence**: Quarterly.  
**Slices to monitor**: `city_tier`, `age_group`, `acquisition_channel`.  
**Alert**: If the predicted churn rate for any single slice deviates more than 15pp from the overall rate, run a fairness audit before the next campaign.

---

## 2. Retraining Triggers (any one is sufficient)

| Trigger | Threshold |
|---|---|
| AUC on rolling 90-day cohort | < 0.70 |
| Recall on rolling 90-day cohort | < 0.60 |
| PSI on any top-5 feature | ≥ 0.20 |
| Predicted churn rate deviation | ±10pp for 2 consecutive weeks |
| Time since last retrain | ≥ 3 months |
| Major product or pricing change | Immediate retrain |

---

## 3. Responsible Use Guidelines

### The API output SHOULD be used to:
- Prioritise which customers receive a retention offer (email, coupon, loyalty nudge).
- Allocate campaign budget across segments — spend more on HIGH risk, less on LOW risk.
- Flag HIGH-risk, high-LTV customers for personal CRM outreach.
- Identify patterns in the scored population to inform product and UX decisions.

### The API output MUST NOT be used to:
- **Deny service or benefits** to customers with a high churn score. A high score means we risk losing them — it is a call to action to help them, not punish them.
- **Communicate scores to customers** directly. Customers should not see their churn probability. Interventions should feel like personalised appreciation, not risk management.
- **Make automated financial decisions** (credit, pricing, refunds) based solely on the churn score.
- **Discriminate by proxy**: The model uses `age_group`, `city_tier`, and `acquisition_channel`. These must not be used as bases for differential pricing or service quality.
- **Replace human judgment** for edge cases. When the model's output conflicts with CRM knowledge about a specific customer, human override should always be possible.

### Data handling:
- Customer features sent to the API are considered PII-adjacent. All API traffic must be encrypted (HTTPS/TLS).
- Prediction logs should be retained for 90 days to support auditing, then deleted.
- Access to the API should be restricted to the retention team via API key or internal network only.

---

## 4. Suggested Dashboard (Minimum Viable)

| Panel | Metric | Alert |
|---|---|---|
| Daily scored volume | Count of /predict + /batch_predict calls | < 100/day (pipeline failure?) |
| Predicted churn rate | 7-day rolling mean probability | ±10pp from 0.47 baseline |
| Risk distribution | % HIGH / MEDIUM / LOW | >30% HIGH for 3 consecutive days |
| Model AUC (monthly) | Rolling 90-day AUC | < 0.70 |
| API error rate | 5xx / total requests | > 1% |
| Feature drift (top 5) | PSI monthly | Any PSI ≥ 0.20 |
