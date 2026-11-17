# Mini Production ML System: Telco Customer Churn

## Business objective
Prioritise telecom customers at risk of cancelling so retention teams can investigate and offer suitable interventions. This is a classification/decision-support prototype, not an automated cancellation or pricing decision.

## Dataset
IBM Telco Customer Churn sample: 7,043 rows and 21 raw columns. Source:
https://github.com/IBM/telco-customer-churn-on-icp4d
Direct CSV:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
The loader downloads the CSV on first run. `TotalCharges` is converted to numeric; blank values are removed (11 expected in the canonical sample), and `customerID` is excluded from predictors.

## Setup (Python 3.10+)
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
```
Training writes `artifacts/churn_pipeline.joblib`, `outputs/metrics.json`, and `outputs/test_predictions.csv`.

## Run the API
```bash
uvicorn serving.app:app --host 127.0.0.1 --port 8000
```
Health: `GET http://127.0.0.1:8000/health`
Prediction: `POST http://127.0.0.1:8000/predict` with JSON `{"features": { ...all predictor columns... }}`. The API returns probability, class, threshold and risk band. Swagger UI: `/docs`.

## Profiling
```bash
python -m serving.benchmark
```
Reports local single-row model inference p50/p95/p99 latency in milliseconds. These are model-call timings, not end-to-end network/API latency.

## Engineering decisions
- Stratified 70/15/15 train/validation/test split because this is a cross-sectional snapshot without historical monthly snapshots; no temporal split can be justified.
- All imputation, scaling and one-hot encoding are inside a Scikit-learn Pipeline fitted on training data only.
- Dummy prior classifier is the baseline; class-weighted Logistic Regression is the candidate.
- Decision threshold is selected on validation F1; test set is held out for final evaluation.
- Unknown categories are tolerated by `OneHotEncoder(handle_unknown='ignore')`; missing numeric/categorical values are imputed.
- Drift: monitor PSI on numeric features and category-frequency shifts; investigate PSI >=0.20 and conduct retraining review at >=0.25. Track prediction volume, missingness, API p95/p99, error rate and delayed outcome-based PR-AUC/recall.
- Fallback: return HTTP 503 when artifact is unavailable; malformed/missing fields receive 422. Do not silently return a fabricated risk score.

## Important
Run the training script and benchmark in your own environment. Insert the actual values from `outputs/metrics.json` and benchmark output into the assignment report. Do not report example or assumed metrics as measured results.
