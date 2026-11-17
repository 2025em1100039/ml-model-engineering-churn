
from pathlib import Path
import numbers

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

# Resolve artifact path independently of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts" / "churn_pipeline.joblib"

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.1.0"
)

# Fail safely if the artifact is missing or cannot be loaded.
bundle = None
model_load_error = None

try:
    if ART.exists():
        bundle = joblib.load(ART)
    else:
        model_load_error = "Model artifact not found"
except Exception:
    model_load_error = "Model artifact could not be loaded"


class PredictRequest(BaseModel):
    features: dict[str, Any]


@app.get("/health")
def health():
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable. Restore the artifact or run python -m src.train"
        )

    expected = bundle["features"]

    missing = [c for c in expected if c not in req.features]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing}
        )

    row = {c: req.features[c] for c in expected}

    # Validate numeric inputs while allowing legitimate missing values.
    numeric_limits = {
        "SeniorCitizen": (0, 1),
        "tenure": (0, 72),
        "MonthlyCharges": (0, 200),
        "TotalCharges": (0, 10000)
    }

    for feature, (minimum, maximum) in numeric_limits.items():
        value = row.get(feature)

        if value is None:
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
        ):
            raise HTTPException(
                status_code=422,
                detail=f"{feature} must be numeric or null"
            )

        if not minimum <= value <= maximum:
            raise HTTPException(
                status_code=422,
                detail=f"{feature} must be between {minimum} and {maximum}"
            )

    try:
        frame = pd.DataFrame([row], columns=expected)

        prob = float(
            bundle["pipeline"].predict_proba(frame)[0, 1]
        )

        pred = int(prob >= bundle["threshold"])

        return {
            "churn_probability": prob,
            "churn_prediction": pred,
            "decision_threshold": bundle["threshold"],
            "risk_band": (
                "high" if prob >= 0.70
                else "medium" if prob >= 0.40
                else "low"
            )
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal model scoring error"
        )
