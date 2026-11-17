from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any

ART=Path("artifacts/churn_pipeline.joblib")
app=FastAPI(title="Telco Churn Prediction API",version="1.0.0")
bundle=joblib.load(ART) if ART.exists() else None

class PredictRequest(BaseModel):
    features: dict[str, Any]

@app.get("/health")
def health():
    return {"status":"ok" if bundle else "model_not_loaded"}

@app.post("/predict")
def predict(req: PredictRequest):
    if bundle is None:
        raise HTTPException(status_code=503,detail="Model artifact unavailable. Run python -m src.train")
    expected=bundle["features"]
    missing=[c for c in expected if c not in req.features]
    if missing:
        raise HTTPException(status_code=422,detail={"missing_features":missing})
    row={c:req.features[c] for c in expected}
    try:
        frame=pd.DataFrame([row],columns=expected)
        prob=float(bundle["pipeline"].predict_proba(frame)[0,1])
        pred=int(prob>=bundle["threshold"])
        return {"churn_probability":prob,"churn_prediction":pred,
                "decision_threshold":bundle["threshold"],
                "risk_band":"high" if prob>=0.70 else ("medium" if prob>=0.40 else "low")}
    except Exception as e:
        raise HTTPException(status_code=422,detail=f"Invalid feature values: {e}")
