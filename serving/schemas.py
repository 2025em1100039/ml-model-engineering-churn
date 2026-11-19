"""Pydantic contract definitions for the Telco Churn API.

The current serving.app module retains its existing inline request model.
This module documents the same request/response contract in a separate
location, as suggested by the assignment repository structure.
"""
from typing import Any
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    decision_threshold: float
    risk_band: str
