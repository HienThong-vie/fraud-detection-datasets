from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import ModelService

app = FastAPI(title="IEEE-CIS Fraud Screening API", version="0.1.0")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(description="Selected IEEE-CIS transaction and identity fields")


@lru_cache
def service() -> ModelService:
    return ModelService(os.getenv("IEEE_MODEL_PATH", "artifacts/production_model.joblib"))


@app.get("/")
def root() -> dict:
    return {"message": "IEEE-CIS fraud screening API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    try:
        loaded = service()
        return {"status": "ok", "model": loaded.winner}
    except FileNotFoundError:
        return {"status": "model_not_trained"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return service().predict(request.features)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Train the model before prediction") from error

