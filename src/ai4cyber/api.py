"""HTTP API for model inference and metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .inference import ModelService


class PredictionRequest(BaseModel):
    features: dict[str, Any]


MODEL_PATH = Path(os.getenv("AI4CYBER_MODEL_PATH", "artifacts/production_model.joblib"))
app = FastAPI(title="AI4Cyber Fraud Review API", version="0.1.0")


def _service() -> ModelService:
    try:
        return ModelService(MODEL_PATH)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/")
def root() -> dict:
    return {
        "name": app.title,
        "architecture": "ML-only",
        "model_ready": MODEL_PATH.exists(),
        "documentation": "/docs",
        "endpoints": ["GET /health", "GET /model", "POST /predict"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": MODEL_PATH.exists(), "model_path": str(MODEL_PATH)}


@app.get("/model")
def model_metadata() -> dict:
    return _service().metadata()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return _service().predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
