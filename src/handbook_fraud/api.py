from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .inference import load_bundle, predict_record

app = FastAPI(title="Fraud Detection Handbook ML API", version="0.1.0")


class TransactionFeatures(BaseModel):
    TX_AMOUNT: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    customer_previous_count: int = Field(ge=0)
    customer_mean_amount_previous: float = Field(ge=0)
    customer_seconds_since_previous: float = Field(ge=-1)
    terminal_previous_count: int = Field(ge=0)
    terminal_seconds_since_previous: float = Field(ge=-1)


@lru_cache(maxsize=1)
def model_bundle() -> dict:
    return load_bundle()


@app.get("/")
def root() -> dict:
    return {"service": "Fraud Detection Handbook ML API", "docs": "/docs"}


@app.post("/predict")
def predict(transaction: TransactionFeatures) -> dict:
    return predict_record(transaction.model_dump(), model_bundle())
