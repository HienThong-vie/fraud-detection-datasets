from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .inference import load_bundle, predict_record

app = FastAPI(title="PaySim Fraud ML API", version="0.1.0")


class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: str
    amount: float = Field(ge=0)


@lru_cache(maxsize=1)
def model_bundle() -> dict:
    return load_bundle()


@app.get("/")
def root() -> dict:
    return {"service": "PaySim Fraud ML API", "docs": "/docs"}


@app.post("/predict")
def predict(transaction: Transaction) -> dict:
    return predict_record(transaction.model_dump(), model_bundle())
