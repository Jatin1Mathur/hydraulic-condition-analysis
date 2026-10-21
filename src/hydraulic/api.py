"""FastAPI service. Thin layer: validates input, delegates to ConditionPredictor."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import N_WINDOWS, SENSORS
from .features import cycle_features
from .model import ConditionPredictor

MODEL_PATH = os.environ.get("HYDRAULIC_MODEL_PATH", "models/condition_models.joblib")

app = FastAPI(
    title="Hydraulic Condition API",
    description="Predicts cooler, valve, pump and accumulator condition from one load cycle.",
    version="1.1.0",
)


class CycleRequest(BaseModel):
    """Raw samples of one 60 s load cycle, one list per sensor."""

    signals: dict[str, list[float]] = Field(
        ..., description="All 17 sensors, e.g. {'PS1': [...], 'TS1': [...], ...}"
    )

    @field_validator("signals")
    @classmethod
    def check_sensors(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        missing = sorted(set(SENSORS) - set(v))
        if missing:
            raise ValueError(f"missing sensors: {missing}")
        unknown = sorted(set(v) - set(SENSORS))
        if unknown:
            raise ValueError(f"unknown sensors: {unknown}")
        short = [k for k, samples in v.items() if len(samples) < N_WINDOWS]
        if short:
            raise ValueError(f"sensors need at least {N_WINDOWS} samples: {short}")
        return v


class FeatureRequest(BaseModel):
    """Pre-computed cycle features, e.g. {'PS1_mean': 160.2, ...}."""

    features: dict[str, float]


class ConditionOut(BaseModel):
    label: int
    description: str
    confidence: float


class PredictionResponse(BaseModel):
    cooler: ConditionOut
    valve: ConditionOut
    pump: ConditionOut
    accumulator: ConditionOut


@lru_cache
def get_predictor() -> ConditionPredictor:
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(503, f"model not found at {MODEL_PATH}; run training first")
    return ConditionPredictor.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict/cycle", response_model=PredictionResponse)
def predict_cycle(
    req: CycleRequest, predictor: ConditionPredictor = Depends(get_predictor)
) -> dict:
    arrays = {k: np.asarray(v, dtype=np.float64)[None, :] for k, v in req.signals.items()}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise HTTPException(422, "signals contain NaN or infinite values")
    row = cycle_features(arrays).iloc[0].to_dict()
    return predictor.predict(row)


@app.post("/predict/features", response_model=PredictionResponse)
def predict_features(
    req: FeatureRequest, predictor: ConditionPredictor = Depends(get_predictor)
) -> dict:
    try:
        return predictor.predict(req.features)
    except KeyError as e:
        raise HTTPException(422, str(e.args[0])) from e
