"""FastAPI inference service for the insurance-cost model.

Endpoints
    GET  /health   liveness + whether the model artifact is loaded
    GET  /schema   form field metadata (drives the React UI)
    POST /predict  predict cost + risk category for one applicant

Run:  uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from insurance import config
from insurance.predict import load_bundle, predict
from insurance.schemas import FIELD_SPECS, Driver, InsuranceApplicant, PredictionResponse

app = FastAPI(
    title="Insurance Price Prediction API",
    description="Estimate insurance cost and risk band from applicant health/lifestyle data.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev convenience; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bundle_or_503() -> dict:
    if not config.MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run `uv run python -m insurance.train` first.",
        )
    return load_bundle()


@app.get("/health")
def health() -> dict:
    loaded = config.MODEL_PATH.exists()
    info = {"status": "ok", "model_loaded": loaded}
    if loaded:
        bundle = load_bundle()
        info["model_name"] = bundle.get("model_name")
        info["trained_at"] = bundle.get("trained_at")
    return info


@app.get("/schema")
def schema() -> dict:
    """Field metadata so the UI can render the right inputs and dropdowns."""
    return {"fields": FIELD_SPECS}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(applicant: InsuranceApplicant) -> PredictionResponse:
    bundle = _bundle_or_503()
    result = predict(applicant.to_frame(), bundle=bundle)[0]
    drivers = [Driver(feature=d["feature"], importance=round(float(d["importance_mean"]), 4))
               for d in bundle.get("feature_importance", [])]
    return PredictionResponse(
        predicted_cost=result["predicted_cost"],
        risk_category=result["risk_category"],
        top_drivers=drivers,
    )
