from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from datetime import date
from typing import Optional
import mlflow
import pandas as pd
import os

app = FastAPI(
    title="Oil & Gas Forecast API",
    version="1.0.0",
    description="API para consultar el listado de pozos y sus pronósticos de producción.",
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "oil_gas_forecast")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ─── Response schemas ─────────────────────────────────────────────────────────

class ProductionPoint(BaseModel):
    date: str
    prod: float

class ForecastResponse(BaseModel):
    id_well: str
    data: list[ProductionPoint]

class WellInfo(BaseModel):
    id_well: str

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_model():
    """Load the Production model from MLflow registry."""
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.sklearn.load_model(model_uri)


def get_well_features(id_well: str) -> pd.DataFrame:
    """Retrieve features for a well from the Feast online store."""
    # TODO: integrate with Feast online store
    # from feast import FeatureStore
    # store = FeatureStore(repo_path="/app/feature_store")
    # feature_vector = store.get_online_features(
    #     features=[...],
    #     entity_rows=[{"idpozo": int(id_well)}],
    # ).to_df()
    raise NotImplementedError("Feature store integration pending")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get(
    "/api/v1/forecast",
    response_model=ForecastResponse,
    summary="Obtiene el pronóstico de producción de un pozo.",
)
def get_forecast(
    id_well: str = Query(..., description="Identificador del pozo."),
    date_start: date = Query(..., description="Fecha de inicio (YYYY-MM-DD)."),
    date_end: date = Query(..., description="Fecha de fin (YYYY-MM-DD)."),
):
    if date_start > date_end:
        raise HTTPException(status_code=400, detail="date_start must be before date_end")

    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")

    # Generate monthly date range between date_start and date_end
    months = pd.date_range(start=date_start, end=date_end, freq="MS")

    # TODO: build feature rows per month from Feast and run inference
    # For now return placeholder zeros
    data = [
        ProductionPoint(date=str(m.date()), prod=0.0)
        for m in months
    ]

    return ForecastResponse(id_well=id_well, data=data)


@app.get(
    "/api/v1/wells",
    response_model=list[WellInfo],
    summary="Obtiene el listado de pozos.",
)
def get_wells(
    date_query: date = Query(..., description="Fecha para la cual se hace la consulta (YYYY-MM-DD)."),
):
    # TODO: query feature store / DB for wells active in date_query month
    # For now return empty list until data pipeline is wired up
    return []
