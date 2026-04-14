from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from datetime import date
from typing import Optional
import mlflow
import pandas as pd
import os
from feast import FeatureStore
from pathlib import Path
from mlflow.tracking import MlflowClient
import json
import ast

models = {}
features_map = {}

app = FastAPI(
    title="Oil & Gas Forecast API",
    version="1.0.0",
    description="API para consultar el listado de pozos y sus pronósticos de producción.",
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:9090")
FEATURE_STORE_PATH = Path("/data")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ─── Response schemas ─────────────────────────────────────────────────────────

class ProductionPoint(BaseModel):
    date: str
    prod_gas: float
    prod_pet: float

class ForecastResponse(BaseModel):
    id_well: str
    data: list[ProductionPoint]

class WellInfo(BaseModel):
    id_well: str

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_models():
    global models, features_map

    client = MlflowClient()

    model_names = ["gas_model", "pet_model"] 

    for model_name in model_names:

        # Cargar modelo
        model_uri = f"models:/{model_name}@production"
        models[model_name] = mlflow.sklearn.load_model(model_uri)

        # vemos version
        model_version = client.get_model_version_by_alias(
            name=model_name,
            alias="production"
        )

        run = client.get_run(model_version.run_id)
        features_str = run.data.params["features"]
        features = ast.literal_eval(features_str)

        features_map[model_name] = features

        print(f"Modelo {model_name} cargado con {len(features_map[model_name])} features")


def get_well_features(id_well: str,features) -> pd.DataFrame:
    store = FeatureStore(repo_path=FEATURE_STORE_PATH)



    df = store.get_online_features(
        features=features,
        entity_rows=[{"idpozo": int(id_well)}],
    ).to_df()

    return df


# ─── Endpoints ────────────────────────────────────────────────────────────────

#@app.on_event("startup")
#def startup_event():
#    load_models()

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
    global models, features_map
    if date_start > date_end:
        raise HTTPException(status_code=400, detail="date_start must be before date_end")
    
    gas_features = features_map["gas_model"]
    pet_features = features_map["pet_model"]
    features = list(set(gas_features + pet_features))

    df=get_well_features(id_well,features)

     # asegurar orden correcto
    gas_input = df.reindex(columns=gas_features)
    pet_input = df.reindex(columns=pet_features)

    output_gas=models["gas_model"].predict(gas_input)
    output_pet=models["pet_model"].predict(pet_input)

    data=[]
    for i in range(3):
        data.append(ProductionPoint(date=str(i), prod_gas=output_gas[i], prod_pet=output_pet[i]))

    return ForecastResponse(id_well=id_well, data=data)


@app.get(
    "/api/v1/wells",
    response_model=list[WellInfo],
    summary="Obtiene el listado de pozos.",
    responses={
        200: {
            "description": "Listado de pozos obtenido exitosamente."
        }}
)
def get_wells(
    date_query: date = Query(..., description="Fecha para la cual se hace la consulta (YYYY-MM-DD)."),
):
    df = pd.read_parquet(FEATURE_STORE_PATH/"data/well_features.parquet")
    df = df[df["fecha"] <= pd.to_datetime(date_query)]
    wells = df["idpozo"].unique()

    return [WellInfo(id_well=str(w)) for w in wells]