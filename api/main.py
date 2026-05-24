# app.py

import ray
from ray import serve
from fastapi import FastAPI
from datetime import date
import mlflow
import pandas as pd
import ast
from feast import FeatureStore
from pathlib import Path
from mlflow.tracking import MlflowClient
from fastapi import HTTPException
import time


app = FastAPI()

MLFLOW_TRACKING_URI = "http://mlflow:9090"
FEATURE_STORE_PATH = Path("/opt/airflow/feature_store")
MONTHLY_DECAY = 0.97


class ModelService:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.last_check = 0
        self.check_interval = 60
        self.current_versions = {}
        self.load_models()

    def load_models(self):
        client = MlflowClient()
        self.models = {}
        self.features_map = {}

        for model_name in ["gas_model", "pet_model"]:
            self.models[model_name] = mlflow.sklearn.load_model(
                f"models:/{model_name}@production"
            )

            mv = client.get_model_version_by_alias(
                name=model_name,
                alias="production",
            )

            self.current_versions[model_name]=str(mv.run_id)

            run = client.get_run(mv.run_id)
            self.features_map[model_name] = ast.literal_eval(
                run.data.params["features"]
            )

    def model_version_changed(self):
        client = MlflowClient()

        for model_name in ["gas_model", "pet_model"]:

            mv = client.get_model_version_by_alias(
                name=model_name,
                alias="production",
            )

            if str(mv.run_id) != self.current_versions[model_name]:
                return True

        return False

    def maybe_reload(self):
        now = time.time()

        if now - self.last_check < self.check_interval:
            return

        self.last_check = now

        if self.model_version_changed():
            self.load_models()

    def forecast(self, id_well, date_start, date_end):
        self.maybe_reload()
        if date_start > date_end:
            raise HTTPException(status_code=400, detail="date_start must be before date_end")
        
        gas_features = self.features_map["gas_model"]
        pet_features = self.features_map["pet_model"]

        all_features = list(set(gas_features + pet_features + ["fecha_ts"]))
        print(all_features)
        store = FeatureStore(repo_path=str(FEATURE_STORE_PATH))
        df = store.get_online_features(
            features=[f"well_stats:{f}" for f in all_features],
            entity_rows=[{"idpozo": int(id_well)}],
        ).to_df()
        print(df)

        fecha_ts = df["fecha_ts"].iloc[0]
        print(fecha_ts)
        
        if fecha_ts is None or pd.isna(fecha_ts) or df.empty:
            raise HTTPException(status_code=404, detail=f"No hay features online para el pozo {df['idpozo'].iloc[0]}")

        max_fecha = pd.to_datetime(fecha_ts, unit="s").to_period("M").to_timestamp()

        gas_input = df.reindex(columns=gas_features)
        pet_input = df.reindex(columns=pet_features)

        base_gas = float(self.models["gas_model"].predict(gas_input)[0])
        base_pet = float(self.models["pet_model"].predict(pet_input)[0])

        months = pd.date_range(
            start=pd.to_datetime(date_start).to_period("M").to_timestamp(),
            end=pd.to_datetime(date_end).to_period("M").to_timestamp(),
            freq="MS",
        )

        data = []
        for m in months:
            if m < max_fecha:
                continue

            months_ahead = (m.year - max_fecha.year) * 12 + (m.month - max_fecha.month)
            decay = MONTHLY_DECAY ** months_ahead

            data.append({
                "date": m.strftime("%Y-%m-%d"),
                "prod_gas": base_gas * decay,
                "prod_pet": base_pet * decay,
            })

        return {"id_well": id_well, "data": data}

    def get_wells(self, date_query):
        df = pd.read_parquet(FEATURE_STORE_PATH / "data" / "well_features.parquet")
        fecha_ref = pd.to_datetime(date_query) - pd.DateOffset(months=1)
        df = df[df["fecha"] > fecha_ref]
        return [{"id_well": str(w)} for w in df["idpozo"].unique()]

    def reload(self):
        self.load_models()


@serve.deployment(num_replicas=3)
@serve.ingress(app)
class API:
    def __init__(self):
        self.svc = ModelService()

    @app.get("/health")
    async def health(self):
        return {"status": "ok"}

    @app.get("/api/v1/forecast")
    def forecast(self, id_well: str, date_start: date, date_end: date):
        return self.svc.forecast(id_well, date_start, date_end)

    @app.get("/api/v1/wells")
    def wells(self, date_query: date):
        return self.svc.get_wells(date_query)

    @app.post("/reload-model")
    def reload(self):
        self.svc.reload()
        return {"status": "reloaded"}


if __name__ == "__main__":
    ray.init(
        include_dashboard=False,
        _node_ip_address="0.0.0.0",
    )

    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 8000,
        }
    )

    serve.run(API.bind())

    
    while True:
        time.sleep(3600)