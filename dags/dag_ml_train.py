from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
from airflow.sdk import get_current_context
import requests
from datetime import datetime
from pathlib import Path
import yaml

DATA_PATH = Path("/opt/airflow/feature_store/data/well_features.parquet")
EXP_PATH = Path("/opt/airflow/experimentos")



@dag(
    dag_id='ml_training_pipeline',
    description='Pipeline de entrenamiento de modelos con Airflow',
    start_date=datetime(2026, 1, 1),
    params={
        "experiment_name": "Airflow-MLflow",
        "fecha_data": "Ultima(Default)"
    }
)
def ml_training_pipeline():
    @task
    def get_experiments():
        context = get_current_context()
        experiment_name = context["params"]["experiment_name"]

        experiment_path = EXP_PATH / f"{experiment_name}.yaml"

        with open(experiment_path, "r") as f:
            config = yaml.safe_load(f)
            experiments = config["experiments"]
        return experiments

    @task
    def train_model(experiment):
        import pandas as pd
        import pickle
        import mlflow
        import mlflow.sklearn

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        context = get_current_context()
        experiment_name = context["params"]["experiment_name"]
        fecha_data = context["params"]["fecha_data"]

        # Config
        model_type = experiment["model_type"]
        params = experiment["model_params"]
        target = experiment["target"]
        features = experiment["features"]

        mlflow.set_experiment(experiment_name)

        #cargo data
        if fecha_data=="Ultima(Default)":
            data = pd.read_parquet(DATA_PATH)
        else:
            data_especifica = DATA_PATH.parent / f"{DATA_PATH.stem}_{fecha_data}{DATA_PATH.suffix}"
            data = pd.read_parquet(data_especifica)

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        mlflow.sklearn.autolog()
        params_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        model_name = f"{model_type}_{params_str}"

        with mlflow.start_run(run_name=model_name):
            # Modelo dinámico
            if model_type == "random_forest":
                model = RandomForestRegressor(**params)
            else:
                raise ValueError(f"Modelo no soportado: {model_type}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            # Loggear TODO el experimento
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("target", target)
            mlflow.log_param("features", features)
            mlflow.log_param("fecha_de_data", fecha_data)

            # Loggear hiperparámetros dinámicamente
            for k, v in params.items():
                mlflow.log_param(k, v)


    experiments = get_experiments()
    train_model.expand(experiment=experiments)

    

ml_training_pipeline()
