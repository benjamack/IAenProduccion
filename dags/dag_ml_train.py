from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
import requests
from datetime import datetime


DATA_OUTPUT_PATH = "/opt/airflow/data/data_preprocessed.csv"

NUM_FEATURES = [
    "prod_pet", "prod_agua", "iny_agua", "iny_gas",
    "iny_co2", "tef", "vida_util", "profundidad",
    "anio", "mes"
]

CAT_FEATURES = [
    "tipoextraccion", "tipopozo", "provincia", "cuenca"
]

TARGET = "prod_gas"

EXPERIMENTS = [
    {'model_type': 'random_forest', 'model_params': {'n_estimators': 50,  'random_state': 204}, 'target': 'prod_gas', 'features': NUM_FEATURES+CAT_FEATURES}
]

@dag(
    dag_id='ml_training_pipeline',
    description='Pipeline de dentrenamiento de modelos con Airflow',
    start_date=datetime(2026, 1, 1)
)

def ml_training_pipeline():
    @task
    def train_model(data_path, experiment):
        import pandas as pd
        import pickle
        import mlflow
        import mlflow.sklearn

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        # Config
        model_type = experiment["model_type"]
        params = experiment["model_params"]
        target = experiment["target"]
        features = experiment["features"]

        mlflow.set_experiment("Airflow-MLflow")

        # Load data
        data = pd.read_csv(data_path)

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        mlflow.sklearn.autolog()

        with mlflow.start_run():
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
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("target", target)
            mlflow.log_param("n_features", len(features))

            # Loggear hiperparámetros dinámicamente
            for k, v in params.items():
                mlflow.log_param(k, v)
  
    for i, exp in enumerate(EXPERIMENTS):
        train_model(
            data_path=DATA_OUTPUT_PATH,
            experiment=exp
        )

ml_training_pipeline()
