from airflow.decorators import dag, task
from airflow.sdk import get_current_context
from datetime import datetime
from pathlib import Path
import yaml
import re

DATA_PATH = Path("/opt/airflow/feature_store/data/well_features.parquet")
EXP_PATH = Path("/opt/airflow/experimentos")
SNAPSHOT_RE = re.compile(r"^well_features_(\d{8})\.parquet$")

def _latest_snapshot():
    snapshots = []

    for p in DATA_PATH.parent.glob("well_features_*.parquet"):
        m = SNAPSHOT_RE.match(p.name)

        if m:
            snapshots.append(m.group(1))

    if not snapshots:
        return None

    return max(snapshots)


@dag(
    dag_id='Automatic_training_gas',
    description='Pipeline de entrenamiento automatico de modelos para prediccion de produccion de gas',
    start_date=datetime(2026, 5, 1),
    schedule="5 0 2 * *",
    is_paused_upon_creation=True,
    params={
        "experiment_name": "Experimento_gas_auto",
        "fecha_data": "Ultima(Default)",
        "decision_metric": "mse",
        "decision_logic": "ASC"
    }
)
def Automatic_training_gas():
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
        import mlflow
        import mlflow.sklearn

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        context = get_current_context()
        experiment_name = context["params"]["experiment_name"]
        fecha_data = context["params"]["fecha_data"]

        model_type = experiment["model_type"]
        params = experiment["model_params"]
        target = experiment["target"]
        features = experiment["features"]

        fecha_actual = context["ds_nodash"]
        exp_name=experiment_name+fecha_actual
        mlflow.set_experiment(exp_name)

        if fecha_data == "Ultima(Default)":
            data = pd.read_parquet(DATA_PATH)
            fecha_data=_latest_snapshot()
        else:
            data_especifica = DATA_PATH.parent / f"{DATA_PATH.stem}_{fecha_data}{DATA_PATH.suffix}"
            data = pd.read_parquet(data_especifica)

        cols = features + [target] if isinstance(target, str) else features + target
        df_model = data[cols].dropna()

        X = df_model[features]
        y = df_model[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        mlflow.sklearn.autolog()
        params_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        model_name = f"{model_type}_{params_str}"

        with mlflow.start_run(run_name=model_name):
            if model_type == "random_forest":
                model = RandomForestRegressor(**params)
            else:
                raise ValueError(f"Modelo no soportado: {model_type}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("target", target)
            mlflow.log_param("features", features)
            mlflow.log_param("fecha_de_data", fecha_data)

            for k, v in params.items():
                mlflow.log_param(k, v)

        return "listo" 


    @task
    def select_best_model():
        import mlflow
        from airflow.sdk import get_current_context
        from mlflow.tracking import MlflowClient

        context = get_current_context()
        decision_metric = context["params"]["decision_metric"]
        decision_logic = context["params"]["decision_logic"]
        experiment_name = context["params"]["experiment_name"]

        fecha_actual = context["ds_nodash"]
        exp_name=experiment_name+fecha_actual


        client = MlflowClient()

        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            raise ValueError(f"Experiment '{exp_name}' no existe")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{decision_metric} {decision_logic}"],
            max_results=1,
        )

        if not runs:
            raise ValueError("No hay runs en el experimento")

        best_run = runs[0]

        if not best_run.outputs or not best_run.outputs.model_outputs:
            raise ValueError(
                f"El run {best_run.info.run_id} no tiene logged models asociados"
            )

        model_id = best_run.outputs.model_outputs[0].model_id
        return {"model_id": model_id}


    @task
    def register_and_promote(model_info):
        from airflow.sdk import get_current_context
        from mlflow.tracking import MlflowClient

        model_name = "gas_model"

        client = MlflowClient()

        model_id = model_info["model_id"]
        model_uri = f"models:/{model_id}"

        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=mv.version,
        )

    experiments = get_experiments()
    trained_models=train_model.expand(experiment=experiments)
    best_model = select_best_model()
    trained_models >> best_model
    register_and_promote(best_model)


Automatic_training_gas()
