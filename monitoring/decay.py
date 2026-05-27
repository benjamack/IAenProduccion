from __future__ import annotations

import ast

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score


def load_production_run(model_name: str):
    """Devuelve (run, features, target) del modelo en producción.

    Replica el patrón de api/main.py: encuentra la versión con alias
    `production`, recupera el run y parsea los params `features` y `target`.
    """
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=model_name, alias="production")
    run = client.get_run(mv.run_id)
    features = ast.literal_eval(run.data.params["features"])
    target = run.data.params["target"]
    fecha_de_data = run.data.params.get("fecha_de_data", "Ultima(Default)")
    return run, features, target, mv, fecha_de_data


def compute_model_decay(model_name: str, df_current: pd.DataFrame) -> dict:
    """Re-evalúa el modelo productivo sobre df_current y compara contra training."""
    run, features, target, _mv, _fecha = load_production_run(model_name)

    # Las inference_rows del feature store tienen target=None: las descartamos.
    cols = features + [target]
    df_eval = df_current[cols].dropna()

    model = mlflow.sklearn.load_model(f"models:/{model_name}@production")
    y_true = df_eval[target].values
    y_pred = model.predict(df_eval[features])

    mse_current = float(mean_squared_error(y_true, y_pred))
    r2_current = float(r2_score(y_true, y_pred))

    mse_train = float(run.data.metrics.get("mse", float("nan")))
    r2_train = float(run.data.metrics.get("r2", float("nan")))

    return {
        "model_name": model_name,
        "target": target,
        "n_samples": int(len(df_eval)),
        "mse_current": mse_current,
        "r2_current": r2_current,
        "mse_train": mse_train,
        "r2_train": r2_train,
        "mse_delta": mse_current - mse_train,
        "r2_delta": r2_current - r2_train,
    }
