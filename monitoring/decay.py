from __future__ import annotations

import ast

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score


def load_production_run(model_name: str):
    """Devuelve (run, features, target, mv, fecha_de_data) del modelo en producción."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=model_name, alias="production")
    run = client.get_run(mv.run_id)
    features = ast.literal_eval(run.data.params["features"])
    target = run.data.params["target"]
    fecha_de_data = run.data.params.get("fecha_de_data", "Ultima(Default)")
    return run, features, target, mv, fecha_de_data


def compute_model_decay(
    model_name: str,
    df_ref: pd.DataFrame,
    df_current: pd.DataFrame,
) -> dict:
    """Compara performance del modelo en df_ref (baseline) vs df_current.

    Ambas evaluaciones usan la misma metodología (dropna de features+target),
    así que el delta mide decay real, no diferencias de dataset.
    """
    _run, features, target, _mv, _fecha = load_production_run(model_name)

    cols = features + [target]
    model = mlflow.sklearn.load_model(f"models:/{model_name}@production")

    df_ref_eval = df_ref[cols].dropna()
    y_ref_true = df_ref_eval[target].values
    y_ref_pred = model.predict(df_ref_eval[features])
    mse_ref = float(mean_squared_error(y_ref_true, y_ref_pred))
    r2_ref = float(r2_score(y_ref_true, y_ref_pred))

    df_cur_eval = df_current[cols].dropna()
    y_cur_true = df_cur_eval[target].values
    y_cur_pred = model.predict(df_cur_eval[features])
    mse_current = float(mean_squared_error(y_cur_true, y_cur_pred))
    r2_current = float(r2_score(y_cur_true, y_cur_pred))

    return {
        "model_name": model_name,
        "target": target,
        "n_samples_ref": int(len(df_ref_eval)),
        "n_samples_current": int(len(df_cur_eval)),
        "mse_ref": mse_ref,
        "r2_ref": r2_ref,
        "mse_current": mse_current,
        "r2_current": r2_current,
        "mse_delta": mse_current - mse_ref,
        "r2_delta": r2_current - r2_ref,
    }
