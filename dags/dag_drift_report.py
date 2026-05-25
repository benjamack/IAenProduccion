from __future__ import annotations

import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.sdk import get_current_context


FEATURE_STORE_DATA = Path("/opt/airflow/feature_store/data")
MONITORING_PATH = "/opt/airflow/monitoring"
if MONITORING_PATH not in sys.path:
    sys.path.insert(0, MONITORING_PATH)


SNAPSHOT_RE = re.compile(r"^well_features_(\d{8})\.parquet$")


def _list_snapshots() -> list[tuple[str, Path]]:
    snaps = []
    for p in FEATURE_STORE_DATA.glob("well_features_*.parquet"):
        m = SNAPSHOT_RE.match(p.name)
        if m:
            snaps.append((m.group(1), p))
    snaps.sort(key=lambda x: x[0])
    return snaps


def _resolve_one(spec: str, snaps: list[tuple[str, Path]], position: str) -> tuple[str, str]:
    if not snaps:
        raise FileNotFoundError(
            f"No hay snapshots en {FEATURE_STORE_DATA}. Corré build_feature_store primero."
        )
    if spec == "Latest":
        date, path = snaps[-1]
    elif spec == "Earliest":
        date, path = snaps[0]
    else:
        match = [s for s in snaps if s[0] == spec]
        if not match:
            raise FileNotFoundError(
                f"No existe snapshot well_features_{spec}.parquet en {FEATURE_STORE_DATA}"
            )
        date, path = match[0]
    return date, str(path)


@dag(
    dag_id="drift_and_decay_report",
    description=(
        "Reporta data drift (KS + Classifier) y model decay (MSE/R²) "
        "para gas_model y pet_model. Sube métricas y artifacts a MLflow "
        "bajo el experimento `monitoring`."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "reference_snapshot": "Earliest",
        "current_snapshot": "Latest",
    },
    tags=["monitoring", "drift", "decay"],
)
def drift_and_decay_report():

    @task
    def resolve_snapshots() -> dict:
        context = get_current_context()
        ref_spec = context["params"]["reference_snapshot"]
        cur_spec = context["params"]["current_snapshot"]
        snaps = _list_snapshots()
        ref_date, ref_path = _resolve_one(ref_spec, snaps, "reference")
        cur_date, cur_path = _resolve_one(cur_spec, snaps, "current")
        return {
            "ref_date": ref_date,
            "ref_path": ref_path,
            "cur_date": cur_date,
            "cur_path": cur_path,
        }

    @task
    def run_for_model(model_name: str, snapshots: dict) -> dict:
        import pandas as pd

        from decay import compute_model_decay, load_production_run
        from drift import compute_classifier_drift, compute_ks_drift

        df_ref = pd.read_parquet(snapshots["ref_path"])
        df_cur = pd.read_parquet(snapshots["cur_path"])

        _run, features, target, _mv = load_production_run(model_name)

        ks = compute_ks_drift(df_ref, df_cur, features)
        clf = compute_classifier_drift(df_ref, df_cur, features)
        decay = compute_model_decay(model_name, df_cur)

        return {
            "model_name": model_name,
            "features": features,
            "target": target,
            "ks": ks,
            "clf": clf,
            "decay": decay,
        }

    @task
    def publish(gas_results: dict, pet_results: dict, snapshots: dict) -> str:
        import mlflow
        import pandas as pd

        from report import generate_report

        df_ref = pd.read_parquet(snapshots["ref_path"])
        df_cur = pd.read_parquet(snapshots["cur_path"])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            report_path = generate_report(
                df_ref=df_ref,
                df_cur=df_cur,
                gas_results=gas_results,
                pet_results=pet_results,
                ref_snapshot=snapshots["ref_date"],
                cur_snapshot=snapshots["cur_date"],
                output_dir=out_dir,
            )

            mlflow.set_experiment("monitoring")
            run_name = f"drift_{snapshots['ref_date']}_vs_{snapshots['cur_date']}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("reference_snapshot", snapshots["ref_date"])
                mlflow.log_param("current_snapshot", snapshots["cur_date"])

                for label, res in [("gas", gas_results), ("pet", pet_results)]:
                    mlflow.log_param(f"{label}_target", res["target"])
                    mlflow.log_metric(f"{label}_ks_n_features_drifted", res["ks"]["n_drifted"])
                    mlflow.log_metric(f"{label}_ks_min_pvalue", res["ks"]["min_p_value"])
                    mlflow.log_metric(f"{label}_classifier_auc", res["clf"]["auc"])
                    mlflow.log_metric(f"{label}_classifier_p_value", res["clf"]["p_value"])
                    mlflow.log_metric(f"{label}_classifier_drift", int(res["clf"]["is_drift"]))
                    mlflow.log_metric(f"{label}_mse_current", res["decay"]["mse_current"])
                    mlflow.log_metric(f"{label}_r2_current", res["decay"]["r2_current"])
                    mlflow.log_metric(f"{label}_mse_train", res["decay"]["mse_train"])
                    mlflow.log_metric(f"{label}_r2_train", res["decay"]["r2_train"])
                    mlflow.log_metric(f"{label}_mse_delta", res["decay"]["mse_delta"])
                    mlflow.log_metric(f"{label}_r2_delta", res["decay"]["r2_delta"])
                    mlflow.log_metric(f"{label}_decay_n_samples", res["decay"]["n_samples"])

                mlflow.log_artifacts(str(out_dir))

                return f"mlflow run {run.info.run_id}: {report_path.name}"

    snapshots = resolve_snapshots()
    gas = run_for_model.override(task_id="run_gas")("gas_model", snapshots)
    pet = run_for_model.override(task_id="run_pet")("pet_model", snapshots)
    publish(gas, pet, snapshots)


drift_and_decay_report()
