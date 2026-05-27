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
    if spec in ("Latest", "Ultima(Default)"):
        date, path = snaps[-1]
    elif spec == "Earliest":
        date, path = snaps[0]
    else:
        spec = str(spec).strip()
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
        "para gas_model y pet_model. La referencia se toma automáticamente "
        "del snapshot con que se entrenó cada modelo (fecha_de_data en MLflow). "
        "Sube métricas y artifacts al experimento `monitoring` de MLflow."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "current_snapshot": "Latest",
    },
    tags=["monitoring", "drift", "decay"],
)
def drift_and_decay_report():

    @task
    def run_for_model(model_name: str) -> dict:
        import pandas as pd

        from decay import compute_model_decay, load_production_run
        from drift import compute_classifier_drift, compute_ks_drift

        context = get_current_context()
        snaps = _list_snapshots()

        cur_spec = context["params"]["current_snapshot"]
        cur_date, cur_path = _resolve_one(cur_spec, snaps, "current")
        df_cur = pd.read_parquet(cur_path)

        _run, features, target, _mv, fecha_de_data = load_production_run(model_name)
        ref_date, ref_path = _resolve_one(fecha_de_data, snaps, "reference")
        df_ref = pd.read_parquet(ref_path)

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
            "ref_date": ref_date,
            "cur_date": cur_date,
        }

    @task
    def publish(gas_results: dict, pet_results: dict) -> str:
        import mlflow
        import pandas as pd

        from report import generate_report

        snaps = _list_snapshots()
        _, gas_ref_path = _resolve_one(gas_results["ref_date"], snaps, "reference")
        _, pet_ref_path = _resolve_one(pet_results["ref_date"], snaps, "reference")
        _, cur_path = _resolve_one(gas_results["cur_date"], snaps, "current")

        df_ref_gas = pd.read_parquet(gas_ref_path)
        df_ref_pet = pd.read_parquet(pet_ref_path)
        df_cur = pd.read_parquet(cur_path)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            report_path = generate_report(
                df_ref_gas=df_ref_gas,
                df_ref_pet=df_ref_pet,
                df_cur=df_cur,
                gas_results=gas_results,
                pet_results=pet_results,
                cur_snapshot=gas_results["cur_date"],
                output_dir=out_dir,
            )

            mlflow.set_experiment("monitoring")
            run_name = (
                f"drift_gas{gas_results['ref_date']}"
                f"_pet{pet_results['ref_date']}"
                f"_vs_{gas_results['cur_date']}"
            )
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("current_snapshot", gas_results["cur_date"])
                mlflow.log_param("gas_ref_snapshot", gas_results["ref_date"])
                mlflow.log_param("pet_ref_snapshot", pet_results["ref_date"])

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

    gas = run_for_model.override(task_id="run_gas")("gas_model")
    pet = run_for_model.override(task_id="run_pet")("pet_model")
    publish(gas, pet)


drift_and_decay_report()
