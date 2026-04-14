from __future__ import annotations

import sys
from datetime import datetime

from airflow.decorators import dag, task


FEATURE_STORE_PATH = "/opt/airflow/feature_store"
if FEATURE_STORE_PATH not in sys.path:
    sys.path.insert(0, FEATURE_STORE_PATH)


@dag(
    dag_id="build_feature_store",
    description=(
        "Construye el offline y online store de Feast para oil_gas_production. "
        "Descarga el dataset crudo, genera features de ventana por pozo, "
        "corre `feast apply` y pobla el online store."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["feature_store", "feast", "oil_gas_production"],
)
def build_feature_store():
    @task
    def download_raw_data() -> str:
        from populate_store import download_data

        return str(download_data())

    @task
    def build_offline_store(csv_path: str) -> str:
        from populate_store import prepare_offline_store

        return str(prepare_offline_store())

    @task
    def feast_apply_task(parquet_path: str) -> str:
        from populate_store import feast_apply

        feast_apply()
        return parquet_path

    @task
    def populate_online_store_task(parquet_path: str) -> None:
        from populate_store import populate_online_store

        populate_online_store()

    csv = download_raw_data()
    parquet = build_offline_store(csv)
    parquet_after_apply = feast_apply_task(parquet)
    populate_online_store_task(parquet_after_apply)


build_feature_store()
