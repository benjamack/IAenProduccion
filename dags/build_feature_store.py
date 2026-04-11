"""DAG: build_feature_store.

Construye el feature store de Feast para el proyecto `oil_gas_production`:

    download_raw_data
          ↓
    build_offline_store
          ↓
    feast_apply
          ↓
    populate_online_store

Cada task es fina y reejecutable de forma independiente desde la UI de
Airflow. La lógica real vive en `feature_store/populate_store.py` para
que el mismo código pueda correrse también manualmente con
`python populate_store.py` durante debugging.

Benja — branch feature_store (scope del plan de Benja para la entrega
parcial del 16/4, materia IA en Producción — UdeSA 2026).
"""

from __future__ import annotations

import sys
from datetime import datetime

from airflow.decorators import dag, task


# Aseguramos que el módulo del feature store sea importable desde el
# contenedor de Airflow. El volumen se monta en /opt/airflow/feature_store
# (ver docker-compose.yaml).
FEATURE_STORE_PATH = "/opt/airflow/feature_store"
if FEATURE_STORE_PATH not in sys.path:
    sys.path.insert(0, FEATURE_STORE_PATH)


@dag(
    dag_id="build_feature_store",
    description=(
        "Construye el offline y online store de Feast para oil_gas_production. "
        "Descarga el dataset crudo, genera features de ventana por pozo, "
        "corre `feast apply` y pobla el online store SQLite."
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
