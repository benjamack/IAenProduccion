# Oil & Gas Forecast — IA en Producción

Pipeline MLOps para pronóstico de producción de pozos de gas y petróleo no convencional.
Dataset: Secretaría de Energía Argentina (datos.gob.ar).

## Stack

| Componente | Herramienta |
|-----------|-------------|
| Orquestación | Apache Airflow 3.0.6 (CeleryExecutor) |
| Experiment tracking | MLflow 3.10.1 |
| Feature store | Feast 0.54 (SQLite online, Parquet offline) |
| API | FastAPI |
| Containerización | Docker Compose |

## Levantar el sistema

```bash
cp .env.example .env
# Ajustar AIRFLOW_UID si hace falta (Linux/Mac: id -u)
docker compose up --build
```

El archivo `.env` es obligatorio — el compose lo requiere. Si no existe, `docker compose` falla con `env file ... not found`.

| Servicio | URL |
|---------|-----|
| Airflow UI | http://localhost:8181 (airflow/airflow) |
| MLflow UI | http://localhost:9191 |
| API | http://localhost:8000 (Franco — pendiente) |

## Feature Store (Feast)

El feature store sigue el patrón de la Clase 3 (IA en Producción, UdeSA 2026):

- **Entidad**: `idpozo`
- **Feature View**: `well_stats` — una fila por pozo, con features instantáneos (última lectura) + features de ventana sobre las últimas 10 lecturas (`avg_prod_gas_10m`, `avg_prod_pet_10m`, `last_prod_gas`, `last_prod_pet`, `n_readings`).
- **Offline store**: Parquet en `feature_store/data/well_features.parquet`.
- **Online store**: SQLite en `feature_store/online_store.db`.
- **Registry**: `feature_store/registry.db`.

### Build del feature store

Desde la UI de Airflow (`http://localhost:8181`), disparar el DAG `build_feature_store`. Alternativa vía API REST:

```bash
curl -X POST http://localhost:8181/api/v2/dags/build_feature_store/dagRuns \
  -u airflow:airflow \
  -H "Content-Type: application/json" \
  -d '{"conf": {}}'
```

El DAG ejecuta en orden:

1. `download_raw_data` — baja el CSV crudo desde datos.gob.ar.
2. `build_offline_store` — calcula features de ventana y escribe `well_features.parquet`.
3. `feast_apply_task` — corre `feast apply` dentro del repo para generar el `registry.db`.
4. `populate_online_store_task` — escribe la última fila por pozo en el online store SQLite.

### Verificar el online store

```python
from feast import FeatureStore
store = FeatureStore(repo_path="/opt/airflow/feature_store")
print(store.get_online_features(
    features=["well_stats:avg_prod_gas_10m", "well_stats:n_readings"],
    entity_rows=[{"idpozo": 132879}],
).to_dict())
```

## Estructura del proyecto

```
├── dags/                       # Airflow DAGs
│   ├── build_feature_store.py  # Build del feature store (Benja)
│   ├── dag_data.py             # Data pipeline legacy (Franco)
│   └── dag_ml_train.py         # Training pipeline skeleton (Franco)
├── feature_store/              # Feast — entidades, feature views, populate
│   ├── feature_store.yaml
│   ├── features.py
│   └── populate_store.py
├── api/                        # FastAPI — endpoints de inferencia (pendiente)
├── config/                     # airflow.cfg
├── data/                       # Dataset descargado (gitignored)
├── mlruns/                     # Artifacts de MLflow (gitignored)
└── docker-compose.yaml
```

## Equipo

- Benja Mackinnon
- Franco Moscato
