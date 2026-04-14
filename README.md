# Oil & Gas Forecast — IA en Producción

Pipeline MLOps end-to-end para pronóstico de producción de pozos de gas y petróleo no convencional. Dataset: Secretaría de Energía Argentina (datos.gob.ar).

Trabajo final de la materia **IA en Producción** (Maestría en IA, UdeSA 2026). Equipo: Benja Mackinnon + Franco Moscato.

## Stack

| Componente | Herramienta |
|-----------|-------------|
| Orquestación | Apache Airflow 3.0.6 (CeleryExecutor) |
| Experiment tracking + Model Registry | MLflow 3.10.1 |
| Feature store | Feast 0.54 (Redis online, Parquet offline) |
| API de inferencia | FastAPI |
| Containerización | Docker Compose |

## Levantar el sistema

```bash
cp .env.example .env
docker compose up --build
```

| Servicio | URL |
|---------|-----|
| Airflow UI | http://localhost:8181 (airflow/airflow) |
| MLflow UI | http://localhost:9191 |
| API | http://localhost:8000 (docs en `/docs`) |

## Pipeline end-to-end

### 1. Feature Store (`build_feature_store`)

- **Entidad**: `idpozo`
- **FeatureView**: `well_stats` — una fila por pozo por mes con targets, lag features (T-1, T-2, T-3), rolling averages (10m), categóricas encoded con `LabelEncoder`, y forward targets (T+1, T+2) para training multi-step.
- **Inference row**: por cada pozo se agrega una fila T+1 adicional con los lags calculados desde la última observación real. Es lo que sirve el online store.
- **Offline**: `feature_store/data/well_features.parquet`.
- **Online**: Redis (`redis:6379`).

Disparar desde Airflow UI o:

```bash
curl -X POST http://localhost:8181/api/v2/dags/build_feature_store/dagRuns \
  -u airflow:airflow \
  -H "Content-Type: application/json" \
  -d '{"conf": {}}'
```

Tasks: `download_raw_data` → `build_offline_store` → `feast_apply_task` → `populate_online_store_task`.

### 2. Training (`ml_training_pipeline`)

Entrena múltiples experimentos definidos en YAML (`experimentos/*.yaml`). Cada YAML declara el `model_type`, `model_params`, `target` y `features` por experimento. Se loggea a MLflow con `autolog` y se crean los logged models.

Parámetros del DAG:
- `experiment_name`: nombre del archivo YAML sin extensión (ej: `Experimento_1`).
- `fecha_data`: `Ultima(Default)` usa `well_features.parquet`; cualquier otro valor busca `well_features_<fecha>.parquet`.

### 3. Selección y promoción

Dos DAGs:

- **`automatic_model_selection`**: elige el mejor run de un experimento según `decision_metric` + `decision_logic` (`ASC`/`DESC`), lo registra como versión del `registered_model_name` y setea alias `production`. Usa `run.outputs.model_outputs[0].model_id` (API de MLflow 3 "logged models").
- **`model_manual_migration`**: mismo flujo pero recibiendo el `model_id` explícito como param.

### 4. API de inferencia

FastAPI levanta al arranque cargando `gas_model@production` y `pet_model@production` desde MLflow, con sus listas de features guardadas como params del run.

Endpoints:

| Método | Path | Descripción |
|--------|------|-------------|
| GET | `/health` | Healthcheck |
| GET | `/api/v1/wells?date_query=YYYY-MM-DD` | Lista pozos con actividad en el mes previo |
| GET | `/api/v1/forecast?id_well=...&date_start=...&date_end=...` | Pronóstico mensual de `prod_gas` y `prod_pet` en el rango |
| POST | `/reload-model` | Refresca los modelos sin reiniciar el container |

El `/forecast` lee el `fecha_ts` del online store como `max_fecha` del pozo, predice el mes base con el modelo y proyecta el resto del rango aplicando un factor de decaimiento mensual de `0.97^n`. Meses anteriores al `max_fecha` se ignoran.

## Estructura

```
├── api/                             FastAPI (Dockerfile propio)
├── config/                          airflow.cfg
├── dags/
│   ├── build_feature_store.py       Construcción del feature store
│   ├── dag_data.py                  Data pipeline legacy
│   ├── dag_ml_train.py              Training multi-experimento
│   ├── dag_selection.py             Selección automática del mejor modelo
│   └── dag_manual_migration.py      Promoción manual por model_id
├── experimentos/                    Experimentos YAML (model_type + params + features + target)
├── feature_store/                   Repo de Feast
│   ├── feature_store.yaml
│   ├── features.py
│   └── populate_store.py
├── mlruns/                          Artifacts de MLflow (gitignored)
├── data/                            Dataset descargado (gitignored)
└── docker-compose.yaml
```

## Equipo

- Benja Mackinnon
- Franco Moscato
