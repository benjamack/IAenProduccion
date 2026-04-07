# Oil & Gas Forecast — IA en Producción

Pipeline MLOps para pronóstico de producción de pozos de gas y petróleo no convencional.  
Dataset: Secretaría de Energía Argentina (datos.gob.ar).

## Stack

| Componente | Herramienta |
|-----------|-------------|
| Orquestación | Apache Airflow 2.9 |
| Experiment tracking | MLflow 2.13 |
| Feature store | Feast 0.40 |
| API | FastAPI |
| Containerización | Docker Compose |

## Levantar el sistema

```bash
cp .env.example .env
docker-compose up --build
```

| Servicio | URL |
|---------|-----|
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 (admin/admin) |

## Entrenar el modelo

```bash
airflow dags trigger train_pipeline --conf '{"run_date": "2023-01-31"}'
```

## Endpoints

```
GET /api/v1/forecast?id_well=<id>&date_start=<YYYY-MM-DD>&date_end=<YYYY-MM-DD>
GET /api/v1/wells?date_query=<YYYY-MM-DD>
GET /health
```

## Estructura del proyecto

```
├── dags/               # Airflow DAGs
├── feature_store/      # Feast — entidades, feature views, scripts de materialización
├── training/           # Lógica de entrenamiento (consumida por Airflow)
├── api/                # FastAPI — endpoints de inferencia
├── data/               # Dataset crudo (gitignored)
├── monitoring/         # Reportes de drift (Entrega 2)
└── docker-compose.yml
```

## Equipo

- Benja Mackinnon
- Franco [apellido]
