"""Definiciones de entidades y feature views de Feast para el proyecto
oil_gas_production.

Sigue el patrón de la Clase 3 (IA en Producción, UdeSA 2026):
- Entidad `idpozo` como join key.
- FeatureView `well_stats` con lag features + rolling averages + categóricas.
  Una fila por pozo por mes (offline), última fila por pozo (online).
- Offline store: archivo Parquet (`well_features.parquet`).
- Online store: SQLite (configurado en `feature_store.yaml`).
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32, Int64


OFFLINE_PARQUET_PATH = "/opt/airflow/feature_store/data/well_features.parquet"


# ---- Entidad ---------------------------------------------------------------
pozo = Entity(
    name="idpozo",
    join_keys=["idpozo"],
    value_type=ValueType.INT64,
    description="Identificador único del pozo de extracción",
)


# ---- Fuente del offline store ---------------------------------------------
well_stats_source = FileSource(
    name="well_stats_source",
    path=OFFLINE_PARQUET_PATH,
    timestamp_field="fecha",
)


# ---- Feature View ---------------------------------------------------------
well_stats = FeatureView(
    name="well_stats",
    entities=[pozo],
    ttl=timedelta(days=365 * 5),
    schema=[
        # --- Targets (mes actual — para training, NO usar como features) ---
        Field(name="prod_gas", dtype=Float32),
        Field(name="prod_pet", dtype=Float32),
        Field(name="prod_agua", dtype=Float32),

        # --- Forward targets (T+1, T+2 — para predicción multi-step) ---
        Field(name="prod_gas_f1", dtype=Float32),
        Field(name="prod_gas_f2", dtype=Float32),
        Field(name="prod_pet_f1", dtype=Float32),
        Field(name="prod_pet_f2", dtype=Float32),
        Field(name="prod_agua_f1", dtype=Float32),
        Field(name="prod_agua_f2", dtype=Float32),

        # --- Estáticas ---
        Field(name="tef", dtype=Float32),
        Field(name="profundidad", dtype=Float32),

        # --- Categóricas (encoded con LabelEncoder) ---
        Field(name="tipoextraccion", dtype=Int32),
        Field(name="tipopozo", dtype=Int32),
        Field(name="provincia", dtype=Int32),
        Field(name="cuenca", dtype=Int32),

        # --- Lag features: producción en T-1, T-2, T-3 ---
        Field(name="prod_gas_t1", dtype=Float32),
        Field(name="prod_gas_t2", dtype=Float32),
        Field(name="prod_gas_t3", dtype=Float32),
        Field(name="prod_pet_t1", dtype=Float32),
        Field(name="prod_pet_t2", dtype=Float32),
        Field(name="prod_pet_t3", dtype=Float32),
        Field(name="prod_agua_t1", dtype=Float32),
        Field(name="prod_agua_t2", dtype=Float32),
        Field(name="prod_agua_t3", dtype=Float32),

        # --- Rolling averages: últimos 10 meses ---
        Field(name="avg_prod_gas_10m", dtype=Float32),
        Field(name="avg_prod_pet_10m", dtype=Float32),
        Field(name="avg_prod_agua_10m", dtype=Float32),
    ],
    source=well_stats_source,
    online=True,
)
