"""Definiciones de entidades y feature views de Feast para el proyecto
oil_gas_production.

Sigue el patrón de la Clase 3 (IA en Producción, UdeSA 2026):
- Entidad `idpozo` como join key.
- FeatureView `well_stats` con features instantáneos + features de ventana
  calculados sobre las últimas 10 lecturas de cada pozo.
- Offline store: archivo Parquet (`well_features.parquet`).
- Online store: SQLite (configurado en `feature_store.yaml`).

Benja — branch feature_store
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32, Int64


# Paths — relativos al repo de Feast (feature_store.yaml está al lado de este
# archivo), pero cuando el DAG corre dentro de Airflow el working dir del
# store será /opt/airflow/feature_store, por lo que usamos un path absoluto
# coherente con el resto del proyecto.
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
        # Numéricos instantáneos (última lectura de cada pozo)
        Field(name="prod_gas", dtype=Float32),
        Field(name="prod_pet", dtype=Float32),
        Field(name="prod_agua", dtype=Float32),
        Field(name="tef", dtype=Float32),
        Field(name="profundidad", dtype=Float32),
        # Categórico encoded con LabelEncoder (ver populate_store.py)
        Field(name="tipoextraccion", dtype=Int32),
        # Features de ventana: últimas 10 lecturas por pozo
        Field(name="avg_prod_gas_10m", dtype=Float32),
        Field(name="avg_prod_pet_10m", dtype=Float32),
        Field(name="last_prod_gas", dtype=Float32),
        Field(name="last_prod_pet", dtype=Float32),
        Field(name="n_readings", dtype=Int32),
    ],
    source=well_stats_source,
    online=True,
)
