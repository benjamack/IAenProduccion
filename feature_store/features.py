from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32, Int64


OFFLINE_PARQUET_PATH = "/opt/airflow/feature_store/data/well_features.parquet"


pozo = Entity(
    name="idpozo",
    join_keys=["idpozo"],
    value_type=ValueType.INT64,
    description="Identificador único del pozo de extracción",
)


well_stats_source = FileSource(
    name="well_stats_source",
    path=OFFLINE_PARQUET_PATH,
    timestamp_field="fecha",
)


well_stats = FeatureView(
    name="well_stats",
    entities=[pozo],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="prod_gas", dtype=Float32),
        Field(name="prod_pet", dtype=Float32),
        Field(name="prod_agua", dtype=Float32),

        Field(name="prod_gas_f1", dtype=Float32),
        Field(name="prod_gas_f2", dtype=Float32),
        Field(name="prod_pet_f1", dtype=Float32),
        Field(name="prod_pet_f2", dtype=Float32),
        Field(name="prod_agua_f1", dtype=Float32),
        Field(name="prod_agua_f2", dtype=Float32),

        Field(name="tef", dtype=Float32),
        Field(name="profundidad", dtype=Float32),

        Field(name="fecha_ts", dtype=Int64),

        Field(name="tipoextraccion", dtype=Int32),
        Field(name="tipopozo", dtype=Int32),
        Field(name="provincia", dtype=Int32),
        Field(name="cuenca", dtype=Int32),

        Field(name="prod_gas_t1", dtype=Float32),
        Field(name="prod_gas_t2", dtype=Float32),
        Field(name="prod_gas_t3", dtype=Float32),
        Field(name="prod_pet_t1", dtype=Float32),
        Field(name="prod_pet_t2", dtype=Float32),
        Field(name="prod_pet_t3", dtype=Float32),
        Field(name="prod_agua_t1", dtype=Float32),
        Field(name="prod_agua_t2", dtype=Float32),
        Field(name="prod_agua_t3", dtype=Float32),

        Field(name="avg_prod_gas_10m", dtype=Float32),
        Field(name="avg_prod_pet_10m", dtype=Float32),
        Field(name="avg_prod_agua_10m", dtype=Float32),
    ],
    source=well_stats_source,
    online=True,
)
