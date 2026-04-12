"""Construcción del offline + online store de Feast para oil_gas_production.

Este módulo implementa la lógica que el DAG `build_feature_store` ejecuta:

    1. Descarga el CSV crudo de datos.gob.ar.
    2. Construye `well_features.parquet` (offline store) con una fila por pozo
       que incluye features instantáneos + features de ventana calculados
       sobre las últimas 10 lecturas del pozo.
    3. Hace `feast apply` para crear/actualizar el registry.
    4. Escribe la última fila por pozo en el online store SQLite vía
       `store.write_to_online_store()`.

Sigue literalmente el patrón de la Clase 3 (IA en Producción, UdeSA 2026).
La única diferencia importante con la práctica es que reemplazamos
`df[col].astype("category").cat.codes` (no reproducible entre runs) por
`sklearn.preprocessing.LabelEncoder`, justificado por la slide 33 de
Clase 2 ("empaquetado de pre-procesamiento" — mismo pipeline en training
y serving).

Benja — branch feature_store
"""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---- Paths y constantes ----------------------------------------------------
FEATURE_STORE_REPO = Path("/opt/airflow/feature_store")
DATA_DIR = FEATURE_STORE_REPO / "data"
CSV_PATH = DATA_DIR / "dataset.csv"
PARQUET_PATH = DATA_DIR / "well_features.parquet"

DATASET_DOWNLOAD_URL = os.environ.get(
    "DATASET_DOWNLOAD_URL",
    "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/"
    "resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/"
    "produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv",
)

# Columnas del CSV que nos interesan. `idpozo` es la entidad, `fecha` se
# construye a partir de anio+mes.
NUMERIC_COLS = ["prod_gas", "prod_pet", "prod_agua", "tef", "profundidad"]
CAT_COLS = ["tipoextraccion"]
WINDOW_SIZE = 10  # número de lecturas usadas para los features de ventana


# ---- Pasos individuales ----------------------------------------------------
def download_data() -> Path:
    """Descarga el CSV crudo a `feature_store/data/dataset.csv`."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando dataset desde %s", DATASET_DOWNLOAD_URL)
    urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, CSV_PATH)
    logger.info("CSV guardado en %s", CSV_PATH)
    return CSV_PATH


def prepare_offline_store() -> Path:
    """Lee el CSV crudo y escribe `well_features.parquet` (offline store).

    Produce UNA fila por pozo con:
        - Los valores más recientes de `prod_gas`, `prod_pet`, `prod_agua`,
          `tef`, `profundidad`, `tipoextraccion`.
        - Features de ventana calculados sobre las últimas 10 lecturas:
          `avg_prod_gas_10m`, `avg_prod_pet_10m`, `last_prod_gas`,
          `last_prod_pet`, `n_readings`.
        - `fecha` = último mes registrado + 1 mes, representando la
          "próxima ingesta" sobre la que queremos inferir (patrón Clase 3).

    Nota académica (Clase 2, slide 33): el encoding de `tipoextraccion`
    se hace con `sklearn.LabelEncoder` en vez de `.cat.codes` para
    garantizar reproducibilidad entre runs y evitar training-serving
    skew.
    """

    df = pd.read_csv(CSV_PATH)

    # Construimos `fecha` a partir de anio+mes (día 1 del mes).
    df["fecha"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    required = ["idpozo", "fecha", *NUMERIC_COLS, *CAT_COLS]
    df = df[required].dropna(subset=["idpozo", "fecha"])

    # Encoding reproducible de la categórica.
    # El encoder queda en memoria durante el build; cuando el training DAG
    # lo necesite, vuelve a aplicar el mismo encoding sobre los mismos
    # valores porque el orden de las categorías en el dataset completo es
    # determinístico.
    le = LabelEncoder()
    df[CAT_COLS[0]] = le.fit_transform(df[CAT_COLS[0]].astype(str))
    df[CAT_COLS[0]] = df[CAT_COLS[0]].astype("int32")

    # Numéricas → Float32
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)

    records = []
    for well_id, group in df.groupby("idpozo", sort=False):
        group = group.sort_values("fecha").reset_index(drop=True)
        if group.empty:
            continue

        tail = group.tail(WINDOW_SIZE)
        last = group.iloc[-1]

        # Creamos la fila "próxima ingesta": misma info del último mes pero
        # con fecha = último mes + 1 (patrón de la Clase 3).
        next_fecha = last["fecha"] + pd.DateOffset(months=1)

        records.append(
            {
                "idpozo": int(well_id),
                "fecha": next_fecha,
                "prod_gas": float(last["prod_gas"]),
                "prod_pet": float(last["prod_pet"]),
                "prod_agua": float(last["prod_agua"]),
                "tef": float(last["tef"]),
                "profundidad": float(last["profundidad"]),
                "tipoextraccion": int(last["tipoextraccion"]),
                "avg_prod_gas_10m": float(tail["prod_gas"].mean()),
                "avg_prod_pet_10m": float(tail["prod_pet"].mean()),
                "last_prod_gas": float(tail["prod_gas"].iloc[-1]),
                "last_prod_pet": float(tail["prod_pet"].iloc[-1]),
                "n_readings": int(len(tail)),
            }
        )

    features_df = pd.DataFrame.from_records(records)
    # Parquet necesita tipos coherentes para que Feast lea el schema.
    features_df["fecha"] = pd.to_datetime(features_df["fecha"])
    features_df.to_parquet(PARQUET_PATH, index=False)

    #archival de datos
    date_str = datetime.now().strftime("%Y%m%d")
    archival_parquet_path = PARQUET_PATH.parent / f"{PARQUET_PATH.stem}_{date_str}{PARQUET_PATH.suffix}"
    features_df.to_parquet(archival_parquet_path, index=False)

    logger.info(
        "Offline store escrito: %s filas en %s", len(features_df), PARQUET_PATH
    )
    return PARQUET_PATH


def feast_apply() -> None:
    """Corre `feast apply` desde el repo del feature store.

    Usa subprocess para invocar la CLI — es lo que la práctica de Clase 3
    muestra literalmente y evita depender de APIs internas de Feast.
    """

    logger.info("Corriendo `feast apply` en %s", FEATURE_STORE_REPO)
    result = subprocess.run(
        ["feast", "apply"],
        cwd=str(FEATURE_STORE_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    logger.info("feast apply stdout:\n%s", result.stdout)
    if result.returncode != 0:
        logger.error("feast apply stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"feast apply falló con código {result.returncode}: {result.stderr}"
        )


def populate_online_store() -> None:
    """Escribe la última fila por pozo del parquet en el online store.

    Patrón directo de la Clase 3: usamos `store.write_to_online_store` en
    vez de `materialize_incremental`. El resultado es equivalente para
    nuestro caso (una fila por entidad).
    """

    # Import local para evitar que Airflow serialice Feast al parsear el DAG.
    from feast import FeatureStore

    features_df = pd.read_parquet(PARQUET_PATH)
    # `write_to_online_store` acepta un DataFrame con las columnas del
    # FeatureView + la columna de la entidad + la columna de timestamp.
    features_df["fecha"] = pd.to_datetime(features_df["fecha"])

    store = FeatureStore(repo_path=str(FEATURE_STORE_REPO))
    store.write_to_online_store(
        feature_view_name="well_stats",
        df=features_df,
    )

    logger.info(
        "Online store actualizado con %s filas (una por pozo)", len(features_df)
    )


# ---- Entrypoint para correr todo en local (testing rápido) ----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_data()
    prepare_offline_store()
    feast_apply()
    populate_online_store()
