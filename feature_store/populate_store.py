"""Construcción del offline + online store de Feast para oil_gas_production.

Este módulo implementa la lógica que el DAG `build_feature_store` ejecuta:

    1. Descarga el CSV crudo de datos.gob.ar.
    2. Construye `well_features.parquet` (offline store) con UNA FILA POR
       POZO POR MES. Cada fila tiene:
       - Lag features (producción de gas/pet/agua en T-1, T-2, T-3)
       - Rolling averages (últimos 10 meses)
       - Variables estáticas/categóricas (tipoextraccion, tipopozo, etc.)
       - Columnas target (prod_gas, prod_pet, prod_agua del mes T) para
         training — el experimento YAML define cuáles son features vs target.
    3. Hace `feast apply` para crear/actualizar el registry.
    4. Escribe la última fila por pozo en el online store SQLite vía
       `store.write_to_online_store()`.

Encoding de categóricas con `sklearn.LabelEncoder` en vez de `.cat.codes`
para reproducibilidad (Clase 2, slide 33 — empaquetado de pre-procesamiento).
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

# Columnas del CSV que nos interesan.
TARGET_COLS = ["prod_gas", "prod_pet", "prod_agua"]
STATIC_NUM_COLS = ["tef", "profundidad"]
CAT_COLS = ["tipoextraccion", "tipopozo", "provincia", "cuenca"]
LAG_PERIODS = [1, 2, 3]  # meses hacia atrás para lag features
WINDOW_SIZE = 10  # meses para rolling averages


# ---- Pasos individuales ----------------------------------------------------
def download_data() -> Path:
    """Descarga el CSV crudo a `feature_store/data/dataset.csv`."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando dataset desde %s", DATASET_DOWNLOAD_URL)
    urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, CSV_PATH)
    logger.info("CSV guardado en %s", CSV_PATH)
    return CSV_PATH


def build_inference_row(df_group: pd.DataFrame) -> pd.Series:
    """Genera la fila T+1 por pozo: lags desde las últimas observaciones,
    rolling avg sobre los últimos WINDOW_SIZE meses, targets en None.
    """

    df_group = df_group.sort_values("fecha")
    if len(df_group) == 0:
        return pd.Series(dtype="object")

    last = df_group.iloc[-1].copy()
    last["fecha"] = last["fecha"] + pd.DateOffset(months=1)

    for target in TARGET_COLS:
        last[f"{target}_t1"] = df_group.iloc[-1][target]
        last[f"{target}_t2"] = df_group.iloc[-2][target] if len(df_group) > 1 else None
        last[f"{target}_t3"] = df_group.iloc[-3][target] if len(df_group) > 2 else None

    for target in TARGET_COLS:
        last[f"avg_{target}_{WINDOW_SIZE}m"] = df_group[target].tail(WINDOW_SIZE).mean()

    for target in TARGET_COLS:
        for lead in [1, 2]:
            last[f"{target}_f{lead}"] = None
        last[target] = None

    return last


def prepare_offline_store() -> Path:
    """Lee el CSV crudo y escribe `well_features.parquet` (offline store).

    Produce UNA FILA POR POZO POR MES con:
        - Targets del mes actual: `prod_gas`, `prod_pet`, `prod_agua`.
        - Lag features: producción de gas/pet/agua en T-1, T-2, T-3.
        - Rolling averages: promedio de los últimos 10 meses.
        - Estáticas: `tef`, `profundidad`.
        - Categóricas: `tipoextraccion`, `tipopozo`, `provincia`, `cuenca`.

    Filas con lags incompletos (primeros 3 meses de cada pozo) se descartan.

    Encoding de categóricas con `sklearn.LabelEncoder` (Clase 2, slide 33).
    """

    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Construimos `fecha` a partir de anio+mes (día 1 del mes).
    df["fecha"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    required = ["idpozo", "fecha", *TARGET_COLS, *STATIC_NUM_COLS, *CAT_COLS]
    df = df[required].dropna(subset=["idpozo", "fecha"])

    # Encoding reproducible de categóricas.
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        df[col] = df[col].astype("int32")

    # Numéricas → Float32
    for col in TARGET_COLS + STATIC_NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)

    # --- Lag features y rolling averages (vectorizado por grupo) ---
    grouped = df.groupby("idpozo", sort=False)

    for target in TARGET_COLS:
        # Lags: T-1, T-2, T-3
        for lag in LAG_PERIODS:
            col_name = f"{target}_t{lag}"
            df[col_name] = grouped[target].shift(lag).astype("float32")

        # Rolling average últimos WINDOW_SIZE meses (sobre datos anteriores)
        col_name = f"avg_{target}_{WINDOW_SIZE}m"
        df[col_name] = (
            grouped[target]
            .shift(1)
            .transform(lambda s: s.rolling(WINDOW_SIZE, min_periods=1).mean())
            .astype("float32")
        )

    # Forward targets: producción en T+1 y T+2 (para predicción multi-step)
    # NO se agregan al FeatureView — son targets, no features. Últimas 1-2
    # filas por pozo quedarán con NaN (Franco filtra en training).
    for target in TARGET_COLS:
        for lead in [1, 2]:
            col_name = f"{target}_f{lead}"
            df[col_name] = grouped[target].shift(-lead).astype("float32")

    # Descartar filas sin lags completos (primeros 3 meses de cada pozo)
    lag_cols = [f"{t}_t{l}" for t in TARGET_COLS for l in LAG_PERIODS]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    # Asegurar tipos para Feast
    df["idpozo"] = df["idpozo"].astype("int64")
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Construimos la fila de inferencia (T+1 desde la última observación) por pozo
    grouped_for_inference = df.groupby("idpozo", sort=False, group_keys=False)
    inference_rows = grouped_for_inference.apply(build_inference_row)
    inference_rows = inference_rows.dropna(subset=["idpozo"]).reset_index(drop=True)

    df = pd.concat([df, inference_rows], ignore_index=True)
    df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)

    # Re-cast después del concat: los inference rows vienen como Series object
    # y ensucian los dtypes, lo que rompe la escritura al online store.
    df["idpozo"] = df["idpozo"].astype("int64")
    df["fecha"] = pd.to_datetime(df["fecha"])
    for col in CAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int32")
    float_cols = (
        TARGET_COLS
        + STATIC_NUM_COLS
        + [f"{t}_t{l}" for t in TARGET_COLS for l in LAG_PERIODS]
        + [f"avg_{t}_{WINDOW_SIZE}m" for t in TARGET_COLS]
        + [f"{t}_f{l}" for t in TARGET_COLS for l in [1, 2]]
    )
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # fecha_ts: unix seconds, para poder leer la fecha máxima por pozo desde el online store
    df["fecha_ts"] = (df["fecha"].astype("int64") // 10**9).astype("int64")

    df.to_parquet(PARQUET_PATH, index=False)

    # Archival con fecha (agregado por Franco)
    date_str = datetime.now().strftime("%Y%m%d")
    archival_path = PARQUET_PATH.parent / f"{PARQUET_PATH.stem}_{date_str}{PARQUET_PATH.suffix}"
    df.to_parquet(archival_path, index=False)

    logger.info(
        "Offline store escrito: %s filas (%s pozos) en %s",
        len(df), df["idpozo"].nunique(), PARQUET_PATH,
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

    Del parquet completo (múltiples filas por pozo), toma solo la última
    fila de cada pozo (la más reciente) para servir en inferencia.
    Patrón directo de la Clase 3: `store.write_to_online_store`.
    """

    # Import local para evitar que Airflow serialice Feast al parsear el DAG.
    from feast import FeatureStore

    features_df = pd.read_parquet(PARQUET_PATH)
    features_df["fecha"] = pd.to_datetime(features_df["fecha"])

    # Solo la última fila por pozo para el online store
    latest = (
        features_df
        .sort_values("fecha")
        .groupby("idpozo", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )

    store = FeatureStore(repo_path=str(FEATURE_STORE_REPO))
    store.write_to_online_store(
        feature_view_name="well_stats",
        df=latest,
    )

    logger.info(
        "Online store actualizado con %s filas (última por pozo, de %s totales en parquet)",
        len(latest), len(features_df),
    )


# ---- Entrypoint para correr todo en local (testing rápido) ----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_data()
    prepare_offline_store()
    feast_apply()
    populate_online_store()
