from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
import requests
from datetime import datetime


DATASET_DOWNLOAD_URL = "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv"
DATA_SAVE_PATH = "/opt/airflow/data/data.csv"
DATA_OUTPUT_PATH = "/opt/airflow/data/data_preprocessed.csv"

NUM_FEATURES = [
    "prod_pet", "prod_agua", "iny_agua", "iny_gas",
    "iny_co2", "tef", "vida_util", "profundidad",
    "anio", "mes"
]

CAT_FEATURES = [
    "tipoextraccion", "tipopozo", "provincia", "cuenca"
]

TARGET = "prod_gas"


@dag(
    dag_id='data_pipeline',
    description='Pipeline de descarga y procesamiento de data con Airflow',
    start_date=datetime(2026, 1, 1),
    #Hay q schedulearla
    params={
    'date_from': Param(
        default=None,
        type=['null', 'string'],
        description='Fecha inicio del rango (YYYY-MM-DD). Dejar vacío para no filtrar.',
    ),
    'date_to': Param(
        default=None,
        type=['null', 'string'],
        description='Fecha fin del rango (YYYY-MM-DD). Dejar vacío para no filtrar.',
    )}
)

def data_pipeline():
  @task
  def download_dataset(url, DATA_SAVE_PATH):
    """Descarga un dataset CSV desde una URL y lo guarda en disco.
    Args: url (str) - URL de descarga, save_path (str) - ruta local del archivo.
    Retorna: la ruta del archivo guardado.
    """
    response = requests.get(url)
    response.raise_for_status()  # lanza error si falla la descarga

    with open(DATA_SAVE_PATH, "wb") as f:
        f.write(response.content)
    return DATA_SAVE_PATH
  
  @task
  def preprocess(csv_path,output_path):
    """Lee el CSV, selecciona las columnas relevantes y codifica
    la columna 'tipoextraccion' como enteros.
    Args: csv_path (str) - ruta al archivo CSV.
    Retorna: dict con nombres de columnas como claves y listas de valores.
    """
    import pandas as pd
    from airflow.sdk import get_current_context

    df = pd.read_csv(csv_path)

    # encoding categóricas
    for col in CAT_FEATURES:
        df[col] = df[col].astype("category").cat.codes

    # seleccionar columnas
    df = df[NUM_FEATURES + CAT_FEATURES + [TARGET]]

    #filtrado por fecha
    context = get_current_context()
    params = context["params"]

    date_from = params.get("date_from")
    date_to = params.get("date_to")
    
    df["fecha"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-01"
    )
    if date_from:
        df = df[df["fecha"] >= pd.to_datetime(date_from)]

    if date_to:
        df = df[df["fecha"] <= pd.to_datetime(date_to)]

    df = df.drop(columns=["fecha"])

    # guardar
    df.to_csv(output_path, index=False)

    return output_path
  

  # secuencia de tasks
  data_path = download_dataset(DATASET_DOWNLOAD_URL, DATA_SAVE_PATH)
  data = preprocess(data_path,DATA_OUTPUT_PATH)
  
data_pipeline()
