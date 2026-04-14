from airflow.decorators import dag, task
from airflow.models.param import Param
from datetime import datetime


@dag(
    dag_id="model_manual_migration",
    description="Migra un modelo seleccionado de manera manual usando model_id",
    start_date=datetime(2024, 1, 1),
    params={
        "model_id": "COMPLETAR",
        "registered_model_name": "gas_model",
    },
)
def model_manual_migration():

    @task
    def migrate():
        from airflow.sdk import get_current_context
        from mlflow.tracking import MlflowClient

        context = get_current_context()
        params = context["params"]

        model_id = params["model_id"]
        model_name = params["registered_model_name"]

        if not model_id or model_id == "COMPLETAR":
            raise ValueError("Debes especificar model_id")

        client = MlflowClient()
        model_uri = f"models:/{model_id}"

        # Crear modelo registrado si no existe
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        # Nueva versión del modelo
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
        )

        # Promover a production (alias)
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=mv.version,
        )

    migrate()


dag = model_manual_migration()