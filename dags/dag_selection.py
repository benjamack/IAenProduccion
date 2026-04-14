from airflow.decorators import dag, task
from datetime import datetime


@dag(
    dag_id="automatic_model_selection",
    description="Pipeline de seleccion de modelo automatico para produccion",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "decision_metric": "mse",
        "experiment_name": "COMPLETAR",
        "decision_logic": "ASC",
        "registered_model_name": "gas_model",
    },
)
def automatic_model_selection():

    @task
    def select_best_model():
        import mlflow
        from airflow.sdk import get_current_context
        from mlflow.tracking import MlflowClient

        context = get_current_context()
        experiment_name = context["params"]["experiment_name"]
        decision_metric = context["params"]["decision_metric"]
        decision_logic = context["params"]["decision_logic"]

        client = MlflowClient()

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' no existe")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{decision_metric} {decision_logic}"],
            max_results=1,
        )

        if not runs:
            raise ValueError("No hay runs en el experimento")

        best_run = runs[0]

        if not best_run.outputs or not best_run.outputs.model_outputs:
            raise ValueError(
                f"El run {best_run.info.run_id} no tiene logged models asociados"
            )

        model_id = best_run.outputs.model_outputs[0].model_id
        return {"model_id": model_id}


    @task
    def register_and_promote(model_info):
        from airflow.sdk import get_current_context
        from mlflow.tracking import MlflowClient

        context = get_current_context()
        model_name = context["params"]["registered_model_name"]

        client = MlflowClient()

        model_id = model_info["model_id"]
        model_uri = f"models:/{model_id}"

        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=mv.version,
        )

    best_model = select_best_model()
    register_and_promote(best_model)


dag = automatic_model_selection()