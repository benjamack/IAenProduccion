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
| Escalabilidad de inferencia | Ray Serve 2.20 |
| Containerización | Docker Compose |

## Levantar el sistema

```bash
docker compose up --build
```

| Servicio | URL |
|---------|-----|
| Airflow UI | http://localhost:8181 (airflow/airflow) |
| MLflow UI | http://localhost:9191 |
| API | http://localhost:8000 (docs en `/docs`) |

## Pipeline end-to-end

### 1. Feature Store (DAG: `build_feature_store`)

- **Entidad**: `idpozo`
- **FeatureView**: `well_stats` — una fila por pozo por mes con targets (produccion de gas y petroleo), lag features de producciones de meses pasados (T-1, T-2, T-3), rolling averages deproduccion (10m), categóricas encoded con `LabelEncoder`, y forward targets (T+1, T+2) para training multi-step (a implementar en un futuro).
- **Inference row**: por cada pozo se agrega una fila T+1 adicional con los lags calculados desde la última observación real. Es lo que sirve el online store.
- **Offline**: `feature_store/data/well_features.parquet`. Además, en cada corrida se guarda una versión con timestamp del dataset (Formato:`well_features_AAAAMMDD.parquet`), lo que permite reproducir experimentos utilizando datos de una fecha específica.
- **Online**: Redis (`redis:6379`).

#### Elección de Redis como Online Store

La elección de Redis se debió a:

- **Baja latencia**: permite acceder a features rápidamente en memoria, lo cual es útil incluso en entornos de prueba cuando se simula serving en tiempo real.
- **Simplicidad de integración**: Redis está bien soportado por herramientas como Feast y es fácil de levantar con Docker.
- **Escalabilidad futura**: aunque el TP no requiere alta carga, Redis permite escalar si el sistema creciera o se utilizara en un contexto más productivo.
- **Consistencia con prácticas reales**: se utiliza comúnmente como online store en sistemas de ML, por lo que ayuda a acercar la solución a un caso de uso real.


#### Orquestación con Airflow

El flujo completo se ejecuta desde la UI de Airflow (`build_feature_store`), permitiendo un manejo más simple e intuitivo.

El pipeline está definido en un único DAG que cubre todo el proceso, desde la descarga de datos hasta la actualización del online store.

**Tasks:**
`download_raw_data` → `build_offline_store` → `feast_apply_task` → `populate_online_store_task`

### 2. Training (DAG:  `ml_training_pipeline`)

Entrena múltiples experimentos definidos en YAML (`experimentos/*.yaml`). Cada YAML declara el `model_type`, `model_params`, `target` y `features` por experimento. Se loggea a MLflow con `autolog` y se crean los logged models.

Parámetros del DAG:
- `experiment_name`: nombre del archivo YAML sin extensión (ej: `Experimento_1`).
- `fecha_data`: `Ultima(Default)` usa `well_features.parquet`; cualquier otro valor busca `well_features_<fecha>.parquet`.

#### Aclaración de diseño

Este pipeline fue diseñado de esta manera para facilitar la **planificación y ejecución de múltiples modelos en simultáneo**, permitiendo comparar rápidamente distintas configuraciones (features, hiperparámetros, targets, etc.) sin necesidad de modificar código.

Cada experimento definido en YAML se ejecuta de forma independiente dentro del DAG, lo que habilita:
- correr varios modelos en paralelo,
- mantener trazabilidad clara entre configuraciones,
- escalar fácilmente la cantidad de experimentos.


#### Logging en MLflow

Durante el entrenamiento, cada modelo loggea automáticamente métricas y parámetros clave en MLflow, lo que permite:

- **trackear fácilmente qué se corrió y con qué configuración**,  
- **comparar resultados entre experimentos desde la UI**,  
- **reproducir modelos en el futuro** (incluyendo qué datos se usaron).

Se registran los siguientes elementos:

**Métricas:**
- `mse`: error cuadrático medio del modelo  
- `r2`: coeficiente de determinación  

**Parámetros:**

- `model_type`: tipo de modelo (ej: random_forest)  
- `target`: variable objetivo a predecir  
- `features`: conjunto de variables utilizadas en el entrenamiento  
- `fecha_de_data`: identifica si se usó la última versión de los datos o un snapshot específico 

**Naming del run en MLflow**

El nombre de cada run se genera automáticamente combinando el tipo de modelo con sus hiperparámetros, lo que permite identificar fácilmente cada experimento en la UI.

**¿Por qué MLflow?**

Se eligió MLflow porque permite **trackear, comparar y reproducir experimentos de forma simple y centralizada**, mejorando significativamente la trazabilidad y gestión del ciclo de vida de los modelos.


### 3. Selección y promoción

Dos DAGs:

- **`automatic_model_selection`**: Dado un `experiment_name`, elige el mejor run de un experimento según `decision_metric` + `decision_logic` (`ASC`/`DESC`), lo registra como versión del `registered_model_name` (los valores posibles son `gas_model` o `pet_model`) y setea el alias `production`.

- **`model_manual_migration`**: Permite promover manualmente un modelo específico a producción a partir de su `model_id`, definiendo también el `registered_model_name`. Este DAG no realiza ninguna lógica automática de selección, sino que está pensado para casos donde la decisión se toma a partir de análisis en la UI de MLflow o cuando se necesita control explícito sobre qué modelo versionar.

#### Aclaración de uso

Se diseñaron ambos DAGs para cubrir distintos escenarios de selección de modelos:

- En algunos casos, dentro de un mismo experimento se entrenan modelos para **distintas predicciones** (por ejemplo, gas y petróleo). En ese contexto, no siempre tiene sentido usar una selección automática, ya que el “mejor modelo” depende del objetivo específico.

- Además, puede ocurrir que la selección del modelo no sea puramente por métrica, sino basada en un **análisis manual en la UI de MLflow** (por ejemplo, evaluando features, estabilidad o comportamiento del modelo).

Por eso, el DAG `model_manual_migration` permite:
- elegir explícitamente el `model_id`,
- definir manualmente el `registered_model_name`,
- y tener control total sobre qué modelo pasa a producción.

### 4. API de inferencia (FastAPI + Ray Serve)

FastAPI levanta al arranque cargando `gas_model@production` y `pet_model@production` desde MLflow, con sus listas de features guardadas como params del run.

#### Escalabilidad con Ray Serve

La API corre sobre **Ray Serve** para soportar múltiples requests concurrentes sin recargar el modelo en cada llamada. El deployment se configura con `@serve.deployment(num_replicas=3)`, lo que levanta tres réplicas del servicio con el modelo ya cargado en memoria. Esto es especialmente relevante para modelos como Random Forest, donde el tiempo de carga es significativo comparado con el de predicción. Ray distribuye el tráfico entre las réplicas automáticamente.

#### Hot reload de modelos

Cada 60 segundos la API chequea si cambió la versión `production` en el registro de MLflow (`model_version_changed()`). Si detecta un cambio (por ejemplo, tras una corrida del pipeline automático), recarga el modelo sin reiniciar el container. También se expone el endpoint `POST /reload-model` para forzar el reload manualmente.

Endpoints:

| Método | Path | Descripción |
|--------|------|-------------|
| GET | `/health` | Healthcheck |
| GET | `/api/v1/wells?date_query=YYYY-MM-DD` | Lista pozos con actividad en el mes previo y, por lo tanto, que se esperaria que sean activos en el mes consultado |
| GET | `/api/v1/forecast?id_well=...&date_start=...&date_end=...` | Pronóstico mensual de `prod_gas` y `prod_pet` en el rango suministrado |
| POST | `/reload-model` | Refresca los modelos sin reiniciar el container |

El `/forecast` lee la `max_fecha` del pozo, predice el mes base con el modelo (mes consecutivo al de la maxima fecha) y proyecta el resto del rango aplicando un factor de decaimiento mensual de `0.97^n`. Meses del rango de consulta anteriores al `max_fecha` se ignoran.

Nota: Se extendió el schema original de `/forecast` para devolver producción de gas y petróleo por separado en lugar de un único valor agregado.

### 5. Auto-train y deploy (DAGs: `Automatic_training_gas`, `Automatic_training_pet`)

Estos DAGs corren automáticamente el día 2 de cada mes (`schedule: 5 0 2 * *`, `is_paused_upon_creation=True`). Cada uno integra en un único pipeline el ciclo completo para su modelo:

**Tasks:** `get_experiments` → `train_model` (expand por experimento, paralelo) → `select_best_model` → `register_and_promote`

El `train_model` itera sobre todos los experimentos definidos en el YAML correspondiente (`Experimento_gas_auto.yaml` / `Experimento_pet_auto.yaml`), loggea a MLflow y registra métricas. El `select_best_model` busca el mejor run según `decision_metric` (default: `mse`, `ASC`) y lo pasa a `register_and_promote`, que crea la versión en el Model Registry y setea el alias `production`.

Esto cierra el loop de reentrenamiento: cada mes se construye el feature store → se entrena → el mejor modelo queda como `production` → la API lo recarga automáticamente.

### 6. Drift detection + Model decay (DAG: `drift_and_decay_report`)

Compara el snapshot actual del feature store contra el snapshot con el que se entrenó cada modelo y reporta:

- **KS Drift** (univariado): test de Kolmogorov–Smirnov por feature. Detecta cambios en la distribución marginal de cada variable. p-value < 0.05 ⇒ drift en esa feature.
- **Classifier Drift** (multivariado): entrena un `RandomForestClassifier` para distinguir referencia vs. actual. Si AUC ≫ 0.5 con p-value < 0.05 ⇒ las distribuciones conjuntas son distinguibles. Las feature importances señalan cuál cambió más.
- **Model decay**: re-evalúa el modelo productivo (`gas_model@production`, `pet_model@production`) sobre los datos actuales con ground truth y compara MSE/R² contra los valores loggeados en el run de entrenamiento.

Los métodos se eligieron siguiendo la práctica de Clase 6: el KS es rápido pero **no detecta drift de correlación**, mientras que ClassifierDrift sí lo detecta pero es más costoso. Son complementarios.

La referencia de cada modelo se obtiene automáticamente del param `fecha_de_data` guardado en MLflow al momento del entrenamiento — no hace falta indicarla manualmente.

Parámetro del DAG:

- `current_snapshot`: `Latest` (default), o una fecha `AAAAMMDD`.

Cada corrida crea un run en el experimento **`monitoring`** de MLflow con:

- Métricas: `gas_ks_n_features_drifted`, `gas_ks_min_pvalue`, `gas_classifier_auc`, `gas_classifier_p_value`, `gas_mse_current`, `gas_r2_current`, `gas_mse_delta`, `gas_r2_delta` (y análogas para `pet_`).
- Artifacts: `drift_report.html` (resumen visual con veredicto por modelo), `ks_results.csv`, `classifier_importances.csv`, histogramas y barras de importances en PNG.

Veredicto por modelo en el HTML: **REVISAR MODELO** si hay drift en KS o classifier, o si `|R²_delta| > 0.1`. **OK** en caso contrario.

## Estructura

```
├── api/                             FastAPI (Dockerfile propio)
├── config/                          airflow.cfg
├── dags/
│   ├── build_feature_store.py       Construcción del feature store
│   ├── dag_ml_train.py              Training multi-experimento
│   ├── dag_selection.py             Selección automática del mejor modelo
│   ├── dag_manual_migration.py      Promoción manual por model_id
│   ├── dag_gas_auto.py              Auto-train + deploy mensual para gas_model
│   ├── dag_pet_auto.py              Auto-train + deploy mensual para pet_model
│   └── dag_drift_report.py          Reporte de drift + model decay
├── experimentos/                    Experimentos YAML (model_type + params + features + target)
├── feature_store/                   Repo de Feast
│   ├── feature_store.yaml
│   ├── features.py
│   └── populate_store.py
├── monitoring/                      Módulo de drift + decay
│   ├── drift.py                     KSDrift + ClassifierDrift (scipy + sklearn)
│   ├── decay.py                     Re-evaluación de modelo productivo
│   └── report.py                    Generación de HTML + plots + CSVs
├── mlruns/                          Artifacts de MLflow (gitignored)
├── data/                            Dataset descargado (gitignored)
└── docker-compose.yaml
```

## Equipo

- Benja Mackinnon
- Franco Moscato
