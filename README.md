# Intelligent Message Analysis Platform

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Intelligent message analysis platform: end-to-end binary text classification (spam vs. not spam) with automated preprocessing, data validation, four different model searches, and experiment tracking. Orchestrated with DVC, validated with Great Expectations, tracked with MLflow (DagsHub), and monitored for CO2 with CodeCarbon.

## Pipeline

`dvc.yaml` defines the stages; parameters live in `params.yaml` and `conf/`:

```
download -> clean -> validate_processed -> split -> train -> evaluate -> select
                                                            (4 models,    (picks best
                                                             Optuna CV)    by metric)
```

* `clean`: deduplication, contact anonymization (`xxemail`/`xxphone`), count features, text normalization.
* `validate_processed`: Great Expectations checkpoint against the cleaned dataset (schema, nulls, ranges, label values); report in `data/processed/gx_validation.json`.
* The contact-detail expectation of `validate_processed` assumes `data.anonymize: true`. With anonymization disabled, the stage fails by design, since raw contact details must not reach training.
* `train`/`evaluate`: per model (`logreg`, `naive_bayes`, `linear_svm`, `random_forest`) logs an MLflow child run per Optuna trial: hyperparameters, CV/test metrics, CO2, top-20 features, and split MD5 hashes. Experimental use groups them under a parent run (see `docs/docs/experiment-tracking.md`). 
* Outputs: `models/`, `reports/metrics/`, `reports/features/`, `reports/emissions/`, `reports/best_model.json`.

## Quickstart (setup & run)

```bash
git clone git@github.com:mlops-2627q1-mds-upc/MLOps-Datatouille.git
cd MLOps-Datatouille
python -m venv venv && source venv/bin/activate   # Python 3.10
pip install -r requirements.txt                   # installs deps + `-e .`

cp env.example .env                               # MLflow (DagsHub) tracking URI + credentials

dvc pull                                          # fetch versioned data/models
dvc exp run -n <exp_name>                         # run the full pipeline as a tracked experiment
```

Deprecated local MLflow server (instead of DagsHub):

```bash
MLFLOW_PORT=5050 docker compose --env-file infra/mlflow/env.example \
  -f infra/mlflow/docker-compose.yaml up -d
export MLFLOW_TRACKING_URI=http://localhost:5050
```

## `dvc repro` vs `dvc exp run`

**Important:** run exploratory experiments with `dvc exp run`, **not** `dvc repro`. An uncommitted `dvc repro` run is logged to MLflow, but no DVC snapshot of its workspace state exists to restore it. Details in `EDN-007`.

Both execute `dvc.yaml`, but they serve different purposes:

| | `dvc exp run` | `dvc repro` |
|---|---|---|
| Purpose | Run the pipeline as a tracked experiment you can keep, compare, and restore | Rebuild pipeline outputs when code, data, or params change |
| Git | Creates a Git ref snapshot of the workspace (commit + params) | Uses the committed code/lock as-is |
| MLflow | Groups runs under a parent run tagged with the experiment name | Stands alone |
| Restore | `dvc exp pull` + `dvc exp apply <exp_name>` restore the workspace state | `dvc pull` restores the current pipeline state |

Rules of thumb:

* Use `dvc exp run -n <exp_name>` for every run of the pipeline that you want to keep: experimentation, hyperparameter searches, threshold changes, model comparisons. Push it with `dvc exp push origin <exp_name>` so teammates can pull it.
* Use `dvc repro` for quick local checks and to update the committed baseline: commit the resulting `dvc.lock` and run `dvc push`, so that `git checkout + dvc pull` correctly restores it.
* Uncommitted runs are only restorable when launched through `dvc exp run`: a manual `python -m src.modeling.train` gets logged to MLflow, but no DVC artifact records the workspace state.

Details on the MLflow run tree and experiment restore: see `docs/docs/experiment-tracking.md` and `EDN-007`.

## Tests

Pytest tests are standalone (not part of the DVC pipeline):

```bash
# Fast suite: hermetic unit + integration tests, with coverage gate (>= 80%)
pytest tests/unit tests/integration -v --cov=src --cov-fail-under=80

# Regression tests (need `dvc pull` or `dvc repro` first: data + trained models)
for model in logreg naive_bayes linear_svm random_forest; do
  MODEL_NAME="$model" pytest -q tests/regression/ || exit 1
done
# Single model: MODEL_NAME=logreg pytest -v tests/regression/test_model.py
```

* `tests/unit/`: helper logic, FastAPI route schemas, CLI, cleaning/anonymization edges.
* `tests/integration/`: preprocessing/large pipeline flows and Great Expectations configuration (GX config tests work from in-memory DataFrames; `test_processed_data` skips itself when `data/processed/clean.csv` is absent).
* tests/regression/: test_model.py (accuracy and macro precision, recall and F1 >= 0.95 on the test split) and test_robustness.py (invariance: macro F1 >= 0.90 on perturbed messages and >= 90% unchanged predictions; directional: appending a spam phrase to legitimate messages does not lower the spam score in >= 90% of cases).

## API

The spam classifier is served through a FastAPI application (`src/api/`). The API loads the model registered in the MLflow Model Registry under the alias `production` (`models:/spam-classifier@production`) once at startup, and does not start if the model cannot be loaded.

### 1. Install the dependencies

```bash
python -m venv venv && source venv/bin/activate   # Python 3.10
pip install -r requirements.txt                   # includes fastapi and uvicorn
```

### 2. Set the environment variables

```bash
cp env.example .env   # then fill in the DagsHub credentials
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | Yes | | MLflow server holding the Model Registry (DagsHub) |
| `MLFLOW_TRACKING_USERNAME` | Yes | | DagsHub user |
| `MLFLOW_TRACKING_PASSWORD` | Yes | | DagsHub token |
| `MLFLOW_MODEL_NAME` | No | `spam-classifier` | Registered model served by the API |
| `MLFLOW_MODEL_ALIAS` | No | `production` | Alias of the version to serve |
| `PREDICTIONS_DB_PATH` | No | `data/api/predictions.db` | SQLite file of predictions and feedback (ignored by Git) |
| `GEMINI_API_KEY` | No | | Gemini API key for summaries |
| `CLOUD_SUMMARIZATION_URL` | No | | Cloud summary endpoint, required for cloud mode or fallback |

### 3. Register the model to serve

The `select` stage writes the best model to `reports/best_model.json`. The following command logs that model to its MLflow run, registers it as a new version of `spam-classifier` and points the `production` alias to it. It only needs to be run when the served model changes.

```bash
dvc pull   # or dvc repro, so that models/ and reports/best_model.json exist
python -m src.modeling.register --selection reports/best_model.json
```

### 4. Start the API

```bash
uvicorn src.api.api:app --reload                     # local development
uvicorn src.api.api:app --host 0.0.0.0 --port 8080   # server (e.g. the FIB virtual machine)
```

The interactive documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

### 5. Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Welcome message, name and version of the API |
| `GET` | `/health` | Service status and loaded model version |
| `GET` | `/model` | Registered name, version, MLflow run and metrics of the production model |
| `POST` | `/predict` | Classification of 1 to 100 messages as `spam` or `not_spam` |
| `POST` | `/feedback` | Storage of the label that the user considers correct for a prediction |

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/model

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Congratulations! You won a FREE iPhone. Claim it at http://win.example",
                 "Hi team, the meeting has been moved to 3pm."],
       "threshold": 0.5}'

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": 2, "corrected_label": "spam"}'
```

Example response of `/predict`:

```json
{
  "model_name": "spam-classifier",
  "model_version": "1",
  "threshold": 0.5,
  "predictions": [
    {"prediction_id": 1, "label": "spam", "confidence": 0.99, "spam_probability": 0.99},
    {"prediction_id": 2, "label": "not_spam", "confidence": 0.99, "spam_probability": 0.01}
  ]
}
```

* Each message must contain between 1 and 10,000 characters. Email addresses and international phone numbers are replaced with `xxemail` and `xxphone` before inference.
* Every prediction is stored with the anonymized message, the model version and a UTC timestamp, for drift monitoring. The `prediction_id` of each result is the reference to correct it through `/feedback`. Each prediction accepts one correction.
* `threshold` (optional, default 0.5) is the spam probability from which a message is labelled as spam. A higher threshold produces fewer false positives.
* `confidence` is the probability of the returned label. `spam_probability` is the probability of spam, independent of the threshold.
* `/predict` tries Gemini for each summary, then falls back to the configured cloud endpoint if Gemini fails or returns an empty summary. Use `/predict?cloud=true` to select cloud directly. If no summary is available, `summary` is `null` and the error is logged; classification is still returned.
* The cloud endpoint receives each original message as JSON `{"text": "..."}` and must return JSON containing a nonempty `summary` string. Both Gemini and cloud receive the original text, including any personal data; anonymization applies to classification and stored predictions.

### Status codes

| Code | Meaning |
|---|---|
| `200` / `201` | Success (`201` when feedback is stored) |
| `400` | Invalid value detected while processing the request |
| `404` | Feedback refers to a prediction that does not exist |
| `409` | The prediction already has feedback |
| `422` | Request does not match the schema (empty or too long message, unknown label, threshold outside [0, 1]) |
| `500` | Unexpected error, logged on the server without exposing its details |
| `503` | Model not loaded |


## 6. Cloud Summarization Service (UPC Virtech and ngrok)

The project includes an independent text summarization service deployed on a virtual machine provided by UPC Virtech. This service was developed to support remote inference using a locally hosted language model, allowing the main application and other team members to generate summaries without executing the model on their own machines.

The service uses the `Falconsai/text_summarization` model, based on the T5 architecture, and performs inference on the virtual machine's CPU. The model and tokenizer are stored in the VM, while a FastAPI application handles incoming requests, executes the model and returns the generated summaries.

The application runs through Uvicorn on port `8080`. Since the virtual machine does not directly expose the service through a public HTTPS domain, ngrok is used to establish a secure tunnel between the local FastAPI server and external clients.

The request flow is:

```text
External Client
      |
      v
ngrok Public HTTPS Endpoint
      |
      v
UPC Virtech Virtual Machine
      |
      v
FastAPI (127.0.0.1:8080)
      |
      v
Local T5 Summarization Model
      |
      v
JSON Response
```

### Service endpoints

The summarization service exposes two endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/cloud/health` | Checks whether the service is available. |
| `POST` | `/cloud/summarize` | Generates a summary of the submitted text and returns the inference latency. |

The public service can be accessed through:

**https://kangaroo-gloater-gag.ngrok-free.dev**

A summarization request can be performed using:

```bash
curl -X POST \
  "https://kangaroo-gloater-gag.ngrok-free.dev/cloud/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning models can help automate repetitive tasks and improve the efficiency of software development."
  }'
```

The response contains the generated summary and the inference latency, reported through the `summary` and `latency_ms` fields.

## 7. Public API Deployment (Render)

The main Datatouille API is deployed on Render as a publicly accessible FastAPI web service. This deployment provides a centralized interface for spam detection, text summarization, model information and user feedback, allowing external applications to consume the system without running the classification model locally.

Unlike the Virtech service, which is dedicated to summarization, the Render deployment hosts the main application and coordinates its interactions with the MLflow Model Registry, Google Gemini and the optional cloud summarization service.

The public API is available at:

**https://datatouille-api.onrender.com**

The interactive Swagger documentation can be accessed at:

**https://datatouille-api.onrender.com/docs**

### Deployment configuration

The service is connected to the project's GitHub repository and deployed using Render's Web Service functionality. The application uses Python `3.13.9`, configured through `.python-version`, while the dependencies required for inference are defined separately in `requirements-render.txt`.

During the build process, Render installs the application dependencies using:

```bash
pip install -r requirements-render.txt
```

The FastAPI application is started with Uvicorn:

```bash
uvicorn src.api.api:app --host 0.0.0.0 --port $PORT
```

The server listens on the port assigned by Render, which manages the public HTTPS endpoint and the application deployment process. This removes the need to expose a local development server or maintain an external tunnel for the main API.

### Model loading and inference

At application startup, FastAPI connects to the MLflow Model Registry hosted on DagsHub. The registered spam classification model is identified using its model name and the `production` alias.

Once the corresponding version has been resolved, the model is downloaded and loaded into memory. Subsequent prediction requests reuse the loaded model, avoiding the need to retrieve it from MLflow for every inference.

When a client submits a request to `/predict`, the application preprocesses the input messages, performs spam classification and returns the predicted labels with their associated confidence scores.

The same endpoint also supports text summarization. Google Gemini is used as the primary summarization backend, while the Virtech service can be used as an alternative when configured.

If Gemini fails or returns an invalid summary, the application attempts to use the configured cloud summarization service. Clients can also explicitly request the cloud backend using `/predict?cloud=true`.

When no summarization backend is available, the classification results can still be returned with a `null` summary.

The application therefore separates the classification process from the summarization service, allowing the spam detection functionality to remain operational when an external summarization provider is unavailable.

### Public access and API usage

The Render deployment exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Returns general information about the API. |
| `GET` | `/health` | Reports the service status and loaded model version. |
| `GET` | `/model` | Returns model metadata and evaluation metrics from MLflow. |
| `POST` | `/predict` | Classifies messages and generates summaries. |
| `POST` | `/feedback` | Records user corrections associated with previous predictions. |

The service can be tested from any external client using its public URL:

```bash
curl https://datatouille-api.onrender.com/health
```

A prediction request can be submitted using:

```bash
curl -X POST \
  "https://datatouille-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Congratulations! You have won a free iPhone!",
      "Hi, the project meeting is scheduled for tomorrow."
    ],
    "threshold": 0.5
  }'
```

The response includes the predicted classification, confidence, spam probability, model version and generated summary when available. Each prediction also receives an identifier that can be used to submit feedback.

### Environment configuration

The deployment uses environment variables to configure its external integrations. These include the MLflow tracking URI, DagsHub authentication credentials, registered model name, production alias and Gemini API key.

The optional Virtech integration is configured through `CLOUD_SUMMARIZATION_URL`.

Sensitive values are managed through Render's environment configuration rather than being stored in the Git repository.

### Storage and free-tier limitations

The application currently uses SQLite to store predictions and user feedback. These records include anonymized messages, model predictions, timestamps and user corrections, providing information that can support future drift monitoring and model retraining.

**Storage limitation:** The public API runs on Render's free instance type, which uses an ephemeral filesystem. As a result, predictions and feedback stored in SQLite are lost when the service restarts, redeploys or is suspended due to inactivity. Persistent storage requires an external database or a persistent disk available on a suitable paid instance.

Render's free tier also suspends inactive services. When a new request arrives, the application must restart and reload the production model from MLflow, which can increase the response time of the first request.

CPU and memory resources are limited, so this deployment is intended primarily for development, integration testing and demonstrations rather than workloads requiring guaranteed availability.

## CO2 experiment

`src/modeling/measure_co2.py` measures the training energy of each model outside the DVC pipeline. It repeats 5-fold CV plus a final fit with fixed hyperparameters 5 times per model, interleaving the models and sampling power every 1 s, because one training lasts only a few seconds. Results are stored in `reports/emissions/co2_runs.jsonl`, together with the hardware that produced them.

```bash
python -m src.modeling.measure_co2 --train data/dataset/train.csv \
    --out reports/emissions/co2_runs.jsonl --reps 5
```

The pipeline itself samples every 15 s, so per-trial emissions in MLflow rely on very few readings and are indicative only.

## Quality checks (same as CI)

```bash
ruff check src tests
ruff format --check src tests
mypy src            # config in [tool.mypy] in pyproject.toml
```

## CI

CI runs the pipeline-level checks via `.github/workflows/ci-pipeline.yml`, with test scope depending on the merge level:

| Merge                  | Job                             | Runs                                                                          |
|------------------------|---------------------------------|-------------------------------------------------------------------------------|
| Task branch → `dev`    | Fast Feedback & Code Integration| Ruff, mypy, hermetic unit + integration tests, coverage gate                      |
| `dev` → `main`         | Comprehensive Pre-Production Validation | Enforces source branch is `dev`; re-runs Ruff + mypy; `dvc pull`; full-file GE validation; regression loop for the 4 models; full unit + integration suite |

## Project organization

```
├── .github/              <- Issue templates and CI workflow
├── conf/                 <- Hydra configuration of the training stage
│   ├── model/            <- Hyperparameter search space of each model
│   └── train.yaml
├── data/                 <- Versioned with DVC
│   ├── dataset/          <- Training and test splits
│   ├── processed/        <- Cleaned dataset and its validation report
│   └── raw/              <- Dataset as downloaded from Hugging Face
├── docs/                 <- Documentation: dataset and model card and experiment tracking guide
│   ├── docs/
│   ├── edn/              <- Engineering Decision Notebook entries
│   └── proposals/        <- Design proposals of each component
├── gx/                   <- Great Expectations suite and checkpoint
├── infra/mlflow/         <- Docker Compose setup of the MLflow server
├── models/               <- Trained models, versioned with DVC
├── notebooks/            <- Exploration notebooks.
├── reports/
│   ├── emissions/        <- CodeCarbon measurements
│   ├── features/         <- Top-20 influential features of each model
│   └── metrics/          <- Training and test metrics of each model
├── scripts/              <- Helper scripts (e.g. synthetic data generation)
├── src/
│   ├── api/              <- Web API for inference and feedback collection
│   ├── data/             <- Download, cleaning, validation, and splitting
│   ├── modeling/         <- Training, evaluation, selection, explanation, and tracking
│   ├── monitoring/       <- Data drift detection and performance monitoring (Reserved for Milestone 6)
│   ├── config.py         <- Paths and tool settings
│   └── emissions.py      <- Energy and CO2 tracking
├── tests/
│   ├── integration/
│   ├── regression/
│   └── unit/
├── .dvcignore
├── .gitignore
├── AGENTS.md
├── CONTRIBUTING.md
├── dvc.lock
├── dvc.yaml
├── env.example           <- MLflow tracking env vars template
├── LICENSE
├── params.yaml
├── pytest.ini
├── pyproject.toml
└── requirements.txt
```

## Documentation & pointing docs

* [`CONTRIBUTING.md`](CONTRIBUTING.md) — **How to contribute**: branching strategy (task branches -> `dev` -> `main`), commit style, DVC data hygiene, CI gates per merge, and code good practices.
* [`AGENTS.md`](AGENTS.md) — **Rules for AI coding agents** working in this repo: sources of truth, layout roles, permitted edits, and validation workflow.
* `docs/docs/dataset_card.md` — Dataset provenance and description.
* `docs/docs/experiment-tracking.md` — How MLflow runs are structured and how to restore an experiment.
* `docs/edn/` — Engineering Decision Notebooks for architecture decisions.
