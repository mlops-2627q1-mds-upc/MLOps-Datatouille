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
