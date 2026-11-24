# Intelligent Message Analysis Platform

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Intelligent message analysis platform: end-to-end binary text classification (spam vs. not spam) with automated preprocessing, data validation, four different model searches, and experiment tracking. Orchestrated with DVC, validated with Great Expectations, tracked with MLflow (DagsHub), and monitored for CO2 with CodeCarbon.

## Table of contents

1. [Pipeline](#1-pipeline)
2. [Quickstart (setup & run)](#2-quickstart-setup--run)
3. [`dvc repro` vs `dvc exp run`](#3-dvc-repro-vs-dvc-exp-run)
4. [Tests](#4-tests)
5. [API](#5-api)
   1. [Install the dependencies](#51-install-the-dependencies)
   2. [Set the environment variables](#52-set-the-environment-variables)
   3. [Register the model to serve](#53-register-the-model-to-serve)
   4. [Start the API](#54-start-the-api)
   5. [Endpoint reference](#55-endpoint-reference)
   6. [Data storage](#56-data-storage)
   7. [Authentication of the administration endpoints (`ADMIN_API_KEY`)](#57-authentication-of-the-administration-endpoints-admin_api_key)
   8. [Testing the API with Postman](#58-testing-the-api-with-postman)
   9. [OpenAPI specification and Swagger Editor](#59-openapi-specification-and-swagger-editor)
   10. [Status codes](#510-status-codes)
   11. [Troubleshooting](#511-troubleshooting)
6. [Cloud Summarization Service (UPC Virtech and ngrok)](#6-cloud-summarization-service-upc-virtech-and-ngrok)
7. [Public API Deployment (Render)](#7-public-api-deployment-render)
8. [CO2 experiment](#8-co2-experiment)
9. [Quality checks (same as CI)](#9-quality-checks-same-as-ci)
10. [CI](#10-ci)
11. [Project organization](#11-project-organization)
12. [Documentation & pointing docs](#12-documentation--pointing-docs)

## 1. Pipeline

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

## 2. Quickstart (setup & run)

```bash
git clone git@github.com:mlops-2627q1-mds-upc/MLOps-Datatouille.git
cd MLOps-Datatouille
python3.13 -m venv .venv && source .venv/bin/activate   # Python 3.13 (same as CI and Render)
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

## 3. `dvc repro` vs `dvc exp run`

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

## 4. Tests

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

## 5. API

The spam classifier is served through a FastAPI application located in `src/api/`. The API classifies messages, optionally summarizes them, stores every prediction and user correction in a PostgreSQL database, and exposes protected administration endpoints to manage the stored data.

When the application starts, it performs three steps before accepting any request:

1. It loads the model registered in the MLflow Model Registry under the alias `production` (`models:/spam-classifier@production`). If the model cannot be loaded, the application does not start.
2. It connects to the PostgreSQL database given by `DATABASE_URL` and creates the tables `predictions` and `feedback` if they do not exist yet. Existing tables and rows are never modified at startup.
3. It starts serving requests. When it shuts down, it closes the database connections.

The code is organized as follows:

| Module | Responsibility |
|---|---|
| `src/api/api.py` | FastAPI application: startup, endpoints, error handling and authentication of the administration endpoints |
| `src/api/schemas.py` | Pydantic schemas that validate every request and response |
| `src/api/settings.py` | Serving settings read from the environment (model, limits, `ADMIN_API_KEY`) |
| `src/api/model_loader.py` | Loading of the production model from MLflow |
| `src/database/connection.py` | SQLAlchemy engine and session factory for PostgreSQL |
| `src/database/models.py` | ORM mapping of the `predictions` and `feedback` tables |
| `src/database/prediction_store.py` | Storage operations (creation, reading, update and deletion of rows) |

### 5.1 Install the dependencies

The project uses **Python 3.13**, the same version as the CI pipeline and the Render deployment. Some pinned dependencies (for example `great_expectations`) cannot be installed on Python 3.14.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # includes fastapi, uvicorn, SQLAlchemy and psycopg
```

> **Note:** do not create the virtual environment inside a folder synchronized with a cloud service, such as the macOS *Desktop* or *Documents* folders with iCloud enabled. The service may remove the local copy of the installed packages, which makes Python block while importing them. In that case, create the environment elsewhere, for example `python3.13 -m venv ~/.venvs/datatouille`.

### 5.2 Set the environment variables

```bash
cp env.example .env   # then fill in the values below
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | Yes | | MLflow server holding the Model Registry (DagsHub) |
| `MLFLOW_TRACKING_USERNAME` | Yes | | DagsHub user |
| `MLFLOW_TRACKING_PASSWORD` | Yes | | DagsHub token |
| `MLFLOW_MODEL_NAME` | No | `spam-classifier` | Registered model served by the API |
| `MLFLOW_MODEL_ALIAS` | No | `production` | Alias of the version to serve |
| `DATABASE_URL` | Yes | | PostgreSQL connection URL, in the form `postgresql://user:password@host:5432/database`. Without it, the API does not start |
| `ADMIN_API_KEY` | No | | Secret key of the administration endpoints. If it is empty, those endpoints are disabled. See [Authentication of the administration endpoints](#57-authentication-of-the-administration-endpoints-admin_api_key) |
| `GEMINI_API_KEY` | No | | Gemini API key, used for summaries |
| `GEMINI_MODEL` | No | `gemini-3.8-flash` | Gemini model used for summaries |
| `CLOUD_SUMMARIZATION_URL` | No | | Cloud summarization endpoint (Virtech), used as fallback or when `cloud=true` |

The `.env` file is ignored by Git and must never be committed, since it contains credentials. The variables are read **only when the application starts**: after changing `.env`, the API must be restarted.

### 5.3 Register the model to serve

The `select` stage writes the best model to `reports/best_model.json`. The following command logs that model to its MLflow run, registers it as a new version of `spam-classifier` and points the `production` alias to it. It only needs to be run when the served model changes.

```bash
dvc pull   # or dvc repro, so that models/ and reports/best_model.json exist
python -m src.modeling.register --selection reports/best_model.json
```

### 5.4 Start the API

```bash
python -m uvicorn src.api.api:app --reload                     # local development
python -m uvicorn src.api.api:app --host 0.0.0.0 --port 8080   # server (e.g. the FIB virtual machine)
```

Running Uvicorn through `python -m` guarantees that the Uvicorn of the active virtual environment is used, and not another installation of the system.

The API is ready when the log shows the following lines (startup takes about 15 seconds, mostly spent downloading the model from DagsHub):

```
INFO | Loaded model spam-classifier version 1 (alias 'production')
INFO | Prediction database (PostgreSQL) ready
INFO:     Application startup complete.
```

With `--reload`, the application restarts, and downloads the model again, every time a Python file is saved. To run a long series of manual tests without modifying the code, start it without `--reload`.

The interactive documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

### 5.5 Endpoint reference

The endpoints are divided into **public** endpoints, which any client can call, and **administration** endpoints, which require the `X-API-Key` header.

| Method | Path | Access | Description |
|---|---|---|---|
| `GET` | `/` | Public | Welcome message, name and version of the API |
| `GET` | `/health` | Public | Service status and loaded model version |
| `GET` | `/model` | Public | Name, alias, version, MLflow run and metrics of the production model |
| `POST` | `/predict` | Public | Classification of 1 to 100 messages as `spam` or `not_spam`, with optional summaries |
| `POST` | `/summarize` | Public | Summary of a single message |
| `POST` | `/feedback` | Public | Storage of the label that the user considers correct for a prediction |
| `GET` | `/predictions` | Admin | List of the stored predictions |
| `GET` | `/predictions/{id}` | Admin | One stored prediction |
| `POST` | `/predictions` | Admin | Manual creation of a prediction row |
| `PUT` | `/predictions/{id}` | Admin | Replacement of the columns of a prediction |
| `DELETE` | `/predictions/{id}` | Admin | Deletion of a prediction and of its feedback |
| `GET` | `/feedback` | Admin | List of the stored feedback entries |
| `GET` | `/feedback/{id}` | Admin | One stored feedback entry |
| `PUT` | `/feedback/{id}` | Admin | Change of the corrected label of a feedback entry |
| `DELETE` | `/feedback/{id}` | Admin | Deletion of a feedback entry (its prediction is kept) |

`/feedback` appears twice on purpose: `POST /feedback` is public and creates a correction, while `GET /feedback` is an administration endpoint that lists them. The HTTP method determines the operation.

#### General endpoints

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/model
```

`/health` answers whenever the service is running. Its field `model_loaded` indicates whether predictions can be served.

#### Prediction: `POST /predict`

Request body:

| Field | Type | Required | Description |
|---|---|---|---|
| `texts` | list of strings | Yes | 1 to 100 messages, each with 1 to 10,000 characters and at least one non-whitespace character |
| `threshold` | number in [0, 1] | No (default `0.5`) | Spam probability from which a message is labelled as spam. A higher threshold produces fewer false positives and more false negatives |

Query parameters:

| Parameter | Default | Description |
|---|---|---|
| `summary` | `false` | When `true`, a summary is generated for every message |
| `cloud` | `false` | When `true` together with `summary=true`, the cloud service is used directly instead of Gemini |

```bash
curl -X POST "http://localhost:8000/predict?summary=true" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Congratulations! You won a FREE iPhone. Claim it at http://win.example",
                 "Hi team, the meeting has been moved to 3pm."],
       "threshold": 0.5}'
```

Example response (the field `summary` only appears when `summary=true`):

```json
{
  "model_name": "spam-classifier",
  "model_version": "1",
  "threshold": 0.5,
  "predictions": [
    {"prediction_id": 1, "label": "spam", "confidence": 0.99, "spam_probability": 0.99,
     "summary": "The message claims that the recipient has won a free iPhone."},
    {"prediction_id": 2, "label": "not_spam", "confidence": 0.99, "spam_probability": 0.01,
     "summary": "The team meeting has been moved to 3pm."}
  ]
}
```

* Email addresses and international phone numbers are replaced with `xxemail` and `xxphone` before inference.
* Every prediction is stored in the `predictions` table with the anonymized message, the model name and version, the threshold and a UTC timestamp. The `prediction_id` of each result identifies the stored row and is the reference used to submit feedback.
* `confidence` is the probability of the returned label. `spam_probability` is the probability of spam, independent of the threshold.
* Summaries are requested from Gemini first. If Gemini fails or returns an empty summary, the configured cloud service is used as fallback. If no summary can be obtained, `summary` is `null`, the error is logged, and the classification is still returned.
* **Privacy:** Gemini receives the anonymized message, while the cloud service receives the **original** message as JSON `{"text": "..."}` and must return JSON with a non-empty `summary` string. Sensitive data should not be submitted when the cloud service is enabled.

#### Summarization: `POST /summarize`

Summarizes a single message without classifying or storing it. It accepts the query parameter `cloud=true` to use the cloud service directly, and returns `502` when no service produces a summary.

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi team, the project meeting has been moved to Monday at 10. Please bring the report."}'
```

#### Feedback: `POST /feedback`

Stores the label that the user considers correct for a stored prediction. Each prediction accepts **one** feedback entry; a second one returns `409`.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": 2, "corrected_label": "spam"}'
```

```json
{
  "id": 1,
  "prediction_id": 2,
  "created_at": "2026-11-24T21:30:00.123456Z",
  "predicted_label": "not_spam",
  "corrected_label": "spam",
  "model_version": "1",
  "message": "Feedback stored"
}
```

The field `id` is the identifier of the **feedback entry**, used by the administration endpoints `/feedback/{id}`. It is different from `prediction_id`, which identifies the corrected prediction.

#### Administration endpoints (CRUD)

These endpoints read and manage the rows stored in the database. All of them require the header `X-API-Key` (see [Authentication of the administration endpoints](#57-authentication-of-the-administration-endpoints-admin_api_key)).

* **Listing** (`GET /predictions`, `GET /feedback`) returns the rows ordered by id and accepts two query parameters: `limit` (rows to return, 1 to 500, default 100) and `offset` (rows to skip, default 0). For example, `?limit=50&offset=50` returns the second page of 50 rows.
* **Creating and replacing a prediction** (`POST /predictions`, `PUT /predictions/{id}`) require **all** the columns in the body. `PUT` replaces the whole row and keeps its `id` and `created_at`. The text is anonymized before being stored, as in `/predict`.

  ```json
  {
    "text": "WIN a FREE prize now!!!",
    "predicted_label": "spam",
    "spam_probability": 0.97,
    "threshold": 0.5,
    "model_name": "spam-classifier",
    "model_version": "1"
  }
  ```

* **Updating a feedback entry** (`PUT /feedback/{id}`) only changes its label: `{"corrected_label": "not_spam"}`. New feedback entries are created with the public `POST /feedback`.
* **Deleting** returns `204 No Content` with an empty body. Deleting a prediction also deletes its feedback; deleting a feedback entry keeps its prediction, which can then receive a new correction.

```bash
curl http://localhost:8000/predictions?limit=10 -H "X-API-Key: $ADMIN_API_KEY"
curl -X PUT http://localhost:8000/feedback/1 \
  -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"corrected_label": "spam"}'
curl -X DELETE http://localhost:8000/predictions/1 -H "X-API-Key: $ADMIN_API_KEY"
```

In Swagger UI (`/docs`), the administration endpoints are grouped under *Admin: predictions* and *Admin: feedback* and show a padlock. Press **Authorize**, enter the key once in the `AdminApiKey` field and confirm; Swagger UI then sends the header with every request.

### 5.6 Data storage

Predictions and feedback are stored in two PostgreSQL tables, mapped with the SQLAlchemy ORM in `src/database/models.py`. They provide the data for drift monitoring and model retraining.

**`predictions`**: one row per message classified by `/predict` (or created through `POST /predictions`).

| Column | Type | Description |
|---|---|---|
| `id` | integer, primary key | Identifier returned as `prediction_id` |
| `created_at` | timestamp with time zone | UTC time of the prediction |
| `text` | text | Anonymized message |
| `predicted_label` | text | `spam` or `not_spam` (enforced by a `CHECK` constraint) |
| `spam_probability` | float | Probability of spam |
| `threshold` | float | Decision threshold used |
| `model_name` | text | Registered name of the model |
| `model_version` | text | Registered version of the model |

**`feedback`**: the label that the user considers correct for a prediction.

| Column | Type | Description |
|---|---|---|
| `id` | integer, primary key | Identifier of the feedback entry |
| `prediction_id` | integer, foreign key to `predictions.id`, unique | Corrected prediction. The unique constraint allows one feedback entry per prediction |
| `created_at` | timestamp with time zone | UTC time of the correction |
| `corrected_label` | text | `spam` or `not_spam` (enforced by a `CHECK` constraint) |

Indexes on `predictions.created_at` and `predictions.model_version` speed up the queries of drift monitoring, which filter by time period and model version. Messages are always stored anonymized, never as received.

### 5.7 Authentication of the administration endpoints (`ADMIN_API_KEY`)

#### Purpose

The API is publicly accessible and has no user accounts, so anyone who knows its URL can send requests to it. Without protection, any person could delete the stored data, falsify the labels used for retraining, or read every message sent to the API. To prevent this, the administration endpoints require a secret key, the **`ADMIN_API_KEY`**, which works as a password shared by the team. The public endpoints (`/`, `/health`, `/model`, `/predict`, `/summarize` and `POST /feedback`) do not require it, since they are the endpoints used by regular clients.

#### How it works

1. At startup, the API reads the environment variable `ADMIN_API_KEY` (from `.env` locally, or from the *Environment* settings on Render).
2. Every administration endpoint declares the dependency `require_admin_key` (`src/api/api.py`). FastAPI runs it **before** the endpoint itself. The header is declared as the OpenAPI security scheme `AdminApiKey`, which is why the documentation shows the **Authorize** button and a padlock on these endpoints.
3. The dependency reads the request header `X-API-Key` and compares it with `ADMIN_API_KEY` using `secrets.compare_digest`, a comparison that takes the same time whatever the input, so the key cannot be guessed by measuring response times.
4. Depending on the result:

| Situation | Response | Effect |
|---|---|---|
| `ADMIN_API_KEY` is not set on the server | `503 Admin endpoints disabled` | The endpoint is not executed |
| Header `X-API-Key` missing or different from the key | `401 Invalid API key` | The endpoint is not executed and the database is not accessed |
| Header `X-API-Key` equal to the key | The endpoint runs normally | |

If the variable is not configured, the administration endpoints are closed by default, so a missing configuration never leaves the data unprotected.

#### Choosing the key

The key is not issued by any external service: the team chooses it, like a password. It must be long and random, never a guessable value such as `1234` or `admin`. A suitable key is generated with:

```bash
openssl rand -hex 32   # prints 64 random hexadecimal characters
```

#### Configuring the key

**Team convention: one key everywhere.** The team uses a single key, the same in every local `.env` file and on Render, so that there is only one value to know.

1. **Locally:** write the key in the `ADMIN_API_KEY` line of `.env`, without quotes or spaces, and restart the API:
   ```
   ADMIN_API_KEY=9f2c4e7a...   # the 64 characters printed by openssl
   ```
2. **On Render:** open the web service, select *Environment* → *Add Environment Variable*, enter the key `ADMIN_API_KEY` with the same value, and save. Render restarts the service with the new value.
3. **In every request** to an administration endpoint, send the header `X-API-Key: <key>`, both locally and against the public deployment.

#### Security rules

* **Never commit the key** to the repository: not in `env.example`, the README or the source code. Share it with the team through a private channel.
* Since every `.env` file holds the production key, the `.env` file must be treated as a password and never shared, uploaded or shown in screenshots.
* **If the key leaks**, generate a new one and update it on Render and in every `.env` at the same time. The old key stops working as soon as the services restart with the new value, without code changes.

### 5.8 Testing the API with Postman

[Postman](https://www.postman.com/downloads/) is a graphical client to send HTTP requests. This section explains where each value goes and proposes a complete test sequence.

#### Where each value goes

Every request in Postman is configured in the same request window:

| Element | Where it is set in Postman | Example |
|---|---|---|
| HTTP method | Drop-down menu on the left of the URL bar | `GET`, `POST`, `PUT`, `DELETE` |
| URL | URL bar | `http://127.0.0.1:8000/predictions/5` |
| Query parameters | Appended to the URL, or in the *Params* tab | `?limit=100&offset=0`, `?summary=true` |
| Administration key | *Headers* tab: **Key** `X-API-Key`, **Value** the content of `ADMIN_API_KEY`. The checkbox of the row must be ticked | `X-API-Key` → `9f2c4e7a...` |
| Request body | *Body* tab → select **raw** → select **JSON** in the drop-down on the right | `{"corrected_label": "spam"}` |

Important details:

* The header name must be exactly `X-API-Key` (case does not matter). The value is the key as it is, with no quotes and no `Bearer` prefix.
* The body type must be **JSON**. If it is left as *Text*, Postman sends plain text and the API answers `422`.
* Identifiers are numbers and are written without quotes in the body (`"prediction_id": 15`).
* The header is only needed by the administration endpoints. For the public endpoints it can be omitted.

#### Importing the ready-made collection

The repository includes a Postman collection with all the requests of the test sequence below, already configured: [`docs/postman/datatouille_api.postman_collection.json`](docs/postman/datatouille_api.postman_collection.json).

1. In Postman, select **Import** (top left) and choose the file.
2. Open the collection *Datatouille IMAP Spam Detection API*, go to its **Variables** tab and set:
   * `baseUrl`: `http://127.0.0.1:8000` locally, or `https://datatouille-api.onrender.com` for the deployment;
   * `adminKey`: the value of `ADMIN_API_KEY`.

   Save the collection (*Ctrl/Cmd + S*). The key is not stored in the repository: each member sets it in their own Postman.
3. Run the requests one by one in the order of their numbers, or all of them at once with **Run collection** (Collection Runner).

Each request includes a script, in its *Scripts* → *Post-response* tab, that checks the expected status code and stores the identifiers returned by the API in the collection variables `A`, `B`, `C` (predictions), `F` and `G` (feedback). The following requests use them as `{{A}}`, `{{B}}`, and so on, so no identifier has to be copied by hand. The last folder, *Cleanup*, deletes every row created by the collection.

Without the collection, the same requests can be created by hand following the next table, writing the full URLs and copying the identifiers from each response.

#### Test sequence

Start the API (section [5.4](#54-start-the-api)) and run the requests in this order. The identifiers depend on the rows already stored: the placeholders `A`, `B`, `C` (predictions) and `F` (feedback) stand for the values returned by the previous requests, which the collection fills in automatically. Requests marked with 🔒 require the `X-API-Key` header.

| # | Method and URL | Body (raw, JSON) | Expected result |
|---|---|---|---|
| 1 | `GET /` | | `200`, name and version of the API |
| 2 | `GET /health` | | `200`, `"model_loaded": true` |
| 3 | `GET /model` | | `200`, model name, version and metrics |
| 4 | `POST /predict` | `{"texts": ["WIN a FREE iPhone now!!! Claim at http://win.example", "Hi team, the meeting is on Monday. Write to ana@example.com"], "threshold": 0.5}` | `200`, two results. Note their `prediction_id` as **A** and **B** |
| 5 | `POST /summarize` | `{"text": "Hi team, the project meeting has been moved to Monday at 10. Please bring the report."}` | `200`, a `summary` |
| 6 | `POST /feedback` | `{"prediction_id": A, "corrected_label": "not_spam"}` | `201`. Note the `id` of the response as **F** |
| 7 | 🔒 `GET /predictions?limit=100&offset=0` | | `200`, a list that includes A and B |
| 8 | 🔒 `GET /predictions/B` | | `200`, the message of B with `xxemail` instead of the address |
| 9 | 🔒 `POST /predictions` | `{"text": "Manual test from Postman, contact bob@example.com", "predicted_label": "spam", "spam_probability": 0.9, "threshold": 0.5, "model_name": "postman-test", "model_version": "0"}` | `201`. Note the `id` as **C** |
| 10 | 🔒 `PUT /predictions/C` | The body of step 9 with `"predicted_label": "not_spam"` and `"spam_probability": 0.2` | `200`, new values with the same `created_at` |
| 11 | 🔒 `GET /feedback?limit=100&offset=0` | | `200`, a list that includes F |
| 12 | 🔒 `GET /feedback/F` | | `200`, `"corrected_label": "not_spam"` |
| 13 | 🔒 `PUT /feedback/F` | `{"corrected_label": "spam"}` | `200`, `"corrected_label": "spam"` |
| 14 | 🔒 `DELETE /feedback/F` | | `204`, empty body. Prediction A is kept |
| 15 | 🔒 `DELETE /predictions/B` | | `204`, empty body |

Error cases that confirm the validations and the protection:

| # | Request | Expected result |
|---|---|---|
| 16 | `GET /predictions` without the `X-API-Key` header (untick the row) | `401` |
| 17 | `GET /predictions` with `X-API-Key: wrong-key` | `401` |
| 18 | `POST /feedback` with `{"prediction_id": A, "corrected_label": "spam"}`, sent twice | First `201`, then `409` |
| 19 | `POST /feedback` with `{"prediction_id": 999999999, "corrected_label": "spam"}` | `404` |
| 20 | `POST /feedback` with `{"prediction_id": A, "corrected_label": "maybe"}` | `422` |
| 21 | 🔒 `GET /predictions/999999999` | `404` |
| 22 | `POST /predict` with `{"texts": []}` | `422` |
| 23 | 🔒 `DELETE /predictions/A` (A has the feedback of step 18) | `204`, and its feedback is also deleted (a later `GET` of that feedback returns `404`) |

#### Checking the database

Every write can be verified directly in PostgreSQL, from the Render dashboard, a client such as DBeaver, or `psql "$DATABASE_URL"`:

```sql
-- Number of rows before and after the tests
SELECT (SELECT count(*) FROM predictions) AS predictions,
       (SELECT count(*) FROM feedback)    AS feedback;

-- Latest stored predictions (step 4): messages are anonymized and times are in UTC
SELECT id, text, predicted_label, spam_probability, model_version, created_at
FROM predictions ORDER BY id DESC LIMIT 5;

-- Latest feedback entries (steps 6 and 13)
SELECT * FROM feedback ORDER BY id DESC LIMIT 5;

-- Manual row (steps 9 and 10)
SELECT * FROM predictions WHERE model_name = 'postman-test';
```

#### Cleaning up

When the local `.env` points to the production database, every test row is stored with the real data. At the end of the tests, delete the rows created (🔒 `DELETE /predictions/A` and 🔒 `DELETE /predictions/C`) and check that the row counts match those observed before the tests.

### 5.9 OpenAPI specification and Swagger Editor

FastAPI builds an [OpenAPI](https://www.openapis.org/) specification of the API directly from the code: endpoints, parameters, request and response schemas, status codes and the `X-API-Key` security scheme. Besides the live documentation at `/docs` and `/redoc`, the repository keeps an exported copy of the specification in [`docs/docs/swagger_api.json`](docs/docs/swagger_api.json), which can be read without running the API.

The file is generated by [`scripts/generate_openapi.py`](scripts/generate_openapi.py). Run it from the root of the repository, with the virtual environment active, **after every change to the API**:

```bash
python scripts/generate_openapi.py
```

```
OpenAPI specification generated: .../docs/docs/swagger_api.json
  API version:  1.1.0
  Last updated: 2026-11-24T21:34:12+00:00
  Endpoints:    15 operations on 9 paths
```

The script only imports the application, without starting it, so it needs neither the model, the database nor the credentials of `.env`. On every run it:

* rebuilds the specification from the current code, so it always matches the implemented endpoints;
* takes the version from `API_VERSION` in `src/api/settings.py` (increase it when the API changes);
* records the generation time in `info.x-last-updated` and the Git commit in `info.x-git-commit` (with the suffix `-dirty` when there are uncommitted changes), and adds both at the end of the description;
* declares two servers, the production deployment on Render and the local development server.

To explore the specification in the online **Swagger Editor**:

1. Open [https://editor.swagger.io](https://editor.swagger.io).
2. Select *File* → *Import file* and choose `docs/docs/swagger_api.json` (alternatively, paste its content in the left panel).
3. The right panel shows every endpoint with its schemas and examples. To send requests from the editor, choose a server in the *Servers* drop-down; the local server requires the API to be running. For the administration endpoints, press **Authorize**, enter the key in the `AdminApiKey` field and confirm.

Regenerate and commit the file together with the changes of the API, so that the exported specification never becomes outdated.

### 5.10 Status codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `201` | A row was created (`POST /feedback`, `POST /predictions`) |
| `204` | A row was deleted; the response has no body |
| `400` | Invalid value detected while processing the request |
| `401` | Missing or wrong `X-API-Key` header on an administration endpoint |
| `404` | The prediction or the feedback entry does not exist |
| `409` | The prediction already has feedback |
| `422` | The request does not match the schema: body not sent as JSON, missing field, empty or too long message, unknown label, threshold or probability outside [0, 1]. The response body indicates the field (`loc`) and the reason (`msg`) |
| `500` | Unexpected error, logged on the server without exposing its details |
| `502` | `/summarize` could not obtain a summary from any service |
| `503` | Model or database not available, or administration endpoints disabled because `ADMIN_API_KEY` is not set |

### 5.11 Troubleshooting

| Symptom | Cause and solution |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'`, with paths of another Python version in the traceback | The `uvicorn` of another Python installation was executed. Activate the virtual environment, install the dependencies with `python -m pip install -r requirements.txt` and start the API with `python -m uvicorn ...` |
| `pip install` fails with dependency conflicts | The environment uses a Python version other than 3.13. Recreate it with `python3.13 -m venv` |
| Startup stops after `PROJ_ROOT path is: ...` and never completes | The virtual environment is inside a cloud-synchronized folder and its files are not available locally. Recreate it outside that folder (see section [5.1](#51-install-the-dependencies)) |
| The API does not start: `DATABASE_URL environment variable is not configured` | Add `DATABASE_URL` to `.env` |
| `401` on an administration endpoint | The `X-API-Key` header is missing, unticked in Postman, or its value differs from `ADMIN_API_KEY` |
| `503 Admin endpoints disabled` | `ADMIN_API_KEY` is not set. Add it to `.env` (or to Render) and restart the API |
| `422` on a request with a body | In Postman, the body is not set to *raw* → *JSON*, or a field is missing or invalid. Read the `detail` of the response |
| `405 Method Not Allowed` or an unexpected `422` on a listing | The wrong HTTP method was selected, for example `POST /predictions` instead of `GET /predictions` |

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

The API described in [section 5](#5-api) is deployed on Render as a publicly accessible web service, so that external applications can classify and summarize messages without running the model locally. Unlike the Virtech service of [section 6](#6-cloud-summarization-service-upc-virtech-and-ngrok), which is dedicated to summarization, the Render deployment hosts the main application and coordinates the MLflow Model Registry, the PostgreSQL database, Google Gemini and the optional cloud summarization service.

| Resource | URL |
|---|---|
| Public API | **https://datatouille-api.onrender.com** |
| Interactive documentation (Swagger UI) | **https://datatouille-api.onrender.com/docs** |

### Deployment configuration

The web service is connected to the GitHub repository and is redeployed automatically when its branch receives new commits. It uses Python `3.13.9`, configured through `.python-version`, and installs only the dependencies required for serving, listed in `requirements-render.txt`:

```bash
pip install -r requirements-render.txt                 # build command
uvicorn src.api.api:app --host 0.0.0.0 --port $PORT    # start command
```

The server listens on the port assigned by Render, which provides the public HTTPS endpoint. This removes the need to expose a local development server or to maintain an external tunnel for the main API.

### Environment configuration

The deployment reads the same environment variables as a local installation (see [section 5.2](#52-set-the-environment-variables)). They are defined in the *Environment* settings of the Render service, never in the repository:

| Variable | Notes for the deployment |
|---|---|
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Access to the DagsHub Model Registry |
| `MLFLOW_MODEL_NAME`, `MLFLOW_MODEL_ALIAS` | Optional; default to `spam-classifier` and `production` |
| `DATABASE_URL` | **Required.** When the database is hosted on Render in the same region, use its *Internal Database URL*, which keeps the traffic inside Render's private network |
| `ADMIN_API_KEY` | Required to enable the administration endpoints. Same value as in the local `.env` files (see [Configuring the key](#configuring-the-key)) |
| `GEMINI_API_KEY`, `CLOUD_SUMMARIZATION_URL` | Summarization backends |

Saving a change in the *Environment* settings restarts the service with the new values.

### Using the public deployment

The public deployment exposes exactly the endpoints of [section 5.5](#55-endpoint-reference), with the same request bodies, responses and status codes. Every example of this README, including the Postman collection of [section 5.8](#58-testing-the-api-with-postman), works against the deployment by replacing `http://127.0.0.1:8000` with `https://datatouille-api.onrender.com`. The administration endpoints then require the key configured on Render.

```bash
curl https://datatouille-api.onrender.com/health

curl -X POST "https://datatouille-api.onrender.com/predict?summary=true" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Congratulations! You have won a free iPhone!",
                 "Hi, the project meeting is scheduled for tomorrow."],
       "threshold": 0.5}'
```

### Deploying a new version

1. Merge the changes into the branch deployed by Render. Render builds and starts the new version automatically.
2. Check the logs of the service in the Render dashboard. The deployment is ready when they show `Loaded model ...`, `Prediction database (PostgreSQL) ready` and `Application startup complete`.
3. Verify the service with `GET /health`, and the protection of the administration endpoints with `GET /predictions` without the key (`401`) and with it (`200`).
4. If the endpoints changed, regenerate the OpenAPI specification (see [section 5.9](#59-openapi-specification-and-swagger-editor)).

### Storage and free-tier limitations

The predictions and the feedback are stored in a PostgreSQL database, which is a separate service from the API. The records are therefore kept when the API restarts, redeploys or is suspended due to inactivity, and the tables are created automatically at startup if they do not exist.

**Database limitation:** Render's free PostgreSQL databases expire some time after their creation and have no backups. Check the expiration date in the Render dashboard, and move to a paid plan or another provider before relying on the data. Only `DATABASE_URL` needs to change.

Render's free tier also suspends inactive web services. When a new request arrives, the application must restart and reload the production model from MLflow, so the first request can take from 30 seconds to one minute. CPU and memory resources are limited, so this deployment is intended for development, integration testing and demonstrations rather than for workloads that require guaranteed availability.

## 8. CO2 experiment

`src/modeling/measure_co2.py` measures the training energy of each model outside the DVC pipeline. It repeats 5-fold CV plus a final fit with fixed hyperparameters 5 times per model, interleaving the models and sampling power every 1 s, because one training lasts only a few seconds. Results are stored in `reports/emissions/co2_runs.jsonl`, together with the hardware that produced them.

```bash
python -m src.modeling.measure_co2 --train data/dataset/train.csv \
    --out reports/emissions/co2_runs.jsonl --reps 5
```

The pipeline itself samples every 15 s, so per-trial emissions in MLflow rely on very few readings and are indicative only.

## 9. Quality checks (same as CI)

```bash
ruff check src tests
ruff format --check src tests
mypy src            # config in [tool.mypy] in pyproject.toml
```

## 10. CI

CI runs the pipeline-level checks via `.github/workflows/ci-pipeline.yml`, with test scope depending on the merge level:

| Merge                  | Job                             | Runs                                                                          |
|------------------------|---------------------------------|-------------------------------------------------------------------------------|
| Task branch → `dev`    | Fast Feedback & Code Integration| Ruff, mypy, hermetic unit + integration tests, coverage gate                      |
| `dev` → `main`         | Comprehensive Pre-Production Validation | Enforces source branch is `dev`; re-runs Ruff + mypy; `dvc pull`; full-file GE validation; regression loop for the 4 models; full unit + integration suite |

## 11. Project organization

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
│   ├── docs/             <- Cards, guides and the exported OpenAPI specification (swagger_api.json)
│   ├── edn/              <- Engineering Decision Notebook entries
│   ├── postman/          <- Postman collection of every API endpoint
│   └── proposals/        <- Design proposals of each component
├── gx/                   <- Great Expectations suite and checkpoint
├── infra/mlflow/         <- Docker Compose setup of the MLflow server
├── models/               <- Trained models, versioned with DVC
├── notebooks/            <- Exploration notebooks.
├── reports/
│   ├── emissions/        <- CodeCarbon measurements
│   ├── features/         <- Top-20 influential features of each model
│   └── metrics/          <- Training and test metrics of each model
├── scripts/              <- Helper scripts (OpenAPI export, synthetic data generation)
├── src/
│   ├── api/              <- Web API: inference, summaries, feedback and administration endpoints
│   ├── data/             <- Download, cleaning, validation, and splitting
│   ├── database/         <- PostgreSQL connection, ORM models and storage of predictions and feedback
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
├── env.example           <- Environment variables template (MLflow, database, API keys)
├── LICENSE
├── params.yaml
├── pytest.ini
├── pyproject.toml
└── requirements.txt
```

## 12. Documentation & pointing docs

* [`CONTRIBUTING.md`](CONTRIBUTING.md) — **How to contribute**: branching strategy (task branches -> `dev` -> `main`), commit style, DVC data hygiene, CI gates per merge, and code good practices.
* [`AGENTS.md`](AGENTS.md) — **Rules for AI coding agents** working in this repo: sources of truth, layout roles, permitted edits, and validation workflow.
* `docs/docs/dataset_card.md` — Dataset provenance and description.
* `docs/docs/experiment-tracking.md` — How MLflow runs are structured and how to restore an experiment.
* [`docs/docs/swagger_api.json`](docs/docs/swagger_api.json) — OpenAPI specification of the API, generated with `python scripts/generate_openapi.py` (see [section 5.9](#59-openapi-specification-and-swagger-editor)).
* [`docs/postman/datatouille_api.postman_collection.json`](docs/postman/datatouille_api.postman_collection.json) — Postman collection of every endpoint (see [section 5.8](#58-testing-the-api-with-postman)).
* `docs/edn/` — Engineering Decision Notebooks for architecture decisions.
