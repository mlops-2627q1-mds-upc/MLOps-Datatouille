# Local MLflow

Requirements: Docker Desktop / Docker Engine with Compose v2 (`docker compose`).

## Quick start

```bash
cd docker/mlflow

cp env.example .env

docker compose --env-file .env up -d
```

## Log your first run

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("spam-classifier")

with mlflow.start_run(run_name="logreg-tfidf"):
    mlflow.log_params({"model": "logistic-regression", "ngram_max": 2})
    mlflow.log_metric("f1", 0.93)
    mlflow.log_artifact("metrics_report.png")
```

From the shell, the equivalent is an env var plus the CLI:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow experiments search
```

Note that clients on your host need the `mlflow` package (`pip install mlflow`); the server
image is separate and only has to run in Docker.


## Configuration

See `env.example` for the full list. Every value has a working default, so no setup is required.
Override by exporting the variable or putting it in a `.env` file next to the compose file
(already gitignored).

### Postgres

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER` | `mlflow` | metadata DB user |
| `POSTGRES_PASSWORD` | `mlflow` | metadata DB password |
| `POSTGRES_DB` | `mlflow` | metadata DB name |
| `PGPORT` | `5432` | host port published for Postgres |

### Object store (RustFS)

| Variable | Default | Purpose |
|---|---|---|
| `RUSTFS_PORT` | `9000` | port for the S3 API (host and container) |
| `RUSTFS_CONSOLE_ENABLE` | `true` | serve the web console on 9001 |
| `S3_BUCKET` | `mlflow` | artifact bucket name, created by `create-bucket` |
| `AWS_ACCESS_KEY_ID` | `s3admin` | object store access key |
| `AWS_SECRET_ACCESS_KEY` | `s3admin` | object store secret key |
| `AWS_DEFAULT_REGION` | `us-east-1` | S3 client region |

### MLflow

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_VERSION` | `latest` | tag of the `ghcr.io/mlflow/mlflow` image |
| `MLFLOW_HOST` | `0.0.0.0` | bind address inside the container |
| `MLFLOW_PORT` | `5000` | host port for the MLflow UI + REST API |
| `MLFLOW_BACKEND_STORE_URI` | `postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:${PGPORT}/${POSTGRES_DB}` | metadata store DSN, composed from the Postgres vars |
| `MLFLOW_ARTIFACTS_DESTINATION` | `s3://${S3_BUCKET}` | artifact destination URI for new runs |
| `MLFLOW_S3_ENDPOINT_URL` | `http://storage:${RUSTFS_PORT}` | S3 endpoint used by the server and `create-bucket` |

## Access Links 

| Component | URL | Login |
|---|---|---|
| MLflow UI | <http://localhost:5000> | none |
| RustFS web console | <http://localhost:9001> | `s3admin` / `s3admin` |

Ports follow the `MLFLOW_PORT` / `RUSTFS_PORT` configuration above; the console is only served
when `RUSTFS_CONSOLE_ENABLE=true` and is always on 9001.

## Troubleshooting

If new troubles are found, add them to the table with a short description of the corresponding fix:

| Symptom | Fix |
|---|---|
| `bind: address already in use` on port *N* | Change port number in the .env file |

