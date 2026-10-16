# Spam Detection ML System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Spam detection ML system with spam detection, summarization, and automated feedback loops.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         message_guard and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Data ingestion pipeline
    │
    ├── features.py             <- Data transformations and feature engineering
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Inference wrappers for model serving          
    │   └── train.py            <- Training loop with MLflow tracking and CodeCarbon
    │
    ├── api/                    <- Web API for inference and feedback collection
    │
    └── monitoring/             <- Data drift detection and performance monitoring
```

--------

## Model tests and MLflow

Unit tests for preprocessing, data splitting, and anonymization are grouped in
`tests/unit/`. Pipeline and Great Expectations tests are grouped in
`tests/integration/`. Run the complete test tree separately with `pytest -q tests`.
Pytest tests are not part of the DVC pipeline.

Phase-one cleaning can also anonymize contact details before count features and text
normalization. Email addresses are replaced with `xxemail`, and phone numbers in
international format (starting with `+` and a country code) are replaced with `xxphone`,
using the regular expressions defined in `src/data/preprocess.py`. Local numbers without
a country code are not detected. This is enabled for the DVC `clean` stage by
`data.anonymize: true` in `params.yaml`.
The anonymization unit tests are in `tests/unit/test_anonymization.py`, and the
phase-one integration test is in `tests/integration/test_preprocess_integration.py`.
Run these tests separately with pytest.

Great Expectations itself is configured under `gx/` and
`src/data/gx_context_configuration.py`. The DVC `validate_processed` stage runs its
checkpoint against the cleaned dataset and writes the validation report.
The checkpoint checks the exact column names/order, non-empty text columns,
string types for text and labels, the allowed label values, non-null and
non-negative integer count features, and the absence of email addresses and
phone numbers in the text column.

Separately, `tests/integration/gx_expectations_test.py` contains **pytest tests for Great
Expectations**. These tests check the existing rules against the real processed
dataset and verify that they accept valid sample data and reject invalid labels,
null cleaned text, negative count features, and text containing email addresses
or phone numbers. They test Great Expectations; they
do not replace or redefine its configuration. They are standalone tests and are
not part of the DVC pipeline. Run them separately with
`pytest -q tests/integration/gx_expectations_test.py`.

Model regression tests are intentionally **outside the DVC pipeline** and are
stored in `tests/regression/`. Run the DVC pipeline first so the trained models
and held-out test split exist, then run the checks independently for every model:

```bash
for model in logreg naive_bayes linear_svm random_forest; do
  MODEL_NAME="$model" pytest -q tests/regression/test_model.py || exit 1
done
```

The tests load each trained scikit-learn model and score it against every
example and label in the held-out test split. Accuracy, macro precision, macro
recall, and macro F1 must each be at least `0.95`; they use the project's CSV
data, not a Transformers pipeline. To test only one model, set `MODEL_NAME`
explicitly, for example `MODEL_NAME=logreg pytest -v tests/regression/test_model.py`.

To log training parameters, cross-validation metrics, the trained model, and
its registration to the local MLflow server, start the existing local stack and
set the tracking URI before running DVC:

```bash
MLFLOW_PORT=5050 \
MLFLOW_BACKEND_STORE_URI='postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow' \
  docker compose --env-file infra/mlflow/env.example \
  -f infra/mlflow/docker-compose.yaml up -d
export MLFLOW_TRACKING_URI=http://localhost:5050

# Re-train and evaluate each model; run the separate pytest checks afterward
dvc repro --force --downstream train
```

Each trained model is registered as `spam-<model-name>` in the `spam-classifier`
experiment. Runs also record the training dataset SHA-256, Git commit, and whether the
working tree had uncommitted changes. If
`MLFLOW_TRACKING_URI` is unset, training continues locally and reports that
tracking is disabled; if it is set but the server is unreachable, training
fails rather than silently discarding the run.

Open <http://localhost:5050> to compare the runs and inspect their metrics.
Use `dvc repro --force` to force the entire pipeline, including data processing,
training, and evaluation. DVC runs the Great Expectations checkpoint to validate
the cleaned dataset, but does not run pytest tests. Run the unit, integration,
and model regression tests separately with pytest.
