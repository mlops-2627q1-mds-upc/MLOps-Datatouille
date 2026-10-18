# Spam Detection ML System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Spam detection ML system with spam detection, summarization, and automated feedback loops.

## Project Organization

```
├── .github/              <- Issue templates and CI workflow
├── conf/                 <- Hydra configuration of the training stage
│   ├── model/            <- Hyperparameter search space of each model
│   └── train.yaml
├── data/                 <- Versioned with DVC
│   ├── dataset/          <- Training and test splits
│   ├── processed/        <- Cleaned dataset and its validation report
│   └── raw/              <- Dataset as downloaded from Hugging Face
├── docs/                 <- A default mkdocs project
│   ├── docs/             <- Dataset card, model card, and experiment tracking guide
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
├── src/
│   ├── api/              <- Web API for inference and feedback collection (Reserved for Milestone 4)
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
├── LICENSE
├── params.yaml
├── pyproject.toml
├── README.md
└── requirements.txt
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
  MODEL_NAME="$model" pytest -q tests/regression/ || exit 1
done
```

`tests/regression/` now contains test_model.py (≥ 0.95 accuracy and macro precision/recall/F1) and test_robustness.py (macro F1 ≥ 0.90 and ≥ 90% unchanged predictions on the perturbed test split, and spam score not decreasing for ≥ 90% of legitimate messages after appending a spam phrase). Both run with the same MODEL_NAME loop. To test only one model, set MODEL_NAME explicitly, for example MODEL_NAME=logreg pytest -v tests/regression/test_model.py.

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

Each model logs a child run in the spam-classifier experiment, with one nested run per Optuna trial. Runs record the hyperparameters, the cross-validation and test metrics, the energy consumption, the 20 most influential features, and the training and test splits identified by the MD5 hash of the file (the same hash DVC stores in dvc.lock). When the pipeline is run with dvc exp run, runs are grouped under one parent run per experiment and record the experiment name and its base commit, together with the commands to restore it (see docs/docs/experiment-tracking.md).

Open http://localhost:5050 to compare the runs and inspect their metrics.
Use `dvc repro --force` to force the entire pipeline, including data processing,
training, and evaluation. DVC runs the Great Expectations checkpoint to validate
the cleaned dataset, but does not run pytest tests. Run the unit, integration,
and model regression tests separately with pytest.
