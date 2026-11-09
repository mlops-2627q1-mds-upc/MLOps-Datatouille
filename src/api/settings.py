"""
Settings of the model serving: the API and the registration of the served model.

These settings are kept apart from ``src.config`` on purpose. The training stages of the DVC
pipeline depend on ``src/config.py``, so any change to that file marks the trained models as
outdated and makes ``dvc repro`` retrain them. Serving settings do not affect training, so
changing them here leaves the pipeline untouched.

Values that may differ between environments are read from ``.env`` (loaded by ``src.config``).
Empty values are treated as unset, so the defaults apply.
"""

import os
from pathlib import Path

from src.config import DATA_DIR

# MLflow Model Registry: model served by the API
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME") or "spam-classifier"
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS") or "production"

# API
API_TITLE = "IMAP Spam Detection API"
API_VERSION = "1.0.0"
MAX_TEXT_LENGTH = 10_000  # Characters per message, far above a typical email body
MAX_BATCH_SIZE = 100  # Messages per prediction request
DEFAULT_SPAM_THRESHOLD = 0.5  # Spam probability from which a message is labelled as spam

# Summarization with Google Gemini (disabled when GEMINI_API_KEY is not set)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
SUMMARY_TIMEOUT_SECONDS = 30.0  # Per Gemini request, so a slow answer cannot block a worker

# Predictions served and feedback collected by the API (SQLite, never versioned with Git)
PREDICTIONS_DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH") or DATA_DIR / "api" / "predictions.db")
