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

import src.config  # noqa: F401  (loads .env before the variables below are read)

# MLflow Model Registry: model served by the API
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME") or "spam-classifier"
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS") or "production"

# API
API_TITLE = "IMAP Spam Detection API"
API_VERSION = "1.1.0"
MAX_TEXT_LENGTH = 10_000  # Characters per message, far above a typical email body
MAX_BATCH_SIZE = 100  # Messages per prediction request
DEFAULT_SPAM_THRESHOLD = 0.5  # Spam probability from which a message is labelled as spam
CLOUD_SUMMARIZATION_URL = os.getenv("CLOUD_SUMMARIZATION_URL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
CLOUD_CALLS_TIMEOUT = 30.0

# Key required in the X-API-Key header by the CRUD endpoints of /predictions and /feedback.
# When it is not set, those endpoints are disabled (503).
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
