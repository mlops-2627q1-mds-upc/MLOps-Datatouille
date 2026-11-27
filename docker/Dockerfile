# Build from the repo root: docker build -f docker/Dockerfile -t imap-api .
# API image. The model is NOT baked in: the API loads models:/<name>@<alias> from the MLflow
# Registry at startup. Secrets come from `--env-file .env` at runtime.
# The Python version must match the one that trained the pickled model.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY docker/requirements-api.txt requirements-api.txt
RUN pip install -r requirements-api.txt

COPY src/ src/

# SQLite directory (mounted as a named volume at runtime). Runs as root by decision of the team.
RUN mkdir -p /app/data/api

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]

# Single worker: SQLite file and one copy of the model in memory.
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
