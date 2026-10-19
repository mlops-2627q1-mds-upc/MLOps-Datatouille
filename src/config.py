import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Configure logging before any module imports this file: basicConfig is a no-op
# if a root handler already exists, so this becomes the app-wide default.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info("PROJ_ROOT path is: %s", PROJ_ROOT)

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EMISSIONS_DIR = REPORTS_DIR / "emissions"

# CodeCarbon (energy and CO2 tracking)
CODECARBON_PROJECT_NAME = "mlops-datatouille"
CODECARBON_COUNTRY_ISO_CODE = "ESP"  # Grid carbon intensity of Spain (offline mode)
CODECARBON_MEASURE_POWER_SECS = 15  # Sampling interval for power readings
CODECARBON_TRACKING_MODE = "process"  # Measure only this Python process, not the whole machine

# MLflow (tracking URI and credentials are read by mlflow from .env)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "spam-classifier")
