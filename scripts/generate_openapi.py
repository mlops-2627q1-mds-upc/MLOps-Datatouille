"""Generate the OpenAPI specification from the FastAPI application."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.api import app  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "docs" / "docs" / "swagger_api.json"
PRODUCTION_URL = "https://datatouille-api.onrender.com"


def generate_openapi() -> None:
    """Export the FastAPI OpenAPI schema to a JSON file."""

    schema = app.openapi()
    schema["servers"] = [
        {
            "url": PRODUCTION_URL,
            "description": "Production API hosted on Render",
        }
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"OpenAPI specification generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_openapi()
