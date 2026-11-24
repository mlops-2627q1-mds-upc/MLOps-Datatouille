"""
Generate the OpenAPI specification of the API from the FastAPI application.

The specification is written to ``docs/docs/swagger_api.json`` and can be opened in any
OpenAPI tool, such as the online Swagger Editor (https://editor.swagger.io, File > Import file).
It is built from the code itself, so it always matches the current endpoints, schemas and
version of the API. Each run also records when and from which commit it was generated.

Run it from the root of the repository after any change to the API:

    python scripts/generate_openapi.py

The application is only imported, not started: neither the model, the database nor the
credentials of ``.env`` are needed.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.api import app  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "docs" / "docs" / "swagger_api.json"
SERVERS = [
    {
        "url": "https://datatouille-api.onrender.com",
        "description": "Production API hosted on Render",
    },
    {"url": "http://127.0.0.1:8000", "description": "Local development server"},
]


def current_commit() -> str | None:
    """
    Return the short hash of the current Git commit.

    Returns:
        The hash, with ``-dirty`` appended if there are uncommitted changes, or ``None`` if Git
        is not available.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if changes else commit


def build_schema() -> dict[str, Any]:
    """
    Build the OpenAPI schema of the application, with servers and generation metadata.

    Returns:
        The OpenAPI schema as a dictionary.
    """
    app.openapi_schema = None  # rebuild it from the current code instead of a cached copy
    schema = app.openapi()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0)
    commit = current_commit()

    schema["servers"] = SERVERS
    info = schema["info"]
    info["x-last-updated"] = generated_at.isoformat()
    if commit:
        info["x-git-commit"] = commit
    origin = f", commit `{commit}`" if commit else ""
    info["description"] = (
        f"{info.get('description', '').rstrip()}\n\n---\n\n"
        f"*Version {info['version']}. Specification generated on "
        f"{generated_at:%Y-%m-%d %H:%M} UTC{origin}.*\n"
    )
    return schema


def generate_openapi(output_path: Path = OUTPUT_PATH) -> Path:
    """
    Export the OpenAPI schema of the application to a JSON file.

    Args:
        output_path: File to write. Its parent directories are created if needed.

    Returns:
        The path of the written file.
    """
    schema = build_schema()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
    output_path.write_text(content, encoding="utf-8")

    endpoints = sum(len(methods) for methods in schema["paths"].values())
    print(f"OpenAPI specification generated: {output_path}")
    print(f"  API version:  {schema['info']['version']}")
    print(f"  Last updated: {schema['info']['x-last-updated']}")
    print(f"  Endpoints:    {endpoints} operations on {len(schema['paths'])} paths")
    return output_path


if __name__ == "__main__":
    generate_openapi()
