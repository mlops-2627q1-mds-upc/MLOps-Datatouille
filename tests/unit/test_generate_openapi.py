"""Unit tests for the export of the OpenAPI specification (``scripts/generate_openapi.py``)."""

from datetime import datetime
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from src.api.settings import API_VERSION

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_openapi.py"


def load_script() -> ModuleType:
    """Import the script as a module, since ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("generate_openapi", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_specification_is_current(tmp_path: Path) -> None:
    """Test of the exported file: current version, servers, endpoints and update time."""
    output = load_script().generate_openapi(tmp_path / "docs" / "swagger_api.json")
    schema = json.loads(output.read_text(encoding="utf-8"))

    assert schema["openapi"].startswith("3.")
    assert schema["info"]["version"] == API_VERSION
    assert datetime.fromisoformat(schema["info"]["x-last-updated"]).tzinfo is not None
    assert f"Version {API_VERSION}" in schema["info"]["description"]
    assert [server["url"] for server in schema["servers"]] == [
        "https://datatouille-api.onrender.com",
        "http://127.0.0.1:8000",
    ]
    assert {"/predict", "/summarize", "/feedback", "/predictions"} <= set(schema["paths"])
