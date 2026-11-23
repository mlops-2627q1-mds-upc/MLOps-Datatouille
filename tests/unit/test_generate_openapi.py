"""
Unit tests for ``scripts/generate_openapi.py`` and the OpenAPI documentation of the API.

The script is loaded from its file because ``scripts/`` is not a package. Building the schema
does not start the application, so no model is needed.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.api.settings import API_LOCAL_URL, API_PUBLIC_URL, OPENAPI_SPEC_PATH
from src.config import PROJ_ROOT

SCRIPT_PATH = PROJ_ROOT / "scripts" / "generate_openapi.py"
TEST_PUBLIC_URL = "https://api.example"
TEST_LOCAL_URL = "http://localhost:9000"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The generator script, imported as a module."""
    spec = spec_from_file_location("generate_openapi", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spec(script: ModuleType) -> dict[str, Any]:
    """OpenAPI specification built with test server URLs."""
    return script.build_spec(TEST_PUBLIC_URL, TEST_LOCAL_URL)


def test_build_spec_lists_servers(spec: dict[str, Any]) -> None:
    """Test that the public and the local servers are listed, public first."""
    assert [server["url"] for server in spec["servers"]] == [TEST_PUBLIC_URL, TEST_LOCAL_URL]


def test_build_spec_documents_every_endpoint(spec: dict[str, Any]) -> None:
    """Test that every endpoint has a summary, a description and a readable operation ID."""
    expected = {
        ("/", "get"): "root",
        ("/health", "get"): "health",
        ("/model", "get"): "model_info",
        ("/predict", "post"): "predict",
        ("/feedback", "post"): "feedback",
    }
    operations = {
        (path, method): operation
        for path, methods in spec["paths"].items()
        for method, operation in methods.items()
    }
    assert set(operations) == set(expected)
    for key, operation in operations.items():
        assert operation["operationId"] == expected[key]
        assert operation["summary"]
        assert operation["description"]
        assert "Args:" not in operation["description"]


def test_build_spec_documents_error_responses(spec: dict[str, Any]) -> None:
    """Test that the error responses use the shared error schema."""
    error_ref = "#/components/schemas/ErrorResponse"
    expected_errors = {
        ("/model", "get"): {"500", "503"},
        ("/predict", "post"): {"400", "500", "503"},
        ("/feedback", "post"): {"404", "409", "500", "503"},
    }
    for (path, method), codes in expected_errors.items():
        responses = spec["paths"][path][method]["responses"]
        for code in codes:
            schema = responses[code]["content"]["application/json"]["schema"]
            assert schema["$ref"] == error_ref


def test_build_spec_includes_examples(spec: dict[str, Any]) -> None:
    """Test that the request and response schemas carry examples."""
    schemas = spec["components"]["schemas"]
    for name in (
        "PredictionRequest",
        "PredictionResponse",
        "FeedbackRequest",
        "FeedbackResponse",
        "ModelInfoResponse",
        "HealthResponse",
        "RootResponse",
        "ErrorResponse",
    ):
        assert schemas[name]["examples"], name


def test_build_spec_does_not_change_live_schema(script: ModuleType) -> None:
    """Test that the servers are added to the export only, not to the live ``/docs``."""
    script.build_spec(TEST_PUBLIC_URL, TEST_LOCAL_URL)
    assert "servers" not in script.app.openapi()


@pytest.mark.parametrize(
    ("spec_dict", "message"),
    [
        ({"openapi": "3.1.0", "info": {}, "paths": {}}, "missing: components"),
        ({"openapi": "3.1.0", "info": {}, "paths": {}, "components": {}}, "no paths"),
    ],
)
def test_validate_spec_rejects_incomplete_spec(
    script: ModuleType, spec_dict: dict[str, Any], message: str
) -> None:
    """Test that an incomplete specification is rejected."""
    with pytest.raises(ValueError, match=message):
        script.validate_spec(spec_dict)


def test_main_writes_file_and_check_detects_changes(script: ModuleType, tmp_path: Path) -> None:
    """Test the write mode and the check mode on a fresh, then on an edited file."""
    output = tmp_path / "nested" / "swagger.json"
    args = ["--output", str(output), "--server-url", TEST_PUBLIC_URL]

    assert script.main([*args, "--check"]) == 1
    assert script.main(args) == 0
    assert output.read_text(encoding="utf-8").endswith("}\n")
    assert script.main([*args, "--check"]) == 0

    output.write_text("{}\n", encoding="utf-8")
    assert script.main([*args, "--check"]) == 1


def test_committed_spec_is_up_to_date(script: ModuleType) -> None:
    """Test that the committed specification matches the code. Regenerate it if this fails."""
    expected = script.render_spec(script.build_spec(API_PUBLIC_URL, API_LOCAL_URL))
    assert OPENAPI_SPEC_PATH.read_text(encoding="utf-8") == expected, (
        "Outdated OpenAPI specification. Run: python3 scripts/generate_openapi.py"
    )
