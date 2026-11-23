"""
Generate the OpenAPI (Swagger) specification of the API from the FastAPI application.

The documentation lives in the code: ``src/api/api.py`` (description, endpoints, errors) and
``src/api/schemas.py`` (fields and examples). This script only exports it, so the JSON file is
never edited by hand. The production model is not loaded: building the schema does not start
the application.

Usage (from any directory):
    python3 scripts/generate_openapi.py            # write the specification
    python3 scripts/generate_openapi.py --check    # fail if the committed file is outdated
    python3 scripts/generate_openapi.py --output other.json --server-url https://my.host
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.api import app  # noqa: E402
from src.api.settings import API_LOCAL_URL, API_PUBLIC_URL, OPENAPI_SPEC_PATH  # noqa: E402

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("openapi", "info", "paths", "components")


def build_spec(public_url: str, local_url: str) -> dict[str, Any]:
    """
    Build the OpenAPI specification of the application with its servers.

    The servers are added only to the exported file. The live ``/docs`` keeps pointing to the
    host that serves it.

    Args:
        public_url: URL of the public deployment, listed first.
        local_url: URL of a local ``uvicorn`` server.

    Returns:
        The OpenAPI specification as a dictionary.
    """
    spec = dict(app.openapi())
    spec["servers"] = [
        {"url": public_url, "description": "Public deployment"},
        {"url": local_url, "description": "Local server (uvicorn src.api.api:app)"},
    ]
    return spec


def validate_spec(spec: dict[str, Any]) -> None:
    """
    Check that the specification has the sections every OpenAPI document needs.

    Args:
        spec: OpenAPI specification.

    Raises:
        ValueError: If a required section is missing or the API has no paths.
    """
    missing = [key for key in REQUIRED_KEYS if key not in spec]
    if missing:
        raise ValueError(f"OpenAPI specification is missing: {', '.join(missing)}")
    if not spec["paths"]:
        raise ValueError("OpenAPI specification has no paths")


def render_spec(spec: dict[str, Any]) -> str:
    """
    Serialize the specification as stable, readable JSON.

    Args:
        spec: OpenAPI specification.

    Returns:
        The JSON text, indented, with a trailing newline.
    """
    return json.dumps(spec, indent=2, ensure_ascii=False) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Args:
        argv: Arguments to parse. ``None`` reads them from ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=OPENAPI_SPEC_PATH,
        help=f"JSON file to write (default: {OPENAPI_SPEC_PATH.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--server-url",
        default=API_PUBLIC_URL,
        help=f"URL of the public deployment (default: {API_PUBLIC_URL}).",
    )
    parser.add_argument(
        "--local-url",
        default=API_LOCAL_URL,
        help=f"URL of the local server (default: {API_LOCAL_URL}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write. Exit with code 1 if the output file is missing or outdated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Generate the specification, or check that the committed one is up to date.

    Args:
        argv: Command-line arguments. ``None`` reads them from ``sys.argv``.

    Returns:
        The exit code: 0 on success, 1 if ``--check`` finds an outdated file.
    """
    args = parse_args(argv)
    spec = build_spec(args.server_url, args.local_url)
    validate_spec(spec)
    content = render_spec(spec)
    output: Path = args.output

    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else None
        if current != content:
            logger.error(
                "%s is outdated. Run: python3 scripts/generate_openapi.py", output.as_posix()
            )
            return 1
        logger.info("%s is up to date", output.as_posix())
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    logger.info(
        "OpenAPI %s specification written to %s (%d paths, %d schemas)",
        spec["openapi"],
        output.as_posix(),
        len(spec["paths"]),
        len(spec["components"].get("schemas", {})),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
