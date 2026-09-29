import os
import sys
import pytest

def test_python_version():
    """Ensure the pipeline is running on Python 3.10+"""
    assert sys.version_info >= (3, 10), "Python version must be 3.10 or higher"

def test_project_structure_exists():
    """Verify that the Cookiecutter Data Science structure is intact"""
    required_directories = [
        "data/raw",
        "data/processed",
        "models",
        "src"
    ]
    for directory in required_directories:
        assert os.path.isdir(directory), f"Required directory '{directory}' is missing."

def test_source_module_imports():
    """Verify that the main source modules can be imported without syntax errors"""
    try:
        import src.config
        import src.dataset
        import src.features
    except ImportError as e:
        pytest.fail(f"Failed to import core source modules: {e}")