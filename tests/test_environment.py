import os
import sys
import pytest

def test_python_version():
    """Ensure the pipeline is running on Python 3.10+"""
    assert sys.version_info >= (3, 10), "Python version must be 3.10 or higher"

def test_project_structure_exists():
    """Verify that the Cookiecutter Data Science structure is intact"""
    required_directories = [
        "models",
        "src"
    ]
    for directory in required_directories:
        assert os.path.isdir(directory), f"Required directory '{directory}' is missing."
