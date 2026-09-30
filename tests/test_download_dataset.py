import os
import sys
import pytest

def test_source_module_imports():
    """Verify that the main source modules can be imported without syntax errors"""
    try:
        import src.config
        import src.download_dataset
        import src.features
    except ImportError as e:
        pytest.fail(f"Failed to import core source modules: {e}")