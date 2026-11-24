"""Unit tests for the configuration of the PostgreSQL connection (no database is contacted)."""

import pytest

from src.database.connection import create_database_engine, create_session_factory
from src.database.settings import get_database_url

POSTGRES_URL = "postgresql://user:secret@db.example:5432/datatouille"


def test_get_database_url_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test of the database URL read from DATABASE_URL."""
    monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)
    assert get_database_url() == POSTGRES_URL


def test_get_database_url_requires_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test of the error raised when DATABASE_URL is not configured."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        get_database_url()


def test_engine_uses_psycopg_driver() -> None:
    """Test of the psycopg driver selected for a plain postgresql:// URL."""
    engine = create_database_engine(POSTGRES_URL)
    assert engine.url.drivername == "postgresql+psycopg"
    assert engine.url.database == "datatouille"
    session_factory = create_session_factory(engine)
    assert session_factory.kw["bind"] is engine
    engine.dispose()


def test_engine_rejects_other_databases() -> None:
    """Test of the error raised for a database URL that is not PostgreSQL."""
    with pytest.raises(ValueError, match="PostgreSQL"):
        create_database_engine("sqlite:///predictions.db")
