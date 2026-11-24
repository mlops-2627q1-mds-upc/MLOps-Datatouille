from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from src.database.settings import get_database_url


def create_database_engine(database_url: str | None = None,) -> Engine:
    """Create a SQLAlchemy database engine."""

    url = make_url(database_url or get_database_url())

    if url.get_backend_name() != "postgresql":
        raise ValueError("A PostgreSQL database URL is required")

    url = url.set(drivername="postgresql+psycopg")

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
        connect_args={"connect_timeout": 10},
        echo=False,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the database session factory."""

    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )