import os

from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not configured")
    return database_url
