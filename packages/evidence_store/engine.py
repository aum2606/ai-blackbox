"""SQLite engine factory. Every connection enforces foreign keys (ADD §5.3)."""

import os
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url

DEFAULT_URL = "sqlite:///./.local/blackbox.db"


def database_url() -> str:
    """The single place DATABASE_URL is read (API, CLI and migrations)."""
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.database and parsed.database != ":memory:":
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")  # bounded wait on a transient lock
        cursor.close()

    return engine
