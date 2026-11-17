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


def artifact_root() -> Path:
    return Path(os.environ.get("ARTIFACT_ROOT", "./.local/artifacts"))


def max_artifact_bytes() -> int:
    """Disk quota for all stored artifacts (initial default 1 GiB, ADD §12.4)."""
    return int(os.environ.get("MAX_LOCAL_ARTIFACT_BYTES", str(1024**3)))


def max_capture_bytes() -> int:
    """Cap on the tensor bytes one capture request may ask for (ADD §7.3)."""
    return int(os.environ.get("MAX_CAPTURE_BYTES", str(64 * 1024 * 1024)))


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
