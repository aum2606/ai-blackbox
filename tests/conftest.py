from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from evidence_store.engine import make_engine
from sqlalchemy import Engine

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Engine on a fresh, fully migrated SQLite file."""
    url = f"sqlite:///{tmp_path / 'blackbox.db'}"
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    engine = make_engine(url)
    yield engine
    engine.dispose()
