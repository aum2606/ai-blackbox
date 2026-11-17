from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from demo_rag.backends import Generation
from evidence_store.engine import make_engine
from sqlalchemy import Engine

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"  # kept apart from the database file in tmp_path


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


class StubBackend:
    """Stands in for a real model: fixed answer and token IDs, so tests need no torch."""

    model_id = "stub-real"
    model_revision = "rev1"
    tokenizer_revision = "tok1"
    neural_access = "available"
    sampling_config: dict[str, Any] = {"do_sample": False}  # noqa: RUF012 - read-only

    def generate(self, prompt: str, document_text: str) -> Generation:
        return Generation("some answer", [1, 2, 3], [4, 5])


@pytest.fixture
def stub_backend() -> StubBackend:
    return StubBackend()
