from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from evidence_store.engine import make_engine
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_TABLES = {"runs", "documents", "model_invocations", "retrieval_results", "spans"}


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'blackbox.db'}"


def _alembic(url: str) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def test_upgrade_then_downgrade(db_url: str) -> None:
    config = _alembic(db_url)
    command.upgrade(config, "head")
    engine = make_engine(db_url)
    assert PHASE1_TABLES <= set(inspect(engine).get_table_names())
    engine.dispose()

    command.downgrade(config, "base")
    engine = make_engine(db_url)
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()


def test_every_connection_enforces_foreign_keys_and_wal(db_url: str) -> None:
    command.upgrade(_alembic(db_url), "head")
    engine = make_engine(db_url)
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO model_invocations (id, run_id, model_id, model_revision,"
                    " tokenizer_revision, prompt_sha256, sampling_config_json, neural_access,"
                    " created_at) VALUES ('inv_x','run_missing','m','r','t','h','{}',"
                    "'unavailable','2026-11-17T00:00:00Z')"
                )
            )
    engine.dispose()
