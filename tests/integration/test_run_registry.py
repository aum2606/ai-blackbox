from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from evidence_store.engine import make_engine
from run_registry import registry
from run_registry.registry import RegistryError
from sqlalchemy import Connection, text

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC: dict[str, Any] = {
    "corpus_id": "stale_document_v1",
    "doc_key": "plan_delta",
    "version": 1,
    "sha256": "a" * 64,
    "source_path": "documents/plan_delta/v1.txt",
    "valid_from": "2025-01-01",
    "valid_to": "2026-09-01",
}


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[Connection]:
    url = f"sqlite:///{tmp_path / 'blackbox.db'}"
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    engine = make_engine(url)
    with engine.begin() as connection:
        yield connection
    engine.dispose()


def _new_run(conn: Connection, capture: bool = False) -> str:
    return registry.create_run(
        conn,
        task_id="stale_document",
        dataset_case_id="case_0007",
        retrieval_fault="stale_version",
        capture_content=capture,
    )


def test_ids_link_run_invocation_and_document(conn: Connection) -> None:
    run_id = _new_run(conn)
    registry.attach_trace(conn, run_id, "t" * 32)
    doc_id = registry.upsert_document(conn, **DOC)
    inv_id = registry.create_invocation(
        conn,
        run_id=run_id,
        model_id="fixture-extractive",
        model_revision="rev",
        tokenizer_revision="not_applicable",
        prompt_sha256="p" * 64,
        sampling_config={"deterministic": True},
        neural_access="unavailable",
    )
    registry.complete_invocation(conn, inv_id, span_id="s" * 16, output_sha256="o" * 64)
    registry.record_retrieval(
        conn, run_id=run_id, invocation_id=inv_id, document_id=doc_id, rank=1, score=1.0
    )
    registry.complete_run(conn, run_id, status="completed")

    row = conn.execute(
        text(
            "SELECT r.trace_id, r.status, i.span_id, d.doc_key, d.version FROM runs r"
            " JOIN model_invocations i ON i.run_id = r.id"
            " JOIN retrieval_results rr ON rr.invocation_id = i.id AND rr.run_id = r.id"
            " JOIN documents d ON d.id = rr.document_id WHERE r.id = :id"
        ),
        {"id": run_id},
    ).one()
    assert tuple(row) == ("t" * 32, "completed", "s" * 16, "plan_delta", 1)


def test_upsert_document_is_idempotent_and_rejects_changed_bytes(conn: Connection) -> None:
    first = registry.upsert_document(conn, **DOC)
    assert registry.upsert_document(conn, **DOC) == first
    with pytest.raises(RegistryError) as err:
        registry.upsert_document(conn, **{**DOC, "sha256": "b" * 64})
    assert err.value.code == "DOCUMENT_HASH_CONFLICT"


def test_answer_text_requires_content_capture_opt_in(conn: Connection) -> None:
    with pytest.raises(RegistryError) as err:
        registry.complete_run(conn, _new_run(conn), status="completed", answer_text="500 units")
    assert err.value.code == "CONTENT_CAPTURE_NOT_ENABLED"

    run_id = _new_run(conn, capture=True)
    registry.complete_run(conn, run_id, status="completed", answer_text="500 units")
    stored = conn.execute(text("SELECT answer_text FROM runs WHERE id = :id"), {"id": run_id})
    assert stored.scalar_one() == "500 units"


def test_run_can_only_be_completed_once(conn: Connection) -> None:
    run_id = _new_run(conn)
    registry.complete_run(conn, run_id, status="failed", error_type="CASE_NOT_FOUND")
    with pytest.raises(RegistryError) as err:
        registry.complete_run(conn, run_id, status="completed")
    assert err.value.code == "RUN_NOT_RUNNING"
