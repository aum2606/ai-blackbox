import json
from collections.abc import Iterator
from typing import Any

import pytest
from run_registry import registry
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError
from trace_ingest.events import record_event
from trace_schema.ir import SCHEMA_VERSION, TraceEvent


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    with engine.begin() as connection:
        yield connection


def _event(run_id: str, invocation_id: str | None, **overrides: Any) -> TraceEvent:
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "evt_1",
        "event_type": "custom.thing",
        "evidence_class": "observed",
        "occurred_at": "2026-11-17T08:30:00Z",
        "run_id": run_id,
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "model_invocation_id": invocation_id,
        "payload": {"anything": 1},
        "provenance": {
            "model_revision": "r",
            "tokenizer_revision": "t",
            "instrumentation_version": "phase2",
        },
    }
    return TraceEvent.model_validate({**data, **overrides})


def _run(conn: Connection) -> str:
    return registry.create_run(
        conn,
        task_id="stale_document",
        dataset_case_id="case_0007",
        retrieval_fault="none",
        capture_content=False,
    )


def test_event_is_persisted_once_with_unknown_fields_kept(conn: Connection) -> None:
    run_id = _run(conn)
    event = _event(run_id, None, schema_version="0.1.4", future_field=[1, 2])
    record_event(conn, event)
    record_event(conn, event)  # replay
    (row,) = conn.execute(text("SELECT * FROM trace_events")).all()
    assert row.known_type == 0 and row.evidence_class == "observed"
    assert row.occurred_at == "2026-11-17T08:30:00.000Z"
    assert json.loads(row.payload_json) == {"anything": 1}
    assert json.loads(row.extra_json) == {"future_field": [1, 2]}


def test_event_for_unknown_run_is_rejected_by_the_database(conn: Connection) -> None:
    with pytest.raises(IntegrityError):
        record_event(conn, _event("run_missing", None))
