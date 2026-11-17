import json
import shutil
from pathlib import Path

import pytest
from demo_rag.fixtures import DEFAULT_ROOT, MANIFEST_NAME, build_manifest, load_case
from demo_rag.pipeline import RunResult, run_case
from demo_rag.retriever import RetrievalFault
from evidence_store.artifacts import get_verified
from sqlalchemy import Engine, text
from trace_ingest.exporter import INVOCATION_ID_ATTR, make_tracer_provider

CASE_IDS = [f"case_{n:04d}" for n in range(1, 11)]


def _run(
    engine: Engine,
    artifact_root: Path,
    case_id: str,
    fault: RetrievalFault,
    capture: bool = False,
    root: Path = DEFAULT_ROOT,
) -> RunResult:
    tracer = make_tracer_provider(engine).get_tracer("test")
    return run_case(
        engine, tracer, artifact_root, case_id, fault, capture_content=capture, fixture_root=root
    )


def _rows(engine: Engine, sql: str, **params: str) -> list[dict[str, object]]:
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql), params)]


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("fault", ["none", "stale_version"])
def test_every_case_answers_correctly_only_without_the_fault(
    engine: Engine, artifact_root: Path, case_id: str, fault: RetrievalFault
) -> None:
    result = _run(engine, artifact_root, case_id, fault, capture=True)
    (run,) = _rows(engine, "SELECT answer_text FROM runs WHERE id = :id", id=result.run_id)
    is_correct = run["answer_text"] == load_case(case_id).expected_answer
    assert result.status == "completed"
    assert is_correct == (fault == "none")


def test_stale_run_links_answer_to_stale_document_and_invocation(
    engine: Engine, artifact_root: Path
) -> None:
    result = _run(engine, artifact_root, "case_0007", "stale_version", capture=True)
    (row,) = _rows(
        engine,
        "SELECT r.answer_text, r.trace_id, d.doc_key, d.version, d.valid_to, d.sha256,"
        " i.id AS inv_id, i.span_id, i.output_sha256 FROM runs r"
        " JOIN retrieval_results rr ON rr.run_id = r.id"
        " JOIN documents d ON d.id = rr.document_id"
        " JOIN model_invocations i ON i.id = rr.invocation_id AND i.run_id = r.id"
        " WHERE r.id = :id",
        id=result.run_id,
    )
    stale = load_case("case_0007").documents[0]
    assert row["answer_text"] == "500 units"
    assert (row["doc_key"], row["version"], row["valid_to"]) == ("plan_delta", 1, "2026-09-01")
    assert row["sha256"] == stale.sha256
    assert row["trace_id"] == result.trace_id and row["output_sha256"] and row["span_id"]


def test_trace_is_complete_and_ids_are_consistent(engine: Engine, artifact_root: Path) -> None:
    result = _run(engine, artifact_root, "case_0007", "stale_version")
    spans = {s["name"]: s for s in _rows(engine, "SELECT * FROM spans")}
    assert set(spans) == {"rag.run", "retriever.retrieve", "llm.generate"}
    root, retriever, llm = spans["rag.run"], spans["retriever.retrieve"], spans["llm.generate"]
    assert {s["trace_id"] for s in spans.values()} == {result.trace_id}
    assert root["parent_span_id"] is None
    assert retriever["parent_span_id"] == llm["parent_span_id"] == root["span_id"]
    assert {s["run_id"] for s in spans.values()} == {result.run_id}
    assert [s["openinference_span_kind"] for s in (root, retriever, llm)] == [
        "CHAIN",
        "RETRIEVER",
        "LLM",
    ]
    (inv,) = _rows(engine, "SELECT id, span_id, neural_access FROM model_invocations")
    assert inv["span_id"] == llm["span_id"] and inv["neural_access"] == "unavailable"
    assert json.loads(str(llm["attributes_json"]))[INVOCATION_ID_ATTR] == inv["id"]


def test_nothing_is_stored_without_content_capture_opt_in(
    engine: Engine, artifact_root: Path
) -> None:
    result = _run(engine, artifact_root, "case_0007", "stale_version")
    (run,) = _rows(engine, "SELECT answer_text FROM runs WHERE id = :id", id=result.run_id)
    assert run["answer_text"] is None
    stored = json.dumps(_rows(engine, "SELECT attributes_json FROM spans"))
    for secret in ("500 units", "Plan Delta", "reimbursement"):
        assert secret not in stored


def test_unextractable_answer_fails_the_run_with_error_status_on_spans(
    engine: Engine, artifact_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "stale_document"
    shutil.copytree(DEFAULT_ROOT, root)
    (root / "documents" / "plan_delta" / "v1.txt").write_text("No limit stated.\n")
    (root / MANIFEST_NAME).write_text(
        json.dumps({"schema_version": "1", "files": build_manifest(root)})
    )
    result = _run(engine, artifact_root, "case_0007", "stale_version", root=root)
    assert (result.status, result.error_type) == ("failed", "ANSWER_NOT_EXTRACTABLE")
    (run,) = _rows(engine, "SELECT status, error_type FROM runs WHERE id = :id", id=result.run_id)
    assert (run["status"], run["error_type"]) == ("failed", "ANSWER_NOT_EXTRACTABLE")
    statuses = {s["name"]: s["status_code"] for s in _rows(engine, "SELECT * FROM spans")}
    assert statuses["llm.generate"] == statuses["rag.run"] == "ERROR"
    (inv,) = _rows(engine, "SELECT span_id, output_sha256 FROM model_invocations")
    assert inv["span_id"] and inv["output_sha256"] is None


def test_document_artifacts_are_stored_and_verifiable_without_content_capture(
    engine: Engine, artifact_root: Path
) -> None:
    _run(engine, artifact_root, "case_0007", "none")
    docs = _rows(engine, "SELECT version, artifact_id FROM documents WHERE doc_key = 'plan_delta'")
    assert len(docs) == 2 and all(d["artifact_id"] for d in docs)
    by_version = {d["version"]: d["artifact_id"] for d in docs}
    with engine.connect() as conn:
        current = get_verified(conn, artifact_root, str(by_version[2]))
    assert current == load_case("case_0007").documents[1].text.encode("utf-8")


def test_prompt_artifact_exists_only_with_opt_in_and_derives_from_the_document(
    engine: Engine, artifact_root: Path
) -> None:
    _run(engine, artifact_root, "case_0007", "stale_version")
    (inv,) = _rows(engine, "SELECT prompt_artifact_id FROM model_invocations")
    assert inv["prompt_artifact_id"] is None
    assert not any(b"Question:" in p.read_bytes() for p in artifact_root.rglob("*") if p.is_file())

    result = _run(engine, artifact_root, "case_0007", "stale_version", capture=True)
    (inv,) = _rows(
        engine,
        "SELECT prompt_artifact_id FROM model_invocations WHERE run_id = :id",
        id=result.run_id,
    )
    (parent,) = _rows(
        engine,
        "SELECT parent_id, relation FROM artifact_lineage WHERE child_id = :id",
        id=str(inv["prompt_artifact_id"]),
    )
    (doc,) = _rows(
        engine, "SELECT artifact_id FROM documents WHERE doc_key = 'plan_delta' AND version = 1"
    )
    assert parent == {"parent_id": doc["artifact_id"], "relation": "derived_from"}
    with engine.connect() as conn:
        prompt = get_verified(conn, artifact_root, str(inv["prompt_artifact_id"]))
    assert b"Question:" in prompt and b"500 units" in prompt


def test_runs_emit_two_observed_events_linked_to_their_spans(
    engine: Engine, artifact_root: Path
) -> None:
    result = _run(engine, artifact_root, "case_0007", "stale_version")
    spans = {s["name"]: s["span_id"] for s in _rows(engine, "SELECT name, span_id FROM spans")}
    (inv,) = _rows(engine, "SELECT id FROM model_invocations")
    events = {e["event_type"]: e for e in _rows(engine, "SELECT * FROM trace_events")}
    assert set(events) == {"retrieval.document.selected", "model.invocation.completed"}
    for event in events.values():
        assert (event["evidence_class"], event["known_type"]) == ("observed", 1)
        assert (event["run_id"], event["trace_id"]) == (result.run_id, result.trace_id)
    assert events["retrieval.document.selected"]["span_id"] == spans["retriever.retrieve"]
    assert events["model.invocation.completed"]["span_id"] == spans["llm.generate"]
    assert events["model.invocation.completed"]["model_invocation_id"] == inv["id"]


def test_rerun_backfills_a_document_row_that_has_no_artifact(
    engine: Engine, artifact_root: Path
) -> None:
    """Rows created by Phase 1 have no artifact_id; the next run attaches one."""
    _run(engine, artifact_root, "case_0007", "none")
    with engine.begin() as conn:
        conn.execute(text("UPDATE documents SET artifact_id = NULL"))
    _run(engine, artifact_root, "case_0007", "none")
    docs = _rows(engine, "SELECT artifact_id FROM documents WHERE doc_key = 'plan_delta'")
    assert all(d["artifact_id"] for d in docs)
