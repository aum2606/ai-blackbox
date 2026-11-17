"""Phase 1 exit test: from an answer, reach the exact document version and model invocation."""

from pathlib import Path

from blackbox_api.app import create_app
from demo_rag.fixtures import load_case
from fastapi.testclient import TestClient
from sqlalchemy import Engine


def test_answer_to_document_version_and_invocation(engine: Engine, artifact_root: Path) -> None:
    client = TestClient(create_app(engine, artifact_root))
    created = client.post(
        "/v1/runs",
        json={"case_id": "case_0007", "retrieval_fault": "stale_version", "capture_content": True},
    )
    assert created.status_code == 201
    run_id = created.json()["run_id"]

    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["answer"]["text"] == "500 units"
    assert run["expected_answer"] == "800 units" and run["is_correct"] is False

    (retrieval,) = run["retrievals"]
    stale = load_case("case_0007").documents[0]
    document = retrieval["document"]
    assert (document["doc_key"], document["version"]) == ("plan_delta", 1)
    assert document["sha256"] == stale.sha256 and document["valid_to"] == "2026-09-01"

    (listed,) = run["invocations"]
    invocation = client.get(f"/v1/invocations/{listed['id']}").json()
    assert invocation == listed
    assert invocation["run_id"] == run_id
    assert invocation["trace_id"] == run["trace_id"] == created.json()["trace_id"]
    assert invocation["neural_access"] == "unavailable"
    llm_span = next(s for s in run["spans"] if s["name"] == "llm.generate")
    assert invocation["span_id"] == llm_span["span_id"]
    assert {s["trace_id"] for s in run["spans"]} == {run["trace_id"]}


def test_current_document_gives_a_correct_answer(engine: Engine, artifact_root: Path) -> None:
    client = TestClient(create_app(engine, artifact_root))
    run_id = client.post("/v1/runs", json={"case_id": "case_0007"}).json()["run_id"]
    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["is_correct"] is True and run["retrievals"][0]["document"]["version"] == 2
