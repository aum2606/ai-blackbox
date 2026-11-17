"""Phase 2 exit test: trace -> invocation -> artifact, reached through IDs only."""

from pathlib import Path

from blackbox_api.app import create_app
from fastapi.testclient import TestClient
from sqlalchemy import Engine


def test_invocation_to_verified_prompt_artifact_and_its_source_document(
    engine: Engine, artifact_root: Path
) -> None:
    client = TestClient(create_app(engine, artifact_root))
    run_id = client.post(
        "/v1/runs",
        json={"case_id": "case_0007", "retrieval_fault": "stale_version", "capture_content": True},
    ).json()["run_id"]
    run = client.get(f"/v1/runs/{run_id}").json()

    invocation = client.get(f"/v1/invocations/{run['invocations'][0]['id']}").json()
    prompt = client.get(f"/v1/artifacts/{invocation['prompt_artifact_id']}").json()
    document_artifact_id = run["retrievals"][0]["document"]["artifact_id"]
    assert prompt["verified"] is True and prompt["category"] == "derived"
    assert prompt["parents"] == [document_artifact_id]
    assert client.get(f"/v1/artifacts/{document_artifact_id}").json()["verified"] is True

    # Corrupting the stored bytes is detected on the next lookup.
    stored = next(p for p in artifact_root.rglob("*") if p.is_file() and p.stem == prompt["sha256"])
    stored.write_bytes(b"corrupted")
    again = client.get(f"/v1/artifacts/{invocation['prompt_artifact_id']}").json()
    assert again["verified"] is False


def test_no_prompt_artifact_without_opt_in(engine: Engine, artifact_root: Path) -> None:
    client = TestClient(create_app(engine, artifact_root))
    run_id = client.post("/v1/runs", json={"case_id": "case_0007"}).json()["run_id"]
    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["invocations"][0]["prompt_artifact_id"] is None
    assert run["retrievals"][0]["document"]["artifact_id"] is not None
