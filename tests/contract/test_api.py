import json
import os
from pathlib import Path
from typing import Any

import pytest
from blackbox_api import app as app_module
from blackbox_api.app import create_app
from fastapi.testclient import TestClient
from sqlalchemy import Engine

SNAPSHOT = Path(__file__).with_name("openapi_v1.json")


@pytest.fixture
def client(engine: Engine, artifact_root: Path) -> TestClient:
    return TestClient(create_app(engine, artifact_root))


def _error(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status
    error: dict[str, Any] = response.json()["error"]
    assert error["code"] == code
    assert error["request_id"] == response.headers["X-Request-ID"]
    return error


def test_health(client: TestClient) -> None:
    assert client.get("/v1/health").json() == {"status": "ok"}


def test_unknown_case_is_a_structured_404(client: TestClient) -> None:
    _error(client.post("/v1/runs", json={"case_id": "case_9999"}), 404, "CASE_NOT_FOUND")


def test_unknown_run_and_invocation_are_structured_404s(client: TestClient) -> None:
    _error(client.get("/v1/runs/run_missing"), 404, "RUN_NOT_FOUND")
    _error(client.get("/v1/invocations/inv_missing"), 404, "INVOCATION_NOT_FOUND")


def test_invalid_body_is_422_and_does_not_echo_input(client: TestClient) -> None:
    response = client.post("/v1/runs", json={"case_id": "case_0007", "retrieval_fault": "SECRET"})
    error = _error(response, 422, "VALIDATION_ERROR")
    assert "retrieval_fault" in error["message"] and "SECRET" not in response.text


def test_unexpected_error_hides_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, engine: Engine, artifact_root: Path
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("internal path /secret")

    monkeypatch.setattr(app_module, "run_case", boom)
    quiet = TestClient(create_app(engine, artifact_root), raise_server_exceptions=False)
    response = quiet.post("/v1/runs", json={"case_id": "case_0007"})
    _error(response, 500, "INTERNAL_ERROR")
    assert "secret" not in response.text and "Traceback" not in response.text


def test_content_is_not_returned_without_opt_in_but_correctness_is_observable(
    client: TestClient,
) -> None:
    run_id = client.post(
        "/v1/runs", json={"case_id": "case_0007", "retrieval_fault": "stale_version"}
    ).json()["run_id"]
    run = client.get(f"/v1/runs/{run_id}").json()
    assert run["answer"]["captured"] is False and run["answer"]["text"] is None
    assert run["answer"]["sha256"] and run["is_correct"] is False
    assert "500 units" not in json.dumps(run)


def test_openapi_matches_checked_in_snapshot(client: TestClient) -> None:
    """Contract guard: API changes must be deliberate. Refresh with UPDATE_SNAPSHOT=1."""
    current = json.dumps(client.app.openapi(), indent=2, sort_keys=True) + "\n"  # type: ignore[attr-defined]
    if os.environ.get("UPDATE_SNAPSHOT"):
        SNAPSHOT.write_text(current, encoding="utf-8", newline="\n")
    assert current == SNAPSHOT.read_text(encoding="utf-8")


def test_unknown_artifact_is_a_structured_404(client: TestClient) -> None:
    _error(client.get("/v1/artifacts/art_missing"), 404, "ARTIFACT_NOT_FOUND")


def test_tampered_artifact_is_reported_not_hidden(client: TestClient, artifact_root: Path) -> None:
    run_id = client.post("/v1/runs", json={"case_id": "case_0007"}).json()["run_id"]
    run = client.get(f"/v1/runs/{run_id}").json()
    artifact_id = run["retrievals"][0]["document"]["artifact_id"]
    ok = client.get(f"/v1/artifacts/{artifact_id}").json()
    assert ok["verified"] is True and ok["integrity_error"] is None
    assert "text" not in ok and "content" not in ok  # metadata only, never raw bytes

    next(p for p in artifact_root.rglob("*") if p.is_file() and p.stem == ok["sha256"]).write_bytes(
        b"tampered"
    )
    bad = client.get(f"/v1/artifacts/{artifact_id}").json()
    assert bad["verified"] is False and bad["integrity_error"] == "ARTIFACT_INTEGRITY_ERROR"
