from pathlib import Path
from typing import Any

import pytest
from blackbox_api.app import create_app
from demo_rag.backends import Backend
from demo_rag.pipeline import run_case
from experiment_worker.worker import run_once
from fastapi.testclient import TestClient
from instrumentation.plan import CaptureData, CapturedMeasurement, CapturePlan
from sqlalchemy import Engine, text
from trace_ingest.exporter import make_tracer_provider

HOOK = "blocks.3.hook_resid_post"


class _Adapter:
    def capture(self, token_ids: list[int], plan: CapturePlan) -> CaptureData:
        return CaptureData(
            measurements=[
                CapturedMeasurement(
                    "activation",
                    HOOK,
                    3,
                    p,
                    "float32",
                    [512],
                    f"bytes-{p}".encode(),
                    {"l2_norm": 2.0, "mean": 0.1, "max_abs": 1.0},
                )  # fmt: skip
                for p in plan.positions
            ],
            argmax_ids=[0, 0, 4, 5, 0],
            max_logit_diff=0.0,
            runtime={"device": "cpu"},
        )


def _client_and_invocation(
    engine: Engine, root: Path, backend: Backend | None, capture: bool = True
) -> tuple[TestClient, str]:
    tracer = make_tracer_provider(engine).get_tracer("test")
    kwargs: dict[str, Any] = {"backend": backend} if backend else {}
    run_case(engine, tracer, root, "case_0007", "none", capture_content=capture, **kwargs)
    with engine.connect() as conn:
        invocation_id = conn.execute(text("SELECT id FROM model_invocations")).scalar_one()
    return TestClient(create_app(engine, root)), invocation_id


def _error(response: Any, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code


def test_capture_is_validated_resolved_and_deduplicated(
    engine: Engine, artifact_root: Path, stub_backend: Backend
) -> None:
    client, inv = _client_and_invocation(engine, artifact_root, stub_backend)
    body = {"hooks": [HOOK], "token_positions": [-1, 0, 4]}
    first = client.post(f"/v1/invocations/{inv}/captures", json=body)
    assert first.status_code == 202
    job = first.json()
    assert job["status"] == "queued" and job["plan"]["positions"] == [0, 4]  # -1 resolved
    assert client.post(f"/v1/invocations/{inv}/captures", json=body).json()["id"] == job["id"]
    assert client.get(f"/v1/captures/{job['id']}").json()["status"] == "queued"


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"hooks": ["blocks.9.hook_resid_post"], "token_positions": [0]}, 422, "HOOK_UNSUPPORTED"),
        ({"hooks": [HOOK], "token_positions": [5]}, 422, "POSITION_OUT_OF_RANGE"),
        ({"hooks": [HOOK], "token_positions": [0], "extra": 1}, 422, "VALIDATION_ERROR"),
    ],
)
def test_invalid_capture_requests_get_structured_errors(
    engine: Engine,
    artifact_root: Path,
    stub_backend: Backend,
    body: dict[str, Any],
    status: int,
    code: str,
) -> None:
    client, inv = _client_and_invocation(engine, artifact_root, stub_backend)
    _error(client.post(f"/v1/invocations/{inv}/captures", json=body), status, code)


def test_capture_refused_without_neural_access(engine: Engine, artifact_root: Path) -> None:
    body = {"hooks": [HOOK], "token_positions": [0]}
    fixture_client, fixture_inv = _client_and_invocation(engine, artifact_root, None)
    _error(
        fixture_client.post(f"/v1/invocations/{fixture_inv}/captures", json=body),
        409,
        "NEURAL_ACCESS_UNAVAILABLE",
    )


def test_capture_refused_when_the_run_did_not_opt_in(
    engine: Engine, artifact_root: Path, stub_backend: Backend
) -> None:
    client, inv = _client_and_invocation(engine, artifact_root, stub_backend, capture=False)
    body = {"hooks": [HOOK], "token_positions": [0]}
    _error(client.post(f"/v1/invocations/{inv}/captures", json=body), 403, "PRIVACY_RESTRICTED")


def test_unknown_invocation_and_capture_are_404(
    engine: Engine, artifact_root: Path, stub_backend: Backend
) -> None:
    client, _ = _client_and_invocation(engine, artifact_root, stub_backend)
    body = {"hooks": [HOOK], "token_positions": [0]}
    _error(client.post("/v1/invocations/inv_x/captures", json=body), 404, "INVOCATION_NOT_FOUND")
    _error(client.get("/v1/invocations/inv_x/measurements"), 404, "INVOCATION_NOT_FOUND")
    _error(client.get("/v1/captures/cap_x"), 404, "CAPTURE_NOT_FOUND")


def test_measurements_are_summaries_labelled_measured_with_verified_artifacts(
    engine: Engine, artifact_root: Path, stub_backend: Backend
) -> None:
    client, inv = _client_and_invocation(engine, artifact_root, stub_backend)
    assert client.get(f"/v1/invocations/{inv}/measurements").json() == []  # nothing captured yet
    job = client.post(
        f"/v1/invocations/{inv}/captures", json={"hooks": [HOOK], "token_positions": [0, -1]}
    ).json()
    assert run_once(engine, artifact_root, _Adapter).status == "completed"

    done = client.get(f"/v1/captures/{job['id']}").json()
    assert done["status"] == "completed" and done["replay"]["status"] == "approximate"
    measurements = client.get(f"/v1/invocations/{inv}/measurements").json()
    assert [(m["token_position"], m["evidence_class"]) for m in measurements] == [
        (0, "measured"),
        (4, "measured"),
    ]
    assert "payload" not in measurements[0] and measurements[0]["summary"]["l2_norm"] == 2.0
    artifact = client.get(f"/v1/artifacts/{measurements[0]['artifact_id']}").json()
    assert artifact["verified"] is True and artifact["mime_type"] == "application/x-safetensors"
