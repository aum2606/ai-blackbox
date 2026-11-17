"""Phase 3 exit test: invocation -> layer/position measurement, verified and golden-checked.

Needs the neural group and the pinned model: `uv run pytest -m neural`.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
from blackbox_api.app import create_app  # noqa: E402
from demo_rag.pipeline import run_case  # noqa: E402
from demo_rag.pythia_backend import PythiaBackend  # noqa: E402
from evidence_store.artifacts import get_verified  # noqa: E402
from experiment_worker.worker import run_once  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from instrumentation.pythia import READ_ONLY_TOLERANCE  # noqa: E402
from safetensors.torch import load  # noqa: E402
from sqlalchemy import Engine  # noqa: E402
from trace_ingest.exporter import make_tracer_provider  # noqa: E402

pytestmark = pytest.mark.neural
GOLDEN = json.loads(
    Path(__file__).parents[1].joinpath("neural", "golden_pythia_70m.json").read_text("utf-8")
)


def test_real_run_to_verified_layer_measurement(engine: Engine, artifact_root: Path) -> None:
    import torch

    backend = PythiaBackend()
    tracer = make_tracer_provider(engine).get_tracer("test")
    # fault=none renders exactly the golden prompt, so prompt tokens equal the golden input_ids.
    run_case(
        engine, tracer, artifact_root, "case_0007", "none", capture_content=True, backend=backend
    )
    client = TestClient(create_app(engine, artifact_root))
    run = client.get(f"/v1/runs/{_only_run_id(engine)}").json()
    invocation = run["invocations"][0]
    assert invocation["neural_access"] == "available"

    last_prompt = len(GOLDEN["input_ids"]) - 1
    job = client.post(
        f"/v1/invocations/{invocation['id']}/captures",
        json={
            "hooks": ["blocks.3.hook_resid_post"],
            "token_positions": [last_prompt],
            "include_logits": True,
        },
    ).json()
    result = run_once(engine, artifact_root, lambda: backend.adapter)
    assert result.status == "completed"

    done = client.get(f"/v1/captures/{job['id']}").json()
    assert done["replay"]["max_logit_diff"] <= READ_ONLY_TOLERANCE
    assert done["replay"]["greedy_consistent"] is True and done["replay"]["status"] == "approximate"

    measurements = client.get(f"/v1/invocations/{invocation['id']}/measurements").json()
    by_kind = {m["kind"]: m for m in measurements}
    activation = by_kind["activation"]
    assert (activation["layer"], activation["token_position"]) == (3, last_prompt)
    for measurement in measurements:
        artifact = client.get(f"/v1/artifacts/{measurement['artifact_id']}").json()
        assert artifact["verified"] is True

    with engine.connect() as conn:
        logits = load(get_verified(conn, artifact_root, by_kind["logits"]["artifact_id"]))["data"]
    top = torch.topk(logits, 10)
    assert top.indices.tolist() == GOLDEN["top10_token_ids"]
    torch.testing.assert_close(
        top.values, torch.tensor(GOLDEN["top10_logits"]), rtol=1e-4, atol=1e-4
    )


def _only_run_id(engine: Engine) -> str:
    from sqlalchemy import text

    with engine.connect() as conn:
        return str(conn.execute(text("SELECT id FROM runs")).scalar_one())
