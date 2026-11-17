"""A real pythia run through the pipeline. Run with `uv run pytest -m neural`."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
from demo_rag.pipeline import run_case  # noqa: E402
from demo_rag.pythia_backend import PythiaBackend  # noqa: E402
from evidence_store.artifacts import get_verified  # noqa: E402
from sqlalchemy import Engine, text  # noqa: E402
from trace_ingest.exporter import make_tracer_provider  # noqa: E402

pytestmark = pytest.mark.neural


def test_pythia_run_records_identity_and_stores_token_ids_on_opt_in(
    engine: Engine, tmp_path: Path
) -> None:
    backend = PythiaBackend()
    root = tmp_path / "artifacts"
    tracer = make_tracer_provider(engine).get_tracer("test")
    result = run_case(
        engine, tracer, root, "case_0007", "stale_version", capture_content=True, backend=backend
    )
    assert result.status == "completed"
    with engine.connect() as conn:
        inv = conn.execute(
            text("SELECT model_id, model_revision, neural_access, id FROM model_invocations")
        ).one()
        sequences = {
            r.role: r
            for r in conn.execute(
                text("SELECT role, n_tokens, artifact_id FROM token_sequences ORDER BY role")
            )
        }
        prompt_ids = json.loads(get_verified(conn, root, sequences["prompt"].artifact_id))
    assert (inv.model_id, inv.neural_access) == ("EleutherAI/pythia-70m-deduped", "available")
    assert inv.model_revision == backend.model_revision
    assert sequences["output"].n_tokens == 4
    assert prompt_ids["token_ids"] == backend.adapter.tokenize(
        backend.adapter.decode(prompt_ids["token_ids"])
    )
