"""Backend seam: a stub real-model backend (no torch) and, under -m neural, pythia itself."""

from pathlib import Path
from typing import Any

from demo_rag.backends import Generation
from demo_rag.pipeline import run_case
from evidence_store.artifacts import get_verified
from sqlalchemy import Engine, text
from trace_ingest.exporter import make_tracer_provider


class StubBackend:
    model_id = "stub-real"
    model_revision = "rev1"
    tokenizer_revision = "tok1"
    neural_access = "available"
    sampling_config: dict[str, Any] = {"do_sample": False}  # noqa: RUF012

    def generate(self, prompt: str, document_text: str) -> Generation:
        return Generation("some answer", [1, 2, 3], [4, 5])


def _run(engine: Engine, root: Path, capture: bool) -> str:
    tracer = make_tracer_provider(engine).get_tracer("test")
    return run_case(
        engine, tracer, root, "case_0007", "stale_version", capture_content=capture,
        backend=StubBackend(),
    ).run_id  # fmt: skip


def _rows(engine: Engine, sql: str) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql))]


def test_token_ids_are_stored_only_with_content_opt_in(engine: Engine, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _run(engine, root, capture=False)
    assert _rows(engine, "SELECT * FROM token_sequences") == []
    (inv,) = _rows(engine, "SELECT model_id, neural_access FROM model_invocations")
    assert (inv["model_id"], inv["neural_access"]) == ("stub-real", "available")

    _run(engine, root, capture=True)
    sequences = {r["role"]: r for r in _rows(engine, "SELECT * FROM token_sequences")}
    assert {k: v["n_tokens"] for k, v in sequences.items()} == {"prompt": 3, "output": 2}
    with engine.connect() as conn:
        data = get_verified(conn, root, sequences["prompt"]["artifact_id"])
    assert data == b'{"token_ids": [1, 2, 3]}'
