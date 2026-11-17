"""Run, invocation and retrieval records.

Functions take a ``Connection`` so the caller owns the transaction. No inference or tracing here.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, text


class RegistryError(Exception):
    """Registry rule violation; ``code`` is the structured error type reported by the API."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def upsert_document(
    conn: Connection,
    *,
    corpus_id: str,
    doc_key: str,
    version: int,
    sha256: str,
    source_path: str,
    valid_from: str,
    valid_to: str | None,
    artifact_id: str | None = None,
) -> str:
    """Register a document version once; a version's bytes may never change.

    ``artifact_id`` is attached when the row has none yet, which also backfills Phase 1 rows.
    """
    conn.execute(
        text(
            "INSERT INTO documents (id, corpus_id, doc_key, version, sha256, source_path,"
            " valid_from, valid_to, artifact_id) VALUES (:id, :corpus_id, :doc_key, :version,"
            " :sha256,"
            " :source_path, :valid_from, :valid_to, :artifact_id)"
            " ON CONFLICT (doc_key, version) DO NOTHING"
        ),
        {
            "id": _new_id("doc"),
            "corpus_id": corpus_id,
            "doc_key": doc_key,
            "version": version,
            "sha256": sha256,
            "source_path": source_path,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "artifact_id": artifact_id,
        },
    )
    row = conn.execute(
        text("SELECT id, sha256, artifact_id FROM documents WHERE doc_key = :k AND version = :v"),
        {"k": doc_key, "v": version},
    ).one()
    if row.sha256 != sha256:
        raise RegistryError(
            "DOCUMENT_HASH_CONFLICT", f"{doc_key} v{version} is already registered with other bytes"
        )
    if row.artifact_id is None and artifact_id is not None:
        conn.execute(
            text("UPDATE documents SET artifact_id = :artifact WHERE id = :id"),
            {"artifact": artifact_id, "id": row.id},
        )
    return str(row.id)


def create_run(
    conn: Connection,
    *,
    task_id: str,
    dataset_case_id: str,
    retrieval_fault: str,
    capture_content: bool,
) -> str:
    run_id = _new_id("run")
    conn.execute(
        text(
            "INSERT INTO runs (id, task_id, dataset_case_id, retrieval_fault, status,"
            " capture_content, started_at) VALUES (:id, :task_id, :case_id, :fault, 'running',"
            " :capture, :now)"
        ),
        {
            "id": run_id,
            "task_id": task_id,
            "case_id": dataset_case_id,
            "fault": retrieval_fault,
            "capture": int(capture_content),
            "now": _now(),
        },
    )
    return run_id


def attach_trace(conn: Connection, run_id: str, trace_id: str) -> None:
    conn.execute(
        text("UPDATE runs SET trace_id = :trace_id WHERE id = :id"),
        {"trace_id": trace_id, "id": run_id},
    )


def create_invocation(
    conn: Connection,
    *,
    run_id: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    prompt_sha256: str,
    sampling_config: dict[str, Any],
    neural_access: str,
    prompt_artifact_id: str | None = None,
) -> str:
    """Create the invocation ID before the model call so it can be put on the trace."""
    invocation_id = _new_id("inv")
    conn.execute(
        text(
            "INSERT INTO model_invocations (id, run_id, model_id, model_revision,"
            " tokenizer_revision, prompt_sha256, sampling_config_json, neural_access, created_at,"
            " prompt_artifact_id) VALUES (:id, :run_id, :model_id, :model_revision,"
            " :tokenizer_revision, :prompt_sha256, :sampling, :neural_access, :now,"
            " :prompt_artifact)"
        ),
        {
            "id": invocation_id,
            "run_id": run_id,
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "prompt_sha256": prompt_sha256,
            "sampling": json.dumps(sampling_config, sort_keys=True),
            "neural_access": neural_access,
            "now": _now(),
            "prompt_artifact": prompt_artifact_id,
        },
    )
    return invocation_id


def complete_invocation(
    conn: Connection, invocation_id: str, *, span_id: str, output_sha256: str | None
) -> None:
    """Link the invocation to its span; ``output_sha256`` is None if the call failed."""
    conn.execute(
        text("UPDATE model_invocations SET span_id = :span, output_sha256 = :out WHERE id = :id"),
        {"span": span_id, "out": output_sha256, "id": invocation_id},
    )


def record_retrieval(
    conn: Connection,
    *,
    run_id: str,
    invocation_id: str,
    document_id: str,
    rank: int,
    score: float,
) -> str:
    retrieval_id = _new_id("ret")
    conn.execute(
        text(
            "INSERT INTO retrieval_results (id, run_id, invocation_id, document_id, rank, score,"
            " created_at) VALUES (:id, :run_id, :inv, :doc, :rank, :score, :now)"
        ),
        {
            "id": retrieval_id,
            "run_id": run_id,
            "inv": invocation_id,
            "doc": document_id,
            "rank": rank,
            "score": score,
            "now": _now(),
        },
    )
    return retrieval_id


def record_token_sequence(
    conn: Connection,
    *,
    invocation_id: str,
    role: str,
    tokenizer_revision: str,
    n_tokens: int,
    artifact_id: str,
) -> str:
    sequence_id = _new_id("tok")
    conn.execute(
        text(
            "INSERT INTO token_sequences (id, invocation_id, role, tokenizer_revision, n_tokens,"
            " artifact_id, created_at) VALUES (:id, :inv, :role, :tok, :n, :art, :now)"
        ),
        {
            "id": sequence_id,
            "inv": invocation_id,
            "role": role,
            "tok": tokenizer_revision,
            "n": n_tokens,
            "art": artifact_id,
            "now": _now(),
        },
    )
    return sequence_id


def complete_run(
    conn: Connection,
    run_id: str,
    *,
    status: str,
    answer_text: str | None = None,
    error_type: str | None = None,
) -> None:
    """Move a run from ``running`` to ``completed``/``failed``, exactly once."""
    if status not in ("completed", "failed"):
        raise RegistryError("INVALID_STATUS", f"cannot complete a run as {status!r}")
    row = conn.execute(
        text("SELECT capture_content, status FROM runs WHERE id = :id"), {"id": run_id}
    ).one_or_none()
    if row is None or row.status != "running":
        raise RegistryError("RUN_NOT_RUNNING", f"{run_id} is not a running run")
    if answer_text is not None and not row.capture_content:
        raise RegistryError("CONTENT_CAPTURE_NOT_ENABLED", "run did not opt in to content capture")
    conn.execute(
        text(
            "UPDATE runs SET status = :status, answer_text = :answer, error_type = :error,"
            " ended_at = :now WHERE id = :id"
        ),
        {
            "status": status,
            "answer": answer_text,
            "error": error_type,
            "now": _now(),
            "id": run_id,
        },
    )
