"""Read side of the registry: typed records, no API or UI types."""

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, text


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    doc_key: str
    version: int
    sha256: str
    valid_from: str
    valid_to: str | None
    artifact_id: str | None


@dataclass(frozen=True)
class RetrievalRecord:
    rank: int
    score: float
    document: DocumentRecord


@dataclass(frozen=True)
class InvocationRecord:
    id: str
    run_id: str
    trace_id: str | None
    span_id: str | None
    model_id: str
    model_revision: str
    tokenizer_revision: str
    prompt_sha256: str
    output_sha256: str | None
    sampling_config: dict[str, Any]
    neural_access: str
    prompt_artifact_id: str | None
    created_at: str


@dataclass(frozen=True)
class RunRecord:
    id: str
    task_id: str
    dataset_case_id: str
    retrieval_fault: str
    trace_id: str | None
    status: str
    error_type: str | None
    capture_content: bool
    answer_text: str | None
    started_at: str
    ended_at: str | None
    invocations: tuple[InvocationRecord, ...]
    retrievals: tuple[RetrievalRecord, ...]


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    sha256: str
    mime_type: str
    nbytes: int
    category: str
    parents: tuple[str, ...]


def get_artifact(conn: Connection, artifact_id: str) -> ArtifactRecord | None:
    row = conn.execute(
        text("SELECT id, sha256, mime_type, nbytes, category FROM artifacts WHERE id = :id"),
        {"id": artifact_id},
    ).first()
    if row is None:
        return None
    parents = conn.execute(
        text("SELECT parent_id FROM artifact_lineage WHERE child_id = :id ORDER BY parent_id"),
        {"id": artifact_id},
    ).scalars()
    return ArtifactRecord(
        id=row.id,
        sha256=row.sha256,
        mime_type=row.mime_type,
        nbytes=row.nbytes,
        category=row.category,
        parents=tuple(parents),
    )


_INVOCATION_SQL = (
    "SELECT i.id, i.run_id, r.trace_id, i.span_id, i.model_id, i.model_revision,"
    " i.tokenizer_revision, i.prompt_sha256, i.output_sha256, i.sampling_config_json,"
    " i.neural_access, i.prompt_artifact_id, i.created_at FROM model_invocations i"
    " JOIN runs r ON r.id = i.run_id"
)


def _invocation(row: Any) -> InvocationRecord:
    return InvocationRecord(
        id=row.id,
        run_id=row.run_id,
        trace_id=row.trace_id,
        span_id=row.span_id,
        model_id=row.model_id,
        model_revision=row.model_revision,
        tokenizer_revision=row.tokenizer_revision,
        prompt_sha256=row.prompt_sha256,
        output_sha256=row.output_sha256,
        sampling_config=json.loads(row.sampling_config_json),
        neural_access=row.neural_access,
        prompt_artifact_id=row.prompt_artifact_id,
        created_at=row.created_at,
    )


def get_invocation(conn: Connection, invocation_id: str) -> InvocationRecord | None:
    row = conn.execute(text(f"{_INVOCATION_SQL} WHERE i.id = :id"), {"id": invocation_id}).first()
    return None if row is None else _invocation(row)


def get_run(conn: Connection, run_id: str) -> RunRecord | None:
    run = conn.execute(text("SELECT * FROM runs WHERE id = :id"), {"id": run_id}).first()
    if run is None:
        return None
    invocations = conn.execute(
        text(f"{_INVOCATION_SQL} WHERE i.run_id = :id ORDER BY i.created_at, i.id"), {"id": run_id}
    )
    retrievals = conn.execute(
        text(
            "SELECT rr.rank, rr.score, d.id, d.doc_key, d.version, d.sha256, d.valid_from,"
            " d.valid_to, d.artifact_id FROM retrieval_results rr"
            " JOIN documents d ON d.id = rr.document_id"
            " WHERE rr.run_id = :id ORDER BY rr.rank"
        ),
        {"id": run_id},
    )
    return RunRecord(
        id=run.id,
        task_id=run.task_id,
        dataset_case_id=run.dataset_case_id,
        retrieval_fault=run.retrieval_fault,
        trace_id=run.trace_id,
        status=run.status,
        error_type=run.error_type,
        capture_content=bool(run.capture_content),
        answer_text=run.answer_text,
        started_at=run.started_at,
        ended_at=run.ended_at,
        invocations=tuple(_invocation(r) for r in invocations),
        retrievals=tuple(
            RetrievalRecord(
                rank=r.rank,
                score=r.score,
                document=DocumentRecord(
                    id=r.id,
                    doc_key=r.doc_key,
                    version=r.version,
                    sha256=r.sha256,
                    valid_from=r.valid_from,
                    valid_to=r.valid_to,
                    artifact_id=r.artifact_id,
                ),
            )
            for r in retrievals
        ),
    )
