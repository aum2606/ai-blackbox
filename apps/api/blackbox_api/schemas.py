"""Versioned /v1 request and response models. The dashboard types mirror these."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    retrieval_fault: Literal["none", "stale_version"] = "none"
    # Off by default: answer and document text are stored only on explicit opt-in (ADR-007).
    capture_content: bool = False


class RunCreated(BaseModel):
    run_id: str
    trace_id: str
    status: str
    error_type: str | None


class _FromRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DocumentOut(_FromRecord):
    id: str
    doc_key: str
    version: int
    sha256: str
    valid_from: str
    valid_to: str | None  # null marks the current version
    artifact_id: str | None


class RetrievalOut(_FromRecord):
    rank: int
    score: float
    document: DocumentOut


class InvocationOut(_FromRecord):
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
    neural_access: Literal["available", "unavailable"]
    prompt_artifact_id: str | None  # null unless the run opted in to content capture
    created_at: str


class ArtifactOut(BaseModel):
    id: str
    sha256: str
    mime_type: str
    nbytes: int
    category: str
    parents: list[str]
    verified: bool  # true only if the stored bytes were re-hashed on this request and match
    integrity_error: str | None  # error code when verified is false


class SpanOut(_FromRecord):
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    openinference_span_kind: str | None
    status_code: str
    start_time: str
    end_time: str


class AnswerOut(BaseModel):
    captured: bool  # false: the text was not stored; only its hash is known
    text: str | None
    sha256: str | None


class RunOut(BaseModel):
    run_id: str
    task_id: str
    case_id: str
    retrieval_fault: str
    trace_id: str | None
    status: str
    error_type: str | None
    started_at: str
    ended_at: str | None
    answer: AnswerOut
    expected_answer: str  # fixture ground truth, kept separate from any system output
    is_correct: bool | None  # observed; null until the run completed with an answer
    invocations: list[InvocationOut]
    retrievals: list[RetrievalOut]
    spans: list[SpanOut]


class HealthOut(BaseModel):
    status: Literal["ok"]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorOut(BaseModel):
    error: ErrorBody
