"""AI Trace IR v0.1: the companion research record next to OpenTelemetry spans (ADR-003).

Only ``schema_version`` 0.1.x is accepted: in 0.x a minor bump may break readers. Fields added by a
later 0.1 patch are kept verbatim in ``model_extra`` and never interpreted.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"


class EvidenceClass(StrEnum):
    OBSERVED = "observed"
    MEASURED = "measured"
    INFERRED = "inferred"
    INTERVENED = "intervened"


class Location(BaseModel):
    model_config = ConfigDict(extra="allow")

    forward_pass_id: str | None = None
    generation_step: int | None = None
    token_position: int | None = None
    layer: int | None = None
    hook_name: str | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_revision: str
    tokenizer_revision: str
    instrumentation_version: str
    capture_config_hash: str | None = None


class RetrievalDocumentSelected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    doc_key: str
    version: int
    sha256: str
    artifact_id: str | None
    rank: int
    score: float


class ModelInvocationCompleted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_sha256: str
    output_sha256: str | None
    prompt_artifact_id: str | None
    neural_access: Literal["available", "unavailable"]


# Event types this version understands: payload model and the one evidence class each may carry.
KNOWN_EVENTS: dict[str, tuple[type[BaseModel], EvidenceClass]] = {
    "retrieval.document.selected": (RetrievalDocumentSelected, EvidenceClass.OBSERVED),
    "model.invocation.completed": (ModelInvocationCompleted, EvidenceClass.OBSERVED),
}


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    evidence_class: EvidenceClass
    occurred_at: AwareDatetime
    run_id: str = Field(min_length=1)
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    model_invocation_id: str | None = None
    location: Location | None = None
    payload: dict[str, Any]
    provenance: Provenance

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        parts = value.split(".")
        if len(parts) != 3 or parts[:2] != ["0", "1"] or not parts[2].isdecimal():
            raise ValueError(f"SCHEMA_VERSION_UNSUPPORTED: {value!r} (supported: 0.1.x)")
        return value

    @model_validator(mode="after")
    def _check_event(self) -> "TraceEvent":
        if self.event_type.startswith(("model.", "neural.")) and not self.model_invocation_id:
            raise ValueError("model_invocation_id is required for model.* and neural.* events")
        known = KNOWN_EVENTS.get(self.event_type)
        if known is not None:
            payload_model, evidence_class = known
            if self.evidence_class is not evidence_class:
                raise ValueError(f"{self.event_type} must have evidence_class {evidence_class}")
            payload_model.model_validate(self.payload)
        return self

    @property
    def is_known_type(self) -> bool:
        return self.event_type in KNOWN_EVENTS
