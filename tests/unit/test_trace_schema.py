from typing import Any

import pytest
from pydantic import ValidationError
from trace_schema.ir import SCHEMA_VERSION, TraceEvent


def _event(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "evt_1",
        "event_type": "model.invocation.completed",
        "evidence_class": "observed",
        "occurred_at": "2026-11-17T08:30:00Z",
        "run_id": "run_1",
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "model_invocation_id": "inv_1",
        "payload": {
            "prompt_sha256": "p" * 64,
            "output_sha256": "o" * 64,
            "prompt_artifact_id": None,
            "neural_access": "unavailable",
        },
        "provenance": {
            "model_revision": "rev",
            "tokenizer_revision": "not_applicable",
            "instrumentation_version": "phase2",
        },
    }
    return {**base, **overrides}


def test_valid_known_event() -> None:
    event = TraceEvent.model_validate(_event())
    assert event.is_known_type and event.evidence_class == "observed"


def test_model_event_requires_invocation_id() -> None:
    with pytest.raises(ValidationError, match="model_invocation_id"):
        TraceEvent.model_validate(_event(model_invocation_id=None))


@pytest.mark.parametrize("version", ["0.2.0", "1.0.0", "0.1", "0.1.x", "0.1.0-rc1"])
def test_unsupported_schema_version_is_rejected(version: str) -> None:
    with pytest.raises(ValidationError, match="SCHEMA_VERSION_UNSUPPORTED"):
        TraceEvent.model_validate(_event(schema_version=version))


def test_later_patch_version_keeps_unknown_fields_uninterpreted() -> None:
    event = TraceEvent.model_validate(_event(schema_version="0.1.7", new_field={"x": 1}))
    assert event.model_extra == {"new_field": {"x": 1}}


def test_unknown_event_type_is_accepted_with_raw_payload() -> None:
    event = TraceEvent.model_validate(_event(event_type="custom.thing", payload={"anything": 1}))
    assert not event.is_known_type and event.payload == {"anything": 1}


def test_known_type_rejects_bad_payload_and_wrong_evidence_class() -> None:
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(_event(payload={"prompt_sha256": "p" * 64}))
    with pytest.raises(ValidationError, match="evidence_class"):
        TraceEvent.model_validate(_event(evidence_class="measured"))


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TraceEvent.model_validate(_event(occurred_at="2026-11-17T08:30:00"))
