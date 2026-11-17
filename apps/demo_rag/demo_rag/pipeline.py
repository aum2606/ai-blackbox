"""Run one stale-document case end to end: retrieve, call the model, persist run + trace.

Registry calls use short separate transactions because the span exporter writes through its own
connection as spans end; one long write transaction would lock it out.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidence_store.artifacts import put_bytes
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode, Tracer
from run_registry import registry
from sqlalchemy import Engine
from trace_ingest.events import record_event
from trace_ingest.exporter import INVOCATION_ID_ATTR, RUN_ID_ATTR
from trace_schema.ir import SCHEMA_VERSION, EvidenceClass, Provenance, TraceEvent

from demo_rag import model
from demo_rag.backends import FIXTURE_BACKEND, Backend, Generation
from demo_rag.fixtures import DEFAULT_ROOT, load_case
from demo_rag.retriever import RetrievalFault, retrieve

TASK_ID = "stale_document"
CORPUS_ID = "stale_document_v1"
INSTRUMENTATION_VERSION = "phase2"
SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_DOC0 = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    trace_id: str
    status: str
    error_type: str | None


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _emit(
    engine: Engine,
    backend: Backend,
    *,
    event_type: str,
    run_id: str,
    trace_id: str,
    span_id: str,
    payload: dict[str, Any],
    invocation_id: str | None = None,
) -> None:
    """Record an ``observed`` AI Trace IR event for something the pipeline directly saw."""
    event = TraceEvent(
        schema_version=SCHEMA_VERSION,
        event_id=f"evt_{uuid.uuid4().hex}",
        event_type=event_type,
        evidence_class=EvidenceClass.OBSERVED,
        occurred_at=datetime.now(UTC),
        run_id=run_id,
        trace_id=trace_id,
        span_id=span_id,
        model_invocation_id=invocation_id,
        payload=payload,
        provenance=Provenance(
            model_revision=backend.model_revision,
            tokenizer_revision=backend.tokenizer_revision,
            instrumentation_version=INSTRUMENTATION_VERSION,
        ),
    )
    with engine.begin() as conn:
        record_event(conn, event)


def _store_token_sequences(
    engine: Engine,
    artifact_root: Path,
    backend: Backend,
    generation: Generation,
    invocation_id: str,
    prompt_artifact_id: str,
) -> None:
    """Token IDs reconstruct the prompt, so they are stored only on content opt-in."""
    for role, ids in (
        ("prompt", generation.prompt_token_ids),
        ("output", generation.output_token_ids),
    ):
        if ids is None:
            continue
        with engine.begin() as conn:
            artifact_id = put_bytes(
                conn,
                artifact_root,
                json.dumps({"token_ids": ids}).encode("utf-8"),
                mime_type="application/json",
                category="derived",
                parents=[prompt_artifact_id],
            )
            registry.record_token_sequence(
                conn,
                invocation_id=invocation_id,
                role=role,
                tokenizer_revision=backend.tokenizer_revision,
                n_tokens=len(ids),
                artifact_id=artifact_id,
            )


def run_case(
    engine: Engine,
    tracer: Tracer,
    artifact_root: Path,
    case_id: str,
    fault: RetrievalFault,
    *,
    capture_content: bool = False,
    fixture_root: Path = DEFAULT_ROOT,
    backend: Backend = FIXTURE_BACKEND,
) -> RunResult:
    # Fixture errors (unknown case, tampered files) propagate before any run is created.
    case = load_case(case_id, fixture_root)
    with engine.begin() as conn:
        document_artifacts = {
            d.version: put_bytes(
                conn, artifact_root, d.text.encode("utf-8"), mime_type="text/plain", category="raw"
            )
            for d in case.documents
        }
        document_ids = {
            d.version: registry.upsert_document(
                conn,
                corpus_id=CORPUS_ID,
                doc_key=case.doc_key,
                version=d.version,
                sha256=d.sha256,
                source_path=d.relative_path,
                valid_from=d.valid_from,
                valid_to=d.valid_to,
                artifact_id=document_artifacts[d.version],
            )
            for d in case.documents
        }
        run_id = registry.create_run(
            conn,
            task_id=TASK_ID,
            dataset_case_id=case_id,
            retrieval_fault=fault,
            capture_content=capture_content,
        )

    status, answer = "failed", None
    error_type: str | None = "INTERNAL_ERROR"
    trace_id = ""
    try:
        root_attrs = {
            RUN_ID_ATTR: run_id,
            SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "blackbox.case_id": case_id,
            "blackbox.retrieval_fault": fault,
        }
        with tracer.start_as_current_span("rag.run", attributes=root_attrs) as root:
            trace_id = f"{root.get_span_context().trace_id:032x}"
            with engine.begin() as conn:
                registry.attach_trace(conn, run_id, trace_id)
            try:
                with tracer.start_as_current_span(
                    "retriever.retrieve",
                    attributes={
                        RUN_ID_ATTR: run_id,
                        SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                    },
                ) as span:
                    retriever_span_id = f"{span.get_span_context().span_id:016x}"
                    document = retrieve(case, fault)
                    span.set_attributes(
                        {
                            f"{_DOC0}.{DocumentAttributes.DOCUMENT_ID}": document_ids[
                                document.version
                            ],
                            f"{_DOC0}.{DocumentAttributes.DOCUMENT_SCORE}": 1.0,
                            "blackbox.document_version": document.version,
                            "blackbox.document_sha256": document.sha256,
                        }
                    )
                    if capture_content:
                        span.set_attribute(SpanAttributes.INPUT_VALUE, case.question)
                        span.set_attribute(
                            f"{_DOC0}.{DocumentAttributes.DOCUMENT_CONTENT}", document.text
                        )

                _emit(
                    engine,
                    backend,
                    event_type="retrieval.document.selected",
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=retriever_span_id,
                    payload={
                        "document_id": document_ids[document.version],
                        "doc_key": case.doc_key,
                        "version": document.version,
                        "sha256": document.sha256,
                        "artifact_id": document_artifacts[document.version],
                        "rank": 1,
                        "score": 1.0,
                    },
                )

                prompt = model.render_prompt(document.text, case.question)
                # The invocation ID exists before the model call so it can be put on the span.
                with engine.begin() as conn:
                    # Prompt text is stored only on explicit opt-in (ADR-007).
                    prompt_artifact_id = (
                        put_bytes(
                            conn,
                            artifact_root,
                            prompt.encode("utf-8"),
                            mime_type="text/plain",
                            category="derived",
                            parents=[document_artifacts[document.version]],
                        )
                        if capture_content
                        else None
                    )
                    invocation_id = registry.create_invocation(
                        conn,
                        run_id=run_id,
                        model_id=backend.model_id,
                        model_revision=backend.model_revision,
                        tokenizer_revision=backend.tokenizer_revision,
                        prompt_sha256=_sha256(prompt),
                        sampling_config=backend.sampling_config,
                        neural_access=backend.neural_access,
                        prompt_artifact_id=prompt_artifact_id,
                    )
                llm_span_id = ""
                output_sha256 = None
                generation = Generation(answer="")
                try:
                    with tracer.start_as_current_span(
                        "llm.generate",
                        attributes={
                            RUN_ID_ATTR: run_id,
                            INVOCATION_ID_ATTR: invocation_id,
                            SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                            SpanAttributes.LLM_MODEL_NAME: backend.model_id,
                            "blackbox.prompt_sha256": _sha256(prompt),
                        },
                    ) as span:
                        llm_span_id = f"{span.get_span_context().span_id:016x}"
                        generation = backend.generate(prompt, document.text)
                        answer = generation.answer
                        output_sha256 = _sha256(answer)
                        span.set_attribute("blackbox.output_sha256", output_sha256)
                        if capture_content:
                            span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
                            span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
                finally:
                    with engine.begin() as conn:
                        registry.complete_invocation(
                            conn, invocation_id, span_id=llm_span_id, output_sha256=output_sha256
                        )
                    _emit(
                        engine,
                        backend,
                        event_type="model.invocation.completed",
                        run_id=run_id,
                        trace_id=trace_id,
                        span_id=llm_span_id,
                        invocation_id=invocation_id,
                        payload={
                            "prompt_sha256": _sha256(prompt),
                            "output_sha256": output_sha256,
                            "prompt_artifact_id": prompt_artifact_id,
                            "neural_access": backend.neural_access,
                        },
                    )
                if capture_content and prompt_artifact_id:
                    _store_token_sequences(
                        engine,
                        artifact_root,
                        backend,
                        generation,
                        invocation_id,
                        prompt_artifact_id,
                    )
                with engine.begin() as conn:
                    registry.record_retrieval(
                        conn,
                        run_id=run_id,
                        invocation_id=invocation_id,
                        document_id=document_ids[document.version],
                        rank=1,
                        score=1.0,
                    )
                status, error_type = "completed", None
            except model.ModelError as err:
                error_type = err.code
                root.set_status(Status(StatusCode.ERROR, err.code))
    finally:
        # Completed only after the root span ended, so a completed run always has its full trace.
        with engine.begin() as conn:
            registry.complete_run(
                conn,
                run_id,
                status=status,
                answer_text=answer if capture_content else None,
                error_type=error_type,
            )
    return RunResult(run_id=run_id, trace_id=trace_id, status=status, error_type=error_type)
