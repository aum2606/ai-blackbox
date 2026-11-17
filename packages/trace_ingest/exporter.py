"""OpenTelemetry spans -> SQLite. No collector or Phoenix server is needed (ADR-003, ADD §6.1)."""

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

# Set on every span so spans can be joined to their run and invocation.
RUN_ID_ATTR = "blackbox.run_id"
INVOCATION_ID_ATTR = "blackbox.model_invocation_id"

_log = logging.getLogger(__name__)


def _iso(unix_ns: int | None) -> str:
    moment = datetime.fromtimestamp((unix_ns or 0) / 1e9, UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SQLiteSpanExporter(SpanExporter):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = []
        for span in spans:
            ctx = span.get_span_context()
            attributes = dict(span.attributes or {})
            rows.append(
                {
                    "span_id": f"{ctx.span_id:016x}" if ctx else "",
                    "trace_id": f"{ctx.trace_id:032x}" if ctx else "",
                    "parent": f"{span.parent.span_id:016x}" if span.parent else None,
                    "run_id": attributes.get(RUN_ID_ATTR),
                    "name": span.name,
                    "kind": span.kind.name,
                    "oi_kind": attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND),
                    "status": span.status.status_code.name,
                    "start": _iso(span.start_time),
                    "end": _iso(span.end_time),
                    "attributes": json.dumps(attributes, sort_keys=True, default=str),
                }
            )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO spans (span_id, trace_id, parent_span_id, run_id, name, kind,"
                        " openinference_span_kind, status_code, start_time, end_time,"
                        " attributes_json) VALUES (:span_id, :trace_id, :parent, :run_id, :name,"
                        " :kind, :oi_kind, :status, :start, :end, :attributes)"
                        " ON CONFLICT (span_id) DO NOTHING"
                    ),
                    rows,
                )
        except SQLAlchemyError:
            _log.exception("span export failed")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS


def make_tracer_provider(engine: Engine) -> TracerProvider:
    """Provider that persists each span as it ends, so a trace is complete when the run returns.

    Deliberately not installed as the global provider; callers use ``provider.get_tracer(...)``.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(engine)))
    return provider
