"""Persist AI Trace IR events. Validation already happened when the TraceEvent was built."""

import json

from sqlalchemy import Connection, text
from trace_schema.ir import TraceEvent


def record_event(conn: Connection, event: TraceEvent) -> None:
    """Store one event; replaying the same ``event_id`` is a no-op."""
    conn.execute(
        text(
            "INSERT INTO trace_events (event_id, schema_version, event_type, evidence_class,"
            " occurred_at, run_id, trace_id, span_id, model_invocation_id, known_type,"
            " location_json, payload_json, provenance_json, extra_json) VALUES (:event_id,"
            " :schema_version, :event_type, :evidence_class, :occurred_at, :run_id, :trace_id,"
            " :span_id, :invocation, :known, :location, :payload, :provenance, :extra)"
            " ON CONFLICT (event_id) DO NOTHING"
        ),
        {
            "event_id": event.event_id,
            "schema_version": event.schema_version,
            "event_type": event.event_type,
            "evidence_class": event.evidence_class.value,
            "occurred_at": event.occurred_at.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "run_id": event.run_id,
            "trace_id": event.trace_id,
            "span_id": event.span_id,
            "invocation": event.model_invocation_id,
            "known": int(event.is_known_type),
            "location": event.location.model_dump_json() if event.location else None,
            "payload": json.dumps(event.payload, sort_keys=True),
            "provenance": event.provenance.model_dump_json(),
            "extra": json.dumps(event.model_extra, sort_keys=True) if event.model_extra else None,
        },
    )
