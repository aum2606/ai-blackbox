"""Read side of the span store. Attributes are deliberately not returned: they may hold content."""

from dataclasses import dataclass

from sqlalchemy import Connection, text


@dataclass(frozen=True)
class SpanRecord:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    openinference_span_kind: str | None
    status_code: str
    start_time: str
    end_time: str


def list_spans(conn: Connection, run_id: str) -> tuple[SpanRecord, ...]:
    # Parents start first and end last, so this order lists a parent before its children.
    rows = conn.execute(
        text(
            "SELECT span_id, trace_id, parent_span_id, name, openinference_span_kind, status_code,"
            " start_time, end_time FROM spans WHERE run_id = :id"
            " ORDER BY start_time, end_time DESC, span_id"
        ),
        {"id": run_id},
    )
    return tuple(SpanRecord(**row._mapping) for row in rows)
