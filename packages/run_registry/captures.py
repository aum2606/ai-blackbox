"""Capture job lifecycle and measurement records.

States: queued -> running -> completed | failed | blocked_resource. A finished job is never
reopened; a retry is a new job (ADD §6.4). The worker is a single process, so claiming is one
conditional UPDATE and any ``running`` row found at worker start is stale.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, text

ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class JobRecord:
    id: str
    invocation_id: str
    plan_json: str
    plan_sha256: str
    status: str
    error_type: str | None
    replay: dict[str, Any] | None
    created_at: str
    started_at: str | None
    ended_at: str | None


@dataclass(frozen=True)
class MeasurementRecord:
    id: str
    capture_job_id: str
    invocation_id: str
    kind: str
    hook_name: str
    layer: int | None
    token_position: int
    artifact_id: str
    dtype: str
    shape: list[int]
    summary: dict[str, Any]
    created_at: str


def _job(row: Any) -> JobRecord:
    return JobRecord(
        id=row.id,
        invocation_id=row.invocation_id,
        plan_json=row.plan_json,
        plan_sha256=row.plan_sha256,
        status=row.status,
        error_type=row.error_type,
        replay=None if row.replay_json is None else json.loads(row.replay_json),
        created_at=row.created_at,
        started_at=row.started_at,
        ended_at=row.ended_at,
    )


def create_job(
    conn: Connection, *, invocation_id: str, plan_json: str, plan_sha256: str
) -> tuple[JobRecord, bool]:
    """Queue a capture; an identical queued or running request returns the existing job."""
    existing = conn.execute(
        text(
            "SELECT * FROM capture_jobs WHERE invocation_id = :inv AND plan_sha256 = :sha"
            " AND status IN ('queued','running') ORDER BY created_at LIMIT 1"
        ),
        {"inv": invocation_id, "sha": plan_sha256},
    ).first()
    if existing is not None:
        return _job(existing), False
    job_id = f"cap_{uuid.uuid4().hex}"
    conn.execute(
        text(
            "INSERT INTO capture_jobs (id, invocation_id, plan_json, plan_sha256, status,"
            " created_at) VALUES (:id, :inv, :plan, :sha, 'queued', :now)"
        ),
        {"id": job_id, "inv": invocation_id, "plan": plan_json, "sha": plan_sha256, "now": _now()},
    )
    created = get_job(conn, job_id)
    assert created is not None  # noqa: S101 - just inserted in this transaction
    return created, True


def get_job(conn: Connection, job_id: str) -> JobRecord | None:
    row = conn.execute(text("SELECT * FROM capture_jobs WHERE id = :id"), {"id": job_id}).first()
    return None if row is None else _job(row)


def recover_stale(conn: Connection) -> int:
    """Single worker: a row still ``running`` at start belongs to an interrupted process."""
    return conn.execute(
        text(
            "UPDATE capture_jobs SET status = 'failed', error_type = 'WORKER_INTERRUPTED',"
            " ended_at = :now WHERE status = 'running'"
        ),
        {"now": _now()},
    ).rowcount


def claim_next(conn: Connection) -> JobRecord | None:
    row = conn.execute(
        text("SELECT id FROM capture_jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1")
    ).first()
    if row is None:
        return None
    claimed = conn.execute(
        text(
            "UPDATE capture_jobs SET status = 'running', started_at = :now"
            " WHERE id = :id AND status = 'queued'"
        ),
        {"now": _now(), "id": row.id},
    ).rowcount
    return get_job(conn, row.id) if claimed == 1 else None


def complete_job(conn: Connection, job_id: str, replay: dict[str, Any]) -> None:
    _finish(conn, job_id, "completed", None, json.dumps(replay, sort_keys=True))


def fail_job(conn: Connection, job_id: str, error_type: str, *, blocked_resource: bool) -> None:
    _finish(conn, job_id, "blocked_resource" if blocked_resource else "failed", error_type, None)


def _finish(
    conn: Connection, job_id: str, status: str, error_type: str | None, replay_json: str | None
) -> None:
    changed = conn.execute(
        text(
            "UPDATE capture_jobs SET status = :status, error_type = :error, replay_json = :replay,"
            " ended_at = :now WHERE id = :id AND status = 'running'"
        ),
        {"status": status, "error": error_type, "replay": replay_json, "now": _now(), "id": job_id},
    ).rowcount
    if changed != 1:
        raise RuntimeError(f"{job_id} is not a running job")


def record_measurement(
    conn: Connection,
    *,
    job_id: str,
    invocation_id: str,
    kind: str,
    hook_name: str,
    layer: int | None,
    token_position: int,
    artifact_id: str,
    dtype: str,
    shape: list[int],
    summary: dict[str, Any],
) -> str:
    measurement_id = f"mea_{uuid.uuid4().hex}"
    conn.execute(
        text(
            "INSERT INTO measurements (id, capture_job_id, invocation_id, kind, hook_name, layer,"
            " token_position, artifact_id, dtype, shape_json, summary_json, created_at) VALUES"
            " (:id, :job, :inv, :kind, :hook, :layer, :pos, :art, :dtype, :shape, :summary, :now)"
        ),
        {
            "id": measurement_id,
            "job": job_id,
            "inv": invocation_id,
            "kind": kind,
            "hook": hook_name,
            "layer": layer,
            "pos": token_position,
            "art": artifact_id,
            "dtype": dtype,
            "shape": json.dumps(shape),
            "summary": json.dumps(summary, sort_keys=True),
            "now": _now(),
        },
    )
    return measurement_id


def list_measurements(conn: Connection, invocation_id: str) -> tuple[MeasurementRecord, ...]:
    rows = conn.execute(
        text(
            "SELECT m.* FROM measurements m JOIN capture_jobs j ON j.id = m.capture_job_id"
            " WHERE m.invocation_id = :inv AND j.status = 'completed'"
            " ORDER BY m.capture_job_id, m.hook_name, m.token_position, m.kind"
        ),
        {"inv": invocation_id},
    )
    return tuple(
        MeasurementRecord(
            id=r.id,
            capture_job_id=r.capture_job_id,
            invocation_id=r.invocation_id,
            kind=r.kind,
            hook_name=r.hook_name,
            layer=r.layer,
            token_position=r.token_position,
            artifact_id=r.artifact_id,
            dtype=r.dtype,
            shape=json.loads(r.shape_json),
            summary=json.loads(r.summary_json),
            created_at=r.created_at,
        )
        for r in rows
    )


@dataclass(frozen=True)
class TokenSequenceRecord:
    role: str
    n_tokens: int
    artifact_id: str


def token_sequences(conn: Connection, invocation_id: str) -> dict[str, TokenSequenceRecord]:
    rows = conn.execute(
        text("SELECT role, n_tokens, artifact_id FROM token_sequences WHERE invocation_id = :inv"),
        {"inv": invocation_id},
    )
    return {r.role: TokenSequenceRecord(r.role, r.n_tokens, r.artifact_id) for r in rows}
