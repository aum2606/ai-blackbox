"""Content-addressed, checksum-verified artifact store on the local filesystem.

Bytes live at ``<root>/sha256/<aa>/<sha256>``; the ``artifacts`` row is the only way to find them,
so no caller depends on a mutable filename (ADD §5.4). Functions take a ``Connection`` so the caller
owns the transaction.
"""

import hashlib
import os
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Connection, text

from evidence_store.engine import max_artifact_bytes


class ArtifactError(Exception):
    """Artifact problem; ``code`` is the structured error type reported by the API."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def put_bytes(
    conn: Connection,
    root: Path,
    payload: bytes,
    *,
    mime_type: str,
    category: str,
    parents: Sequence[str] = (),
    dtype: str | None = None,
    shape_json: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Store ``payload`` once and return its artifact ID; identical bytes give the same ID."""
    sha256 = hashlib.sha256(payload).hexdigest()
    existing = conn.execute(
        text("SELECT id FROM artifacts WHERE sha256 = :sha"), {"sha": sha256}
    ).scalar_one_or_none()
    if existing is not None:
        artifact_id = str(existing)
    else:
        limit = max_artifact_bytes() if max_bytes is None else max_bytes
        used = conn.execute(text("SELECT COALESCE(SUM(nbytes), 0) FROM artifacts")).scalar_one()
        if used + len(payload) > limit:
            raise ArtifactError(
                "ARTIFACT_QUOTA_EXCEEDED", f"{used + len(payload)} bytes would exceed {limit}"
            )
        relative_path = f"sha256/{sha256[:2]}/{sha256}"
        _write_atomically(root / relative_path, payload, sha256)
        artifact_id = f"art_{uuid.uuid4().hex}"
        conn.execute(
            text(
                "INSERT INTO artifacts (id, sha256, relative_path, mime_type, nbytes, category,"
                " dtype, shape_json, created_at) VALUES (:id, :sha, :path, :mime, :nbytes,"
                " :category, :dtype, :shape, :now)"
            ),
            {
                "id": artifact_id,
                "sha": sha256,
                "path": relative_path,
                "mime": mime_type,
                "nbytes": len(payload),
                "category": category,
                "dtype": dtype,
                "shape": shape_json,
                "now": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            },
        )
    for parent_id in parents:
        conn.execute(
            text(
                "INSERT INTO artifact_lineage (child_id, parent_id, relation)"
                " VALUES (:child, :parent, 'derived_from') ON CONFLICT DO NOTHING"
            ),
            {"child": artifact_id, "parent": parent_id},
        )
    return artifact_id


def _write_atomically(destination: Path, payload: bytes, sha256: str) -> None:
    """Temp file, fsync, re-hash, rename: a reader never sees a partial or wrong file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if hashlib.sha256(temp.read_bytes()).hexdigest() != sha256:
            raise ArtifactError("ARTIFACT_INTEGRITY_ERROR", "bytes changed while being written")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def get_verified(conn: Connection, root: Path, artifact_id: str) -> bytes:
    """Read an artifact and re-hash it on every read."""
    row = conn.execute(
        text("SELECT sha256, relative_path FROM artifacts WHERE id = :id"), {"id": artifact_id}
    ).one_or_none()
    if row is None:
        raise ArtifactError("ARTIFACT_NOT_FOUND", f"unknown artifact {artifact_id!r}")
    root = root.resolve()
    path = (root / str(row.relative_path)).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError("ARTIFACT_PATH_INVALID", f"{artifact_id} path escapes the store root")
    try:
        data = path.read_bytes()
    except FileNotFoundError as err:
        raise ArtifactError("ARTIFACT_INTEGRITY_ERROR", f"{artifact_id} file is missing") from err
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise ArtifactError("ARTIFACT_INTEGRITY_ERROR", f"{artifact_id} does not match its hash")
    return data
