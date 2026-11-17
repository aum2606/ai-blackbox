"""Phase 2: artifacts, artifact lineage, AI Trace IR events and artifact references.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"

_UP = [
    """CREATE TABLE artifacts (
      id TEXT PRIMARY KEY,
      sha256 TEXT NOT NULL UNIQUE,
      relative_path TEXT NOT NULL,
      mime_type TEXT NOT NULL,
      nbytes INTEGER NOT NULL,
      category TEXT NOT NULL CHECK (category IN ('raw','redacted','derived')),
      dtype TEXT,
      shape_json TEXT,
      retention_until TEXT,
      created_at TEXT NOT NULL
    )""",
    """CREATE TABLE artifact_lineage (
      child_id TEXT NOT NULL REFERENCES artifacts(id),
      parent_id TEXT NOT NULL REFERENCES artifacts(id),
      relation TEXT NOT NULL,
      PRIMARY KEY (child_id, parent_id, relation)
    )""",
    """CREATE TABLE trace_events (
      event_id TEXT PRIMARY KEY,
      schema_version TEXT NOT NULL,
      event_type TEXT NOT NULL,
      evidence_class TEXT NOT NULL
        CHECK (evidence_class IN ('observed','measured','inferred','intervened')),
      occurred_at TEXT NOT NULL,
      run_id TEXT NOT NULL REFERENCES runs(id),
      trace_id TEXT NOT NULL,
      span_id TEXT NOT NULL,
      model_invocation_id TEXT REFERENCES model_invocations(id),
      known_type INTEGER NOT NULL CHECK (known_type IN (0,1)),
      location_json TEXT,
      payload_json TEXT NOT NULL,
      provenance_json TEXT NOT NULL,
      extra_json TEXT
    )""",
    "CREATE INDEX idx_events_run ON trace_events(run_id)",
    "CREATE INDEX idx_events_invocation ON trace_events(model_invocation_id)",
    "ALTER TABLE documents ADD COLUMN artifact_id TEXT REFERENCES artifacts(id)",
    "ALTER TABLE model_invocations ADD COLUMN prompt_artifact_id TEXT REFERENCES artifacts(id)",
]
_DOWN = [
    "ALTER TABLE model_invocations DROP COLUMN prompt_artifact_id",
    "ALTER TABLE documents DROP COLUMN artifact_id",
    "DROP TABLE trace_events",  # drops its indexes
    "DROP TABLE artifact_lineage",
    "DROP TABLE artifacts",
]


def upgrade() -> None:
    for statement in _UP:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWN:
        op.execute(statement)
