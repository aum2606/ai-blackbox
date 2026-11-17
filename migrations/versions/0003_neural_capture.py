"""Phase 3: token sequences, capture jobs and measurements.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"

_UP = [
    """CREATE TABLE token_sequences (
      id TEXT PRIMARY KEY,
      invocation_id TEXT NOT NULL REFERENCES model_invocations(id),
      role TEXT NOT NULL CHECK (role IN ('prompt','output')),
      tokenizer_revision TEXT NOT NULL,
      n_tokens INTEGER NOT NULL,
      artifact_id TEXT NOT NULL REFERENCES artifacts(id),
      created_at TEXT NOT NULL,
      UNIQUE (invocation_id, role)
    )""",
    """CREATE TABLE capture_jobs (
      id TEXT PRIMARY KEY,
      invocation_id TEXT NOT NULL REFERENCES model_invocations(id),
      plan_json TEXT NOT NULL,
      plan_sha256 TEXT NOT NULL,
      status TEXT NOT NULL
        CHECK (status IN ('queued','running','completed','failed','blocked_resource')),
      error_type TEXT,
      replay_json TEXT,
      created_at TEXT NOT NULL,
      started_at TEXT,
      ended_at TEXT
    )""",
    """CREATE TABLE measurements (
      id TEXT PRIMARY KEY,
      capture_job_id TEXT NOT NULL REFERENCES capture_jobs(id),
      invocation_id TEXT NOT NULL REFERENCES model_invocations(id),
      kind TEXT NOT NULL CHECK (kind IN ('activation','logits')),
      hook_name TEXT NOT NULL,
      layer INTEGER,
      token_position INTEGER NOT NULL,
      artifact_id TEXT NOT NULL REFERENCES artifacts(id),
      dtype TEXT NOT NULL,
      shape_json TEXT NOT NULL,
      summary_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_capture_jobs_invocation ON capture_jobs(invocation_id)",
    "CREATE INDEX idx_capture_jobs_status ON capture_jobs(status)",
    "CREATE INDEX idx_measurements_invocation ON measurements(invocation_id)",
    "CREATE INDEX idx_measurements_job ON measurements(capture_job_id)",
]
_DOWN_TABLES = ["measurements", "capture_jobs", "token_sequences"]  # dropping drops the indexes


def upgrade() -> None:
    for statement in _UP:
        op.execute(statement)


def downgrade() -> None:
    for table in _DOWN_TABLES:
        op.execute(f"DROP TABLE {table}")
