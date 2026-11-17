"""Phase 1 core tables: runs, documents, invocations, retrievals, spans.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None

# Timestamps are ISO-8601 UTC text and JSON is validated text, both written by the application.
_UP = [
    """CREATE TABLE runs (
      id TEXT PRIMARY KEY,
      task_id TEXT NOT NULL,
      dataset_case_id TEXT NOT NULL,
      retrieval_fault TEXT NOT NULL CHECK (retrieval_fault IN ('none','stale_version')),
      trace_id TEXT,
      status TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
      capture_content INTEGER NOT NULL CHECK (capture_content IN (0,1)),
      answer_text TEXT,
      error_type TEXT,
      started_at TEXT NOT NULL,
      ended_at TEXT
    )""",
    """CREATE TABLE documents (
      id TEXT PRIMARY KEY,
      corpus_id TEXT NOT NULL,
      doc_key TEXT NOT NULL,
      version INTEGER NOT NULL,
      sha256 TEXT NOT NULL,
      source_path TEXT NOT NULL,
      valid_from TEXT NOT NULL,
      valid_to TEXT,
      UNIQUE (doc_key, version)
    )""",
    """CREATE TABLE model_invocations (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(id),
      span_id TEXT,
      model_id TEXT NOT NULL,
      model_revision TEXT NOT NULL,
      tokenizer_revision TEXT NOT NULL,
      prompt_sha256 TEXT NOT NULL,
      sampling_config_json TEXT NOT NULL,
      output_sha256 TEXT,
      neural_access TEXT NOT NULL CHECK (neural_access IN ('available','unavailable')),
      created_at TEXT NOT NULL
    )""",
    """CREATE TABLE retrieval_results (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(id),
      invocation_id TEXT NOT NULL REFERENCES model_invocations(id),
      document_id TEXT NOT NULL REFERENCES documents(id),
      rank INTEGER NOT NULL,
      score REAL NOT NULL,
      created_at TEXT NOT NULL
    )""",
    """CREATE TABLE spans (
      span_id TEXT PRIMARY KEY,
      trace_id TEXT NOT NULL,
      parent_span_id TEXT,
      run_id TEXT REFERENCES runs(id),
      name TEXT NOT NULL,
      kind TEXT NOT NULL,
      openinference_span_kind TEXT,
      status_code TEXT NOT NULL,
      start_time TEXT NOT NULL,
      end_time TEXT NOT NULL,
      attributes_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_invocations_run ON model_invocations(run_id)",
    "CREATE INDEX idx_retrievals_run ON retrieval_results(run_id)",
    "CREATE INDEX idx_spans_trace ON spans(trace_id)",
    "CREATE INDEX idx_spans_run ON spans(run_id)",
]
_TABLES_CHILD_FIRST = ["spans", "retrieval_results", "model_invocations", "documents", "runs"]


def upgrade() -> None:
    for statement in _UP:
        op.execute(statement)


def downgrade() -> None:
    for table in _TABLES_CHILD_FIRST:  # dropping a table drops its indexes
        op.execute(f"DROP TABLE {table}")
