import json
import os
from pathlib import Path

from trace_schema.ir import TraceEvent

SNAPSHOT = (
    Path(__file__).resolve().parents[2] / "packages" / "trace_schema" / "ai_trace_ir_v0.1.json"
)


def test_json_schema_matches_checked_in_export() -> None:
    """Contract guard for AI Trace IR. Refresh deliberately with UPDATE_SNAPSHOT=1."""
    current = json.dumps(TraceEvent.model_json_schema(), indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOT"):
        SNAPSHOT.write_text(current, encoding="utf-8", newline="\n")
    assert current == SNAPSHOT.read_text(encoding="utf-8")
