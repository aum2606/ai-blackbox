# Runbook: Phase 2 (AI Trace IR and evidence store)

Setup, running the demo and the checks are the same as in the [Phase 1 runbook](phase1_demo.md). After pulling Phase 2, run `uv run alembic upgrade head` once to apply migration 0002.

## What Phase 2 adds

- **Artifacts:** every retrieved fixture document is stored as a content-addressed file under `ARTIFACT_ROOT` (default `./.local/artifacts`, layout `sha256/<aa>/<sha256>`). A row in `artifacts` is the only way to find a file, and each read re-hashes it.
- **Prompt artifact (opt-in):** with `--capture-content` (or `"capture_content": true`), the rendered prompt is also stored, linked to the document artifact it came from (`artifact_lineage`, relation `derived_from`). Without the opt-in no prompt bytes are written anywhere.
- **AI Trace IR events:** each run records two `observed` events in `trace_events` (`retrieval.document.selected`, `model.invocation.completed`), validated against the v0.1 schema (`packages/trace_schema/ai_trace_ir_v0.1.json`).
- **API:** `GET /v1/artifacts/{id}` returns metadata plus `verified` (a fresh checksum check). Documents carry `artifact_id` and invocations carry `prompt_artifact_id`. Raw bytes are never returned.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARTIFACT_ROOT` | `./.local/artifacts` | Where artifact files live |
| `MAX_LOCAL_ARTIFACT_BYTES` | `1073741824` (1 GiB) | Quota across all artifacts; a write that would exceed it fails with `ARTIFACT_QUOTA_EXCEEDED` before anything is written |

Like `DATABASE_URL`, these are read from the process environment, not from `.env`.

## Exit-test walkthrough

1. Create a run with capture on: `uv run python -m demo_rag --case case_0007 --fault stale_version --capture-content`.
2. Open the run page. Under **Retrieved document** the "Stored artifact" line and under **Model invocation** the "Prompt artifact" line both show "✔ checksum verified".
3. Corrupt a file under `.local/artifacts/` (change any byte) and reload: the matching line now reads "✖ integrity check failed: ARTIFACT_INTEGRITY_ERROR".

The same chain is checked by `tests/e2e/test_phase2_exit.py`.

## Backup and restore

Stop the API first, then copy the whole `.local/` folder: the database (with its `-wal`/`-shm` files) **and** `artifacts/`. They must be restored together, because rows reference files by hash. After restoring, open any run page or call `GET /v1/artifacts/{id}`: a missing or altered file shows up as `verified: false`.

## Limitations

- Token IDs, measurements and experiment manifests are not stored yet; they arrive with their producers in Phases 3 and 5 (ADR-010).
- Events are stored but not exposed by an API endpoint yet.
- Retention is recorded (`artifacts.retention_until`) but not enforced; there is no deletion or audit trail yet (Phase 7).
- The quota is the sum of stored artifact sizes in the database. Files left behind by a crash between the file write and the database insert are not counted, and are harmless because names are content hashes.
- Older databases keep working: documents created in Phase 1 get their `artifact_id` attached the next time the same case runs.
