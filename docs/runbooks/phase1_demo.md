# Runbook: Phase 1 demo (observable stale-document run)

Everything runs on localhost with no cloud account, credentials or paid service. Commands are shown for a bash-style shell; in PowerShell set variables with `$env:NAME = "value"`.

## Requirements

[uv](https://docs.astral.sh/uv/) (it installs Python 3.12 as pinned) and Node.js 20+. Internet is needed only to install packages.

## Setup (once per clone)

```bash
uv sync
npm ci --prefix apps/dashboard
uv run alembic upgrade head
```

The database defaults to `./.local/blackbox.db` (git-ignored). Override it with the `DATABASE_URL` environment variable. `.env` files are **not** read automatically; export the variable in your shell. Do not keep `.local/` inside a cloud-synced folder (OneDrive, Dropbox): SQLite WAL files can be corrupted by sync.

## Run the demo

1. API (localhost only):

   ```bash
   uv run uvicorn blackbox_api.app:app_from_env --factory --host 127.0.0.1 --port 8000
   ```

2. Create a run with the injected stale-version fault (in another terminal). `--capture-content` stores the answer text; leave it off to store hashes only:

   ```bash
   uv run python -m demo_rag --case case_0007 --fault stale_version --capture-content
   ```

3. Dashboard:

   ```bash
   npm run dev --prefix apps/dashboard
   ```

   Open the `view:` URL the CLI printed (`http://127.0.0.1:5173/?run=<run_id>`).

## Exit-test walkthrough

On the run page, confirm in order:

1. **Answer** shows `500 units`, expected `800 units`, result "✖ Incorrect".
2. **Retrieved document** shows `plan_delta, version 1`, "⚠ Superseded version (valid 2025-01-01 to 2026-09-01)" and its SHA-256.
3. **Model invocation** shows the invocation ID, `fixture-extractive`, the prompt and output hashes, the linked span ID and "Neural access: unavailable".
4. **Trace** lists `rag.run` → `retriever.retrieve` and `llm.generate`, all sharing the trace ID.

Run the same case with `--fault none` and the answer is `800 units`, "✔ Correct", from version 2.

The same chain is checked automatically by `tests/e2e/test_exit_test.py`.

## Checks

```bash
uv run ruff format --check . && uv run ruff check .
uv run mypy apps/api/blackbox_api apps/demo_rag/demo_rag packages/evidence_store packages/run_registry packages/trace_ingest tests
uv run pytest
npm run lint --prefix apps/dashboard && npm run build --prefix apps/dashboard
```

After an intentional API change, refresh the contract snapshot with `UPDATE_SNAPSHOT=1 uv run pytest tests/contract` and update `apps/dashboard/src/types.ts` to match.

## Backup and restore

Stop the API first, then copy the `.local/` folder (the database and its `-wal`/`-shm` files travel together). To restore, stop the API, replace `.local/` with the copy and restart. From Phase 2 on, `.local/` also holds the artifact files; see the [Phase 2 runbook](phase2_evidence.md).

## Privacy defaults

Prompt, answer and document text are not stored unless a run is created with content capture on (`--capture-content`, or `"capture_content": true` in `POST /v1/runs`). Without it, only hashes, IDs and span timings are kept. The fixtures are synthetic.

## Limitations

- The "model" is a deterministic fixture (ADR-009). Phase 1 shows trace and provenance correlation, **not** model behaviour or internal reasoning. No activations are captured.
- The run executes inside the API request. That is acceptable only because the fixture model is instant; Phase 3 moves real model work to a worker process.
- The model revision is a hash of `model.py`, so it can differ between checkouts with different line endings.
- Fixture files are hashed byte for byte; `.gitattributes` stops git from converting their line endings. Do not edit them by hand without refreshing `experiments/stale_document/manifest.json`.
- Optional providers (Phoenix, free notebooks) are not used in Phase 1; their limits are variable and not promised.
