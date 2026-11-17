# Phase 1 Implementation Plan — Observable RAG Demonstration

> **Status:** Draft for maintainer review
> **Date:** 2026-11-17
> **Governs:** `rules_v1.1.md` (mandatory) and `AI_Black_Box_Lab_Architecture_v1.1_Free_Tier.md` (ADD v1.1), §13 Phase 1 and §17 build order.
> **Phase goal (ADD §13):** A complete trace of a controlled wrong answer.
> **Exit test (ADD §13):** Starting from an answer, navigate to the exact retrieved document version and its model invocation; every run's IDs are consistent.

---

## 1. Starting point and assumptions

- The repository holds only the ADD, the rules and `skills-lock.json`. It is **not a git repository** yet, and nothing from Phase 0 exists.
- Local toolchain: Python 3.14.3, uv 0.11.6, Node 24.13.0, npm 11.6.2, git 2.53. No GPU is assumed.
- Phase 1 needs only part of Phase 0. That part is **Step 0** below. The Phase 0 model/SAE spikes are **not** prerequisites for Phase 1 (see Decision D1).

## 2. Scope

### In scope (from ADD §13 Phase 1, nothing more)

1. A deterministic retriever over frozen, versioned synthetic documents.
2. `run_id`, `model_invocation_id` and document-version records.
3. OpenTelemetry spans with OpenInference attributes for the run, the retrieval and the model call.
4. An in-process span exporter that writes span metadata to SQLite. Exporting to Phoenix stays an optional configuration switch and is not built in Phase 1.
5. `POST /v1/runs`, `GET /v1/runs/{run_id}`, `GET /v1/invocations/{id}`, plus `GET /v1/health`.
6. A minimal first-party run detail page.
7. Trace-completeness tests and a known stale-document failure fixture.

### Explicitly out of scope (later phases)

| Not built in Phase 1 | Belongs to |
|---|---|
| AI Trace IR event envelope, JSON Schema export, `evidence_class` on events | Phase 2 |
| Artifact store, checksummed tensor files, quotas | Phase 2 |
| Token IDs, tokenizer metadata, prompt snapshots | Phase 2 / 3 |
| Neural capture, worker CLI, job state machine | Phase 3 |
| SAE, experiments, reports, `GET /v1/runs` list, `/v1/capabilities` | Phases 4–6 |
| Phoenix integration, Docker Compose, CI workflow | Optional; not needed for the exit test |
| Auth | Not needed while bound to localhost (rules §6, ADD §10.2.3) |

## 3. Decisions to confirm before coding

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | **What model answers in Phase 1?** | A deterministic **extractive fixture model** (`model_id="fixture-extractive"`, `model_revision` = SHA-256 of its source file, `tokenizer_revision="not_applicable"`). It reads the cap value out of the retrieved document. The run viewer shows **"neural access unavailable"** (ADD §4.1). | The Phase 1 failure is *retrieval-induced*, so a deterministic reader reproduces it exactly. A tiny HF model may not answer the numeric question reliably (ADD §8.2 confound policy). Phase 3 then adds the tiny pinned model behind the same gateway function. That second implementation justifies the boundary, and no speculative interface is needed now. |
| D2 | **Python version** | Pin **3.12** with `uv python pin 3.12`. | Phase 3+ needs PyTorch/TransformerLens. 3.12 is the safest supported target, and 3.14 support in research libraries is unverified. Phase 0 can confirm this. |
| D3 | **Persistence layer** | SQLAlchemy **Core** (no ORM) plus Alembic. | Alembic depends on SQLAlchemy anyway. Core gives parameterized queries without ORM model classes duplicating the Pydantic models. |
| D4 | **Type checker** | `mypy --strict` on `packages/` and `apps/api`. | The rules require "the repository's configured type checker", so one must be chosen. Either mypy or pyright meets the rule. |
| D5 | **Response/document body capture** | Off by default. Each `POST /v1/runs` takes `capture_content: bool = false`. The demo script passes `true` explicitly, because fixtures are synthetic. | Rules §6 and ADR-007 require opt-in. The exit test needs a visible answer, and an explicit opt-in satisfies both. |
| D6 | **Rules filename** | Keep `rules_v1.1.md` and link it from the README. Do not rename it to `rules.md`. | This avoids a gratuitous change. Note the mismatch with ADD §12.1. |

## 4. Target layout after Phase 1

Only directories that Phase 1 actually uses are created (rules §1: no empty scaffolding).

```text
ai_blackbox/
├── apps/
│   ├── api/blackbox_api/            # FastAPI app, routes, response models
│   ├── dashboard/                   # React/Vite + TS, run detail page only
│   └── demo_rag/demo_rag/           # retriever, fixture model, pipeline, CLI
├── packages/
│   ├── run_registry/                # run + invocation + retrieval records (SQL Core)
│   ├── trace_ingest/                # SQLite SpanExporter, span queries
│   └── evidence_store/              # engine factory (WAL + FK pragmas), documents table access
├── experiments/stale_document/
│   ├── cases/case_0001..0010.json   # question, expected answer, doc refs
│   └── documents/<doc_id>/v<N>.txt  # frozen document versions
├── migrations/                      # Alembic env + 0001_phase1_core
├── docs/{adr,research-protocols,runbooks}/
├── tests/{unit,integration,contract,e2e}/
├── pyproject.toml  uv.lock  .env.example  .gitignore  README.md
```

`packages/trace_schema` is **not** created in Phase 1. Its owner is Phase 2 (AI Trace IR). The Phase 1 API response models live in `apps/api` because they are API contracts, not IR.

Packaging: one `pyproject.toml` with a single uv project, and the packages listed as source roots. A uv workspace with per-package pyprojects would add structure nobody needs yet.

## 5. Data model (migration `0001_phase1_core`)

Fields follow ADD §5.3. Columns that only later phases need are left out and arrive with their migrations.

```text
runs                (id PK, task_id, dataset_case_id, retrieval_fault, trace_id,
                     status CHECK(running|completed|failed), capture_content,
                     answer_text NULL, error_type NULL, started_at, ended_at NULL)
documents           (id PK, corpus_id, doc_key, version, sha256, source_path,
                     valid_from, valid_to NULL, UNIQUE(doc_key, version))
model_invocations   (id PK, run_id FK runs, span_id, model_id, model_revision,
                     tokenizer_revision, prompt_sha256, sampling_config_json,
                     output_sha256 NULL, neural_access TEXT CHECK(available|unavailable),
                     created_at)
retrieval_results   (id PK, run_id FK, invocation_id FK model_invocations,
                     document_id FK documents, rank, score, created_at)
spans               (span_id PK, trace_id, parent_span_id NULL, run_id FK NULL,
                     name, kind, openinference_span_kind, status_code,
                     start_time, end_time, attributes_json)
INDEX on model_invocations(run_id), retrieval_results(run_id), spans(trace_id), spans(run_id)
```

Notes:

- `spans` is a Phase 1 addition that ADD §5.3 does not list. It is needed by "persist OTel span metadata in SQLite". It is owned by `trace_ingest`. `attributes_json` stores whatever the span carries; content attributes are only ever set at the source when the run opted in (§7).
- `documents.sha256` covers the exact fixture bytes. Fixtures are loaded and their hashes verified at startup or seed time, and the run fails if a hash mismatches (rules §5: never mutate a reference fixture).
- `documents.source_path` is a repo-relative fixture path. It is resolved under `experiments/stale_document/documents/` with a traversal check. There is no artifact store yet. Fixtures are committed, synthetic and small.
- `answer_text` is NULL unless `capture_content` is true. `output_sha256` is always stored.
- All timestamps are ISO-8601 UTC `TEXT` written by the application. IDs are `uuid4` hex strings from the stdlib, with prefixes `run_`, `inv_`, `ret_`.
- `PRAGMA foreign_keys=ON` runs on **every** connection through a SQLAlchemy `connect` event. WAL is set once.

## 6. Run flow (ADD §4.1, adapted)

```text
POST /v1/runs {case_id, retrieval_fault: "none"|"stale_version", capture_content}
 └─ pipeline.run_case()                       # apps/demo_rag; no FastAPI imports
     ├─ run_registry.create_run()  → run_id, root span "rag.run" (CHAIN); trace_id saved on run
     ├─ span "retriever.retrieve" (RETRIEVER)
     │    deterministic: fault=none → current version; stale_version → superseded version
     ├─ run_registry.create_invocation()  → model_invocation_id BEFORE the model call
     ├─ span "llm.generate" (LLM), attribute blackbox.model_invocation_id
     │    fixture model extracts the answer from the retrieved text
     ├─ run_registry.record_retrieval(run_id, invocation_id, document_id, rank, score)
     ├─ run_registry.complete_invocation(output_sha256)
     └─ run_registry.complete_run(status, answer_text if opted in)
SQLiteSpanExporter (SimpleSpanProcessor) writes spans synchronously at span end.
```

- The pipeline is a plain function called by both the API route and a CLI (`python -m demo_rag run --case case_0007 --fault stale_version`). The route only validates input and maps the response (rules §2).
- Running in the request handler is acceptable here because the fixture model costs microseconds. Once a real model arrives (Phase 3), this moves to the worker, as ADD §6.4 already requires.
- `SimpleSpanProcessor` (synchronous) is used instead of the batch processor so that the trace is complete when `POST /v1/runs` returns. This is needed for "no intentionally sampled loss" (ADD §11.1).
- On failure, spans record the exception status, the run is marked `failed` with a structured `error_type` (`FIXTURE_HASH_MISMATCH`, `CASE_NOT_FOUND`, `ANSWER_NOT_EXTRACTABLE`), and the API returns that error, not a stack trace.

## 7. Privacy rules applied in Phase 1

- `capture_content=false` (the default): spans carry document IDs, versions and hashes. The prompt and output appear as **SHA-256 only**. The OpenInference `input.value`, `output.value`, `llm.input_messages.*` and `retrieval.documents.*.document.content` attributes are **not set**.
- `capture_content=true`: the answer text is stored on the run and the span content attributes are set. Document bodies are still served from the committed fixture, not copied into spans.
- No raw prompts in logs. Structured logs carry `run_id`, `trace_id` and `request_id`.
- The API binds `127.0.0.1` only (uvicorn `--host 127.0.0.1` is the documented command).

## 8. API contract (Phase 1 subset of ADD §6.1)

All responses are versioned Pydantic models. Errors use one shape: `{"error": {"code", "message", "request_id"}}`.

| Endpoint | Request | Response (key fields) |
|---|---|---|
| `POST /v1/runs` | `case_id` (must exist), `retrieval_fault` enum, `capture_content` bool | `201` with `run_id`, `trace_id`, `status` |
| `GET /v1/runs/{run_id}` | – | run fields, `answer` (`{captured: bool, text?, sha256}`), `invocations[]` (id, model_id, revision, span_id), `retrievals[]` (rank, score, `document{id, doc_key, version, sha256, valid_from, valid_to}`), `spans[]` (tree-orderable), `expected_answer` from the case fixture, `is_correct` |
| `GET /v1/invocations/{id}` | – | identity, revisions, hashes, sampling config, `run_id`, `trace_id`, `span_id`, `neural_access: "unavailable"` |
| `GET /v1/health` | – | `{status: "ok"}`. Makes no GPU or model claims. |

`is_correct` compares against the fixture's ground truth and is labelled **observed** in the UI. Ground truth is stored apart from explanations (ADD §8.4).

Cases and expected answers come from fixtures. Nothing generates them at runtime.

## 9. Dashboard (minimal)

- Vite + React + TypeScript (strict), ESLint. **No component library and no router.** The page reads `?run=<run_id>` from the URL. The demo CLI prints that URL.
- Types are written by hand from the `/v1` response models. Codegen is not worth it for three endpoints. A contract test (§10) keeps the two in sync.
- One page with four sections, in the exit-test order:
  1. **Answer**: text, or "content not captured", with its hash; correct/incorrect shown as icon **and** text (ADD §9.2) with an `observed` badge.
  2. **Retrieved document**: doc key, **version**, validity dates, SHA-256 and rank. A stale version is flagged by comparing `valid_to`.
  3. **Model invocation**: ID, model/revision, prompt and output hashes, and "neural access unavailable".
  4. **Trace**: an indented span list with durations (a simple timeline, no chart library).
- Loading, failed and not-found states are shown honestly. Missing data is labelled "not captured", never blank or zero.
- Vite's dev proxy points to `127.0.0.1:8000`. No CORS is opened to other origins.

## 10. Tests (proportional to risk)

| Tier | Test |
|---|---|
| unit | Retriever returns the current version for `none` and the superseded version for `stale_version`. It is deterministic across calls. |
| unit | Fixture loader rejects a document whose bytes don't match the recorded SHA-256. |
| unit | Fixture loader rejects a path that escapes the fixture root. |
| unit | Fixture model extracts `800 units` and `500 units` from the two versions, and raises `ANSWER_NOT_EXTRACTABLE` otherwise. |
| unit | `llm.generate` and `retriever.retrieve` spans carry no content attributes when `capture_content=false`. |
| integration | Migration `upgrade head` then `downgrade base` on a temp SQLite file. FK pragma is on for a fresh connection, and an orphan insert fails. |
| integration | **Trace completeness:** one run produces exactly the spans `rag.run`, `retriever.retrieve` and `llm.generate`. All share the run's `trace_id`, parent links are valid, `llm.generate.span_id == invocation.span_id`, and the span carries `blackbox.model_invocation_id == invocation.id`. |
| integration | **Stale-document failure:** `case_0007` + `stale_version` gives `is_correct=false` and answer `500 units`, and the retrieval links to `doc_key=plan_delta`, superseded version and its SHA. With `none`, the answer is `800 units` and `is_correct=true`. |
| integration | Privacy default: with `capture_content=false`, `answer_text` is NULL and no stored span attribute contains the answer or document text. |
| contract | API errors: unknown `case_id` returns `422`/`404` in the structured shape, and an unknown run returns `404`. Nothing returns a stack trace. |
| contract | A snapshot of the `/v1` OpenAPI schema for the three endpoints is checked in and compared, so changes are deliberate. The dashboard types are reviewed against it. |
| e2e | Script: start the API on a temp DB, `POST` a stale run, `GET` the run, follow `retrievals[0].document` and `invocations[0].id` to `GET /v1/invocations/{id}`, and assert the IDs agree. This is the exit test, automated at API level. |
| frontend | `tsc --noEmit` and ESLint. One render test of the run page against a fixture response, if Vitest is accepted; otherwise this is verified manually and documented. |

Commands: `uv run ruff format --check`, `uv run ruff check`, `uv run mypy`, `uv run pytest`, `npm run lint && npm run typecheck` in `apps/dashboard`.

## 11. Build order (one coherent commit each)

| Step | Commit | Contents | Acceptance check |
|---|---|---|---|
| 0a | `chore(repo): bootstrap` | `git init`, `.gitignore` (`.local/`, `.env`, `node_modules`, `__pycache__`), `uv init`, Python pin, `pyproject.toml` with ruff/mypy/pytest config, README skeleton | `ruff`, `mypy` and `pytest` run clean on an empty suite |
| 0b | `docs(adr): foundation` | Move the ADD to `docs/architecture/`, extract ADR-001..008 as short files in `docs/adr/`, add ADR-009 for D1 (fixture model in Phase 1), link from the README | Links resolve |
| 0c | `docs(protocol): stale_document_v1` | `docs/research-protocols/stale_document_v1.md`: cases, versions, the failure being injected, and what Phase 1 does and does not measure | Reviewed by the maintainer |
| 1 | `feat(fixtures): stale-document cases` | 10 synthetic cases plus current/superseded document versions, a `manifest.json` with SHA-256 per file, and the loader with hash and path checks | Unit tests; hashes stable |
| 2 | `feat(store): sqlite core + migration 0001` | Engine factory with pragmas, Alembic env, the five tables | Up/down migration test; FK test |
| 3 | `feat(registry): runs, invocations, retrievals` | `run_registry` functions using SQL Core, plus document seeding from the fixture manifest (idempotent upsert on `(doc_key, version)`) | Unit/integration tests |
| 4 | `feat(trace): otel spans to sqlite` | `SQLiteSpanExporter`, OpenInference span-kind attributes, tracer setup | Exporter integration test |
| 5 | `feat(demo): deterministic rag pipeline + cli` | Retriever, fixture model, `run_case`, CLI | Stale-failure and trace-completeness tests |
| 6 | `feat(api): /v1 runs, invocations, health` | FastAPI app, request IDs, error handler, response models, OpenAPI snapshot | Contract and e2e tests |
| 7 | `feat(dashboard): run detail page` | Vite app, single page, dev proxy | Lint and typecheck; manual exit-test walkthrough recorded in the runbook |
| 8 | `docs(runbook): phase 1 demo` | `docs/runbooks/phase1_demo.md`: setup, run, the exit-test walkthrough, backup (copy `.local/`), limitations | A fresh-clone run-through passes the §20.2 items that apply to Phase 1 |

Tag `v0.1-traces` after step 8 passes, as ADD §12.5 requires.

## 12. Dependencies (new, each with a reason)

| Package | Why |
|---|---|
| `fastapi`, `uvicorn` | The API stack approved in the ADD |
| `pydantic` (via FastAPI) | Ingress validation (rules §3) |
| `sqlalchemy`, `alembic` | Migrations are mandatory (rules §5). Core gives parameterized SQL. |
| `opentelemetry-api`, `opentelemetry-sdk` | Span creation and the exporter interface (ADR-003) |
| `openinference-semantic-conventions` | Canonical OpenInference attribute names. Avoids hand-typed strings drifting from the spec. |
| dev: `ruff`, `mypy`, `pytest`, `httpx` | Lint/format, types, tests, and the FastAPI `TestClient` transport |
| JS: `react`, `react-dom`, `vite`, `typescript`, `@vitejs/plugin-react`, `eslint` (+ TS plugin) | The approved dashboard stack. Vitest only if the render test is accepted. |

Not added: Phoenix, an OTLP exporter or collector, Docker, Redis, a router, a UI kit, PyTorch or transformers. Torch and transformers arrive in Phase 3 behind an optional dependency group.

All versions are pinned in `uv.lock` and `package-lock.json`.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The fixture model could be mistaken for real model behaviour | `model_id="fixture-extractive"`, "neural access unavailable" in the UI and API, and ADR-009 and the README say Phase 1 demonstrates the **trace**, not model cognition |
| OpenInference attribute names drift between versions | Pin the conventions package. The trace-completeness test asserts the span-kind attribute. |
| Span export races the API response | Synchronous `SimpleSpanProcessor`, plus the completeness test right after `POST` |
| The run viewer leaks content when capture is off | Content attributes are set in one place (the pipeline, gated on `capture_content`); the privacy integration test asserts on what SQLite actually stored |
| SQLite placed on a synced folder (e.g. OneDrive under `Documents`) | The runbook warns about it (ADD §5.3). `DATABASE_URL` stays configurable in `.env.example`. |

## 14. Definition of done for Phase 1

- [ ] Every row of §10 passes locally; any check that cannot run is reported with its reason.
- [ ] The exit test passes both as the automated e2e test and as the manual dashboard walkthrough in the runbook.
- [ ] A fresh clone runs `uv sync`, `alembic upgrade head`, the API, the demo CLI and the dashboard with no cloud credentials and no internet after install.
- [ ] By default no prompt, answer or document body is stored in SQLite.
- [ ] The README and runbook are updated, and ADR-009 is recorded.
- [ ] The rules §9 self-review checklist has been completed against the final diff.
