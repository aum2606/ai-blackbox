# Phase 3 Implementation Plan: Diagnostic Neural Capture

> **Status:** Draft for maintainer review · **Date:** 2026-11-17
> **Governs:** `rules_v1.1.md` (mandatory), ADD v1.1 §4.2, §7, §13 Phase 3; ADR-006, ADR-009, ADR-010.
> **Goal (ADD §13):** Reproduce a supported model call and inspect selected internal signals.
> **Exit test (ADD §13):** One invocation links to the correct layer and position measurements, verified by checksum and a golden numerical test.

## 1. Decisions (confirmed this session)

| # | Decision | Choice |
|---|---|---|
| D1 | First real model | **`EleutherAI/pythia-70m-deduped`**, pinned to an exact commit SHA, `model.safetensors` only (166 MB), `trust_remote_code=False`, float32, CPU, eval mode |
| D2 | Capture mechanism | **PyTorch forward hooks on HF `transformers`**. Hook names follow TransformerLens style (`blocks.{i}.hook_resid_post`) and map to `gpt_neox.layers[i]` outputs |
| D3 | How the model enters runs | **Optional per run.** The CLI gains `--model fixture\|pythia-70m-deduped`. The fixture stays the default, and the API's `POST /v1/runs` stays fixture-only, because model execution never runs in an HTTP handler (rules §2) |

Recorded as **ADR-011**. It also covers the deferred items from ADR-010 that land now: token IDs and measurements.

### Facts this plan relies on

- **Machine:** 15.2 GB RAM, with only about **0.4 GB free** at the time of checking. There is also an RTX 4060 8 GB, which is not used: Phase 3 is CPU-first.
- **Gate dependency:** pythia-70m needs roughly 1 GB of RAM including torch. If memory is short, the compatibility gate (§3, step 1) records it rather than failing silently.
- **Model:** pythia-70m will **not** answer the cap question reliably. Its answers are recorded as observed, never filtered (ADD §8.2). Phase 3 proves capture, not semantic interpretability (ADD §7.1).
- **Downloads need your approval:** the torch CPU wheel (about 200 MB) and the model files (about 170 MB). Nothing is committed to git.

## 2. Scope

**In scope**

1. Phase 0 compatibility gate for the pinned model, with a written report.
2. Token IDs for prompt and output, stored as artifacts **only with content-capture opt-in**. Token IDs are the prompt in another form, so the privacy rule is the same.
3. A capture plan, the `capture_jobs` state machine and a bounded **local worker CLI** with no queue service.
4. `measurements` rows. Each records the tensor's checksum, shape, dtype, model revision and hook, and the tensor itself lives in the Phase 2 artifact store as safetensors.
5. A read-only instrumentation check against uninstrumented logits, plus a golden-logit test.
6. API endpoints:
   - `POST /v1/invocations/{id}/captures` creates a capture job.
   - `GET /v1/captures/{id}` returns its status.
   - `GET /v1/invocations/{id}/measurements` returns measurement summaries.
7. Dashboard: a basic token/layer view with provenance and replay status.

**Out of scope**

- SAE work (Phase 4) and patching or experiments (Phase 5). The gate checks once that a patch mechanically changes logits, but ships no patch API.
- GPU support, and any export to free notebooks.
- `/v1/capabilities`. A static capability registry inside the code is enough to validate requests.
- Cancelling capture jobs. Jobs are short and the worker is single-shot.

## 3. Build order (one commit each, no co-author trailer)

| # | Commit | Contents | Acceptance check |
|---|---|---|---|
| 1 | `docs(adr): ADR-011 pythia-70m and HF hooks` | ADR with context, alternatives, consequences, validation and rollback | Reviewed |
| 2 | `chore(deps): optional neural dependency group` | `[dependency-groups] neural = [torch, transformers, safetensors]` with a uv CPU index for torch. Core install and the default `pytest` run stay torch-free. Pytest marker `neural`, excluded by default | `uv run pytest` is unchanged; `uv sync --group neural` installs |
| 3 | `feat(instrumentation): compatibility gate` | `scripts/compat_pythia.py` (see the checklist below), writing `docs/compatibility/pythia-70m-deduped.md`. If it fails, Phase 3 stops and the report says `failed` or `blocked_resource` | The report exists and the gate passes |
| 4 | `feat(store): migration 0003` | New tables `token_sequences`, `capture_jobs` and `measurements` (§4) | Up/down and FK tests |
| 5 | `feat(instrumentation): pythia adapter` | `packages/instrumentation/pythia.py`: load the pinned revision, tokenize, greedy generate, forward logits, capture at positions. `capabilities.py` holds the static hook list and model dimensions (6 layers, `d_model` 512) | Neural tests: shapes, read-only logits within tolerance, golden logits |
| 6 | `feat(demo): real-model runs from the CLI` | `--model pythia-70m-deduped`: render the prompt, tokenize, generate 4 tokens greedily, decode. The invocation gets `neural_access=available` and the real revision. Token sequences are stored only on opt-in | Neural integration test; default tests unaffected |
| 7 | `feat(worker): capture jobs` | `workers/experiment_worker` (`python -m experiment_worker`): recover stale jobs, claim one queued job atomically, run the capture, store artifacts and measurements, then complete. Timeout and errors are structured | Unit tests with a stub adapter (state machine, claim, error codes); neural end-to-end test |
| 8 | `feat(api): captures and measurements` | Capture endpoint with validation; capture status; measurements list (summaries only) | Contract tests; OpenAPI snapshot refreshed |
| 9 | `feat(dashboard): neural explorer MVP` | Layer × position grid of L2 norms. Provenance and replay status. "not captured" or "neural access unavailable" when absent | Lint, typecheck, build; browser check |
| 10 | `docs(runbook): phase 3` | Setup of the neural group, the download note, how to run the worker, limitations. Tag `v0.3-neural`; push only when you say so | Fresh-clone run-through |

### Compatibility gate checklist (ADD §7.2)

- Load the model by exact commit SHA. Record the torch and transformers versions.
- List hook points with their shapes.
- Run a fixed synthetic prompt with no hooks and save the target logits.
- Run the same prompt with read-only hooks: the maximum absolute difference must be at most 1e-6.
- Run 10 repeats and check they are bit-identical on CPU.
- Patch one residual position and check the logits change, with no shape or device error.
- Record peak RSS and cold-start time.
- Use `use_cache=False` for replay.
- Mark the SAE check `not_run (Phase 4)`.

## 4. Data model (migration `0003_neural_capture`)

```text
token_sequences (id PK, invocation_id FK, role CHECK(prompt|output), tokenizer_revision,
                 n_tokens, artifact_id FK artifacts, created_at)          -- JSON artifact `{"token_ids": [...]}`
capture_jobs    (id PK, invocation_id FK, plan_json, plan_sha256, status CHECK(queued|running|
                 completed|failed|blocked_resource), error_type NULL, replay_json NULL,
                 created_at, started_at NULL, ended_at NULL)
measurements    (id PK, capture_job_id FK, invocation_id FK, kind CHECK(activation|logits),
                 hook_name, layer NULL, token_position, artifact_id FK artifacts,
                 dtype, shape_json, summary_json, created_at)
```

- **Immutability:** a completed job is immutable. A re-capture is a new job.
- **Duplicate requests:** submitting the same invocation with the same `plan_sha256` while a job is queued or running returns the existing job.
- **`replay_json` contents:**
  - mode `teacher_forced`
  - device, dtype, torch version
  - the maximum difference between hooked and unhooked logits
  - whether the teacher-forced argmax reproduces the generated output tokens (`greedy_consistent`)
  - status: `approximate` within tolerance, or `not_verified`
- **Summaries:** `summary_json` holds L2 norm, mean and max-abs. These are measurements (`measured`), not interpretations.

## 5. Capture request and validation

```json
{ "hooks": ["blocks.3.hook_resid_post"], "token_positions": [-1], "include_logits": true }
```

Checks run in this order. Each failure has a structured error code.

| Check | Error code |
|---|---|
| Invocation exists | `INVOCATION_NOT_FOUND` |
| `neural_access == available` | `NEURAL_ACCESS_UNAVAILABLE` |
| Prompt token IDs were stored (the run opted in) | `PRIVACY_RESTRICTED` |
| Every hook is in the capability registry | `HOOK_UNSUPPORTED` |
| Positions are inside the token sequence | `POSITION_OUT_OF_RANGE` |
| Estimated bytes are within `MAX_CAPTURE_BYTES` (64 MiB) | `CAPTURE_TOO_LARGE` |

The byte estimate is `positions × d_model × 4 × hooks`, plus the logits row (ADD §7.3). Worker errors are `MODEL_ASSET_UNAVAILABLE`, `NUMERICAL_MISMATCH`, `OUT_OF_MEMORY`, `JOB_TIMEOUT` (`WORKER_JOB_TIMEOUT_SECONDS`, default 600) and `WORKER_INTERRUPTED`, which marks stale `running` jobs at worker start.

## 6. Worker flow

1. Recover stale running jobs.
2. Claim the job with `UPDATE … SET status='running' WHERE id=:id AND status='queued'` and check `rowcount`.
3. Load the model and verify the golden prompt's baseline.
4. Load the token IDs through `get_verified`.
5. Run one unhooked forward pass and one hooked forward pass, teacher-forced, `use_cache=False`. Compare their logits. A mismatch fails the job with `NUMERICAL_MISMATCH` before anything is stored.
6. Save each hook × position slice as a safetensors artifact, category `raw`, through `put_bytes`. This writes to a temp file, verifies and renames, and enforces the quota.
7. Insert measurement rows, then mark the job `completed`, in one short transaction.

## 7. Tests

- **Default suite (no torch):**
  - plan validation and byte estimate
  - privacy gate
  - capture job state machine and atomic claim
  - duplicate submission
  - stale-job recovery
  - worker against a stub adapter
  - API contract and OpenAPI snapshot
  - migrations
- **`-m neural` suite** (needs the neural group and a cached model; skips with a reason otherwise):
  - golden logits for a fixed prompt (rtol 1e-4, committed JSON)
  - read-only hooks preserve logits
  - hook shapes
  - CLI pythia run followed by capture, end to end
- **Exit test** (`tests/e2e/test_phase3_exit.py`, neural):
  1. Run with pythia and capture on.
  2. Submit a capture for `blocks.3.hook_resid_post`.
  3. Run the worker once.
  4. `GET /v1/invocations/{id}/measurements` returns layer 3 at the last position, and `GET /v1/artifacts/{artifact_id}` reports `verified=true`.
  5. The replayed logits match the golden test within tolerance.

## 8. Dependencies

| Package | Group | Why |
|---|---|---|
| `torch` (CPU wheel, pinned) | `neural` | Model execution and hooks |
| `transformers` (pinned) | `neural` | Loading the pinned checkpoint and tokenizer |
| `safetensors` | `neural` | Tensor files that can't run code when loaded (rules §5) |

Not added: TransformerLens, numpy as a direct dependency, CUDA builds, and any notebook tooling.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Low free RAM (0.4 GB seen) | The gate records peak RSS. On failure the job is marked `blocked_resource` or `OUT_OF_MEMORY`, never silently skipped |
| Model download or Hugging Face unavailable | `MODEL_ASSET_UNAVAILABLE`; core features unaffected |
| CPU numerics differ across machines | Tolerances are documented, and the replay is labelled `approximate`, never "exact" (ADD §7.4) |
| Someone reads pythia's output as reasoning | The UI and report state that activations are measurements, not explanations |
