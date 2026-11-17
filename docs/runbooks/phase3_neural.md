# Runbook: Phase 3 (diagnostic neural capture)

Everything in Phases 1 and 2 still works without any of this. Phase 3 is optional and needs the `neural` dependency group and a one-time model download. Decisions are in [ADR-011](../adr/adr-011.md).

## Setup

```bash
uv sync --group neural
uv run alembic upgrade head
```

A plain `uv sync` removes the neural packages again, so use `--group neural` every time you sync while working on Phase 3. Installing adds torch (CPU build, about 200 MB) and transformers.

The first real run downloads `EleutherAI/pythia-70m-deduped` at the pinned commit (about 166 MB, safetensors only) into your Hugging Face cache (`~/.cache/huggingface`). Nothing is committed to git. On Windows without Developer Mode, Hugging Face warns that it cannot use symlinks; this is harmless and only uses a little more disk.

## Compatibility gate

```bash
uv run python scripts/compat_pythia.py
```

Regenerates [the report](../compatibility/pythia-70m-deduped.md) (exact revision, hook points and shapes, read-only hooks within 1e-6, a patch that changes logits, 10 bit-identical repeats, peak memory) and the golden-logit file used by the neural tests. The report states the checks were run on one CPU and software stack; it does not promise identical bits elsewhere.

## Capture an invocation

1. Create a run with the real model. Content capture is required, because token IDs are stored only on opt-in:

   ```bash
   uv run python -m demo_rag --case case_0007 --fault stale_version --capture-content --model pythia-70m-deduped
   ```

2. With the API running (`uvicorn blackbox_api.app:app_from_env --factory --host 127.0.0.1 --port 8000`), queue a capture. Negative positions count from the end of prompt plus output:

   ```bash
   curl -X POST localhost:8000/v1/invocations/<invocation_id>/captures -H "content-type: application/json" -d '{"hooks": ["blocks.3.hook_resid_post"], "token_positions": [0, -1]}'
   ```

   Supported hooks are `blocks.0` to `blocks.5` `.hook_resid_post`. Add `"include_logits": true` to also capture next-token logits at those positions.

3. Run the worker once. It claims one queued job, replays the stored tokens, checks that hooks changed nothing, and stores the tensors as artifacts:

   ```bash
   uv run python -m experiment_worker
   ```

4. Open the run page. The **Neural explorer** table shows activation sizes per hook and position. `GET /v1/invocations/{id}/measurements` returns the same summaries, and `GET /v1/artifacts/{artifact_id}` verifies each tensor file.

| Error code | Meaning |
|---|---|
| `NEURAL_ACCESS_UNAVAILABLE` (409) | The invocation used the fixture model |
| `PRIVACY_RESTRICTED` (403) | The run did not store token IDs (no content capture) |
| `HOOK_UNSUPPORTED`, `POSITION_OUT_OF_RANGE`, `CAPTURE_TOO_LARGE` (422) | Request outside the capability registry or size cap |
| `MODEL_ASSET_UNAVAILABLE`, `NUMERICAL_MISMATCH`, `OUT_OF_MEMORY`, `JOB_TIMEOUT`, `WORKER_INTERRUPTED` | Recorded on a failed job (`GET /v1/captures/{id}`); `OUT_OF_MEMORY` is labelled `blocked_resource` |

A failed job is never retried automatically. Post the same request again to create a new job; the failed one stays in the history.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MAX_CAPTURE_BYTES` | `67108864` (64 MiB) | Largest tensor payload one capture request may ask for |
| `WORKER_JOB_TIMEOUT_SECONDS` | `600` | A capture that took longer is marked `JOB_TIMEOUT` and stores nothing |

Like the other settings these come from the process environment, not `.env`.

## Checks

```bash
uv run pytest                 # default suite: no torch needed, no model download
uv run pytest -m neural       # needs the neural group and the cached model (about a minute)
```

## Limitations

- pythia-70m does not answer the cap question reliably, and its wrong answers are recorded as observed. Phase 3 proves capture, not interpretation: activation sizes are measurements, not explanations of reasoning. Nothing here supports a causal claim (that needs Phase 5 controls).
- Replay is teacher-forced on the stored tokens with `use_cache=False`. It is labelled `approximate`, never exact, and the record includes the maximum logit difference against an unhooked pass and whether greedy decoding reproduces the stored output.
- The checkpoint ships in fp16; the adapter loads it as float32 on CPU. Hook names map to the residual stream after each of the 6 layers; other hook types are not supported yet.
- Token IDs are stored as small JSON artifacts, not safetensors (they are integers, and JSON cannot execute code). Tensors are safetensors.
- The timeout is checked after the capture finishes; a hung capture is not interrupted. One worker, one job per invocation.
- Token IDs, like prompts, exist only for runs that opted in. A run without content capture cannot be captured later.
- The API cannot start real-model runs, by design: model work never runs in an HTTP request.
- Retention and deletion of artifacts is still not enforced (Phase 7).
