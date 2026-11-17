"""Single-shot capture worker: claim one queued job, replay the stored tokens, store measurements.

Heavy model work lives here, never in an HTTP handler (ADR-002). Results are written only after
the read-only check passes, so a failed job leaves no partial measurements.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_store.artifacts import ArtifactError, get_verified, put_bytes
from instrumentation.plan import CaptureAdapter, CaptureData, CaptureError, CapturePlan
from run_registry import captures
from sqlalchemy import Engine

DEFAULT_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class WorkResult:
    job_id: str | None  # None: nothing was queued
    status: str
    error_type: str | None = None


def _replay_record(
    data: CaptureData, output_ids: list[int], n_prompt: int, elapsed: float
) -> dict[str, Any]:
    """What the replay reproduces and how well; never claims exact replay (ADD §7.4)."""
    greedy = [data.argmax_ids[n_prompt - 1 + i] for i in range(len(output_ids))]
    return {
        **data.runtime,
        "max_logit_diff": data.max_logit_diff,
        "greedy_consistent": greedy == output_ids,
        "status": "approximate",
        "seconds": round(elapsed, 3),
    }


def run_once(
    engine: Engine,
    artifact_root: Path,
    adapter_factory: Callable[[], CaptureAdapter],
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> WorkResult:
    with engine.begin() as conn:
        captures.recover_stale(conn)
        job = captures.claim_next(conn)
    if job is None:
        return WorkResult(None, "idle")

    def fail(code: str, *, blocked: bool = False) -> WorkResult:
        with engine.begin() as conn:
            captures.fail_job(conn, job.id, code, blocked_resource=blocked)
        return WorkResult(job.id, "blocked_resource" if blocked else "failed", code)

    try:
        with engine.connect() as conn:
            sequences = captures.token_sequences(conn, job.invocation_id)
            prompt = json.loads(get_verified(conn, artifact_root, sequences["prompt"].artifact_id))
            output = json.loads(get_verified(conn, artifact_root, sequences["output"].artifact_id))
        prompt_ids, output_ids = prompt["token_ids"], output["token_ids"]
        try:
            adapter = adapter_factory()
        except (OSError, ImportError):  # missing download, no network, or no neural group
            return fail("MODEL_ASSET_UNAVAILABLE")
        started = time.perf_counter()
        data = adapter.capture(prompt_ids + output_ids, CapturePlan.from_json(job.plan_json))
        elapsed = time.perf_counter() - started
        if elapsed > timeout_seconds:
            # shortcut: checked after the fact; a hung capture is not interrupted. Upgrade to a
            # subprocess with a hard kill if jobs ever run long.
            return fail("JOB_TIMEOUT")
        with engine.begin() as conn:
            for m in data.measurements:
                artifact_id = put_bytes(
                    conn,
                    artifact_root,
                    m.payload,
                    mime_type="application/x-safetensors",
                    category="raw",
                    parents=[sequences["prompt"].artifact_id],
                    dtype=m.dtype,
                    shape_json=json.dumps(m.shape),
                )
                captures.record_measurement(
                    conn,
                    job_id=job.id,
                    invocation_id=job.invocation_id,
                    kind=m.kind,
                    hook_name=m.hook_name,
                    layer=m.layer,
                    token_position=m.token_position,
                    artifact_id=artifact_id,
                    dtype=m.dtype,
                    shape=m.shape,
                    summary=m.summary,
                )
            captures.complete_job(
                conn, job.id, _replay_record(data, output_ids, len(prompt_ids), elapsed)
            )
    except CaptureError as err:
        return fail(err.code)
    except ArtifactError as err:
        return fail(err.code)
    except MemoryError:
        return fail("OUT_OF_MEMORY", blocked=True)
    except Exception:
        fail("INTERNAL_ERROR")
        raise
    return WorkResult(job.id, "completed")
