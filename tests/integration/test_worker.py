"""Capture job lifecycle and worker, driven by stub adapters (no torch needed)."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from demo_rag.backends import Backend
from demo_rag.pipeline import run_case
from evidence_store.artifacts import get_verified
from experiment_worker.worker import run_once
from instrumentation.plan import (
    CaptureAdapter,
    CaptureData,
    CapturedMeasurement,
    CaptureError,
    CapturePlan,
    build_plan,
)
from run_registry import captures
from sqlalchemy import Engine, text
from trace_ingest.exporter import make_tracer_provider

HOOK = "blocks.3.hook_resid_post"


class GoodAdapter:
    """Pretends to replay the 5-token sequence [1,2,3 | 4,5]; greedy choices match the output."""

    def capture(self, token_ids: list[int], plan: CapturePlan) -> CaptureData:
        assert token_ids == [1, 2, 3, 4, 5]
        return CaptureData(
            measurements=[
                CapturedMeasurement(
                    "activation",
                    hook,
                    3,
                    position,
                    "float32",
                    [512],
                    b"tensor-bytes",
                    {"l2_norm": 1.0, "mean": 0.0, "max_abs": 0.5},
                )  # fmt: skip
                for hook in plan.hooks
                for position in plan.positions
            ],
            argmax_ids=[0, 0, 4, 5, 0],
            max_logit_diff=0.0,
            runtime={"mode": "teacher_forced", "device": "cpu"},
        )


class FailingAdapter:
    def capture(self, token_ids: list[int], plan: CapturePlan) -> CaptureData:
        raise CaptureError("NUMERICAL_MISMATCH", "hooks changed logits")


class MemoryHogAdapter:
    def capture(self, token_ids: list[int], plan: CapturePlan) -> CaptureData:
        raise MemoryError


@pytest.fixture
def queued(
    engine: Engine, artifact_root: Path, stub_backend: Backend
) -> tuple[str, captures.JobRecord]:
    tracer = make_tracer_provider(engine).get_tracer("test")
    run_case(
        engine, tracer, artifact_root, "case_0007", "stale_version",
        capture_content=True, backend=stub_backend,
    )  # fmt: skip
    plan = build_plan([HOOK], [-1, 0], include_logits=False, n_tokens=5)
    with engine.begin() as conn:
        invocation_id = conn.execute(text("SELECT id FROM model_invocations")).scalar_one()
        job, _ = captures.create_job(
            conn,
            invocation_id=invocation_id,
            plan_json=plan.canonical_json(),
            plan_sha256=plan.sha256,
        )
    return invocation_id, job


def _factory(adapter: CaptureAdapter) -> Callable[[], CaptureAdapter]:
    return lambda: adapter


def test_completed_job_stores_verified_measurements_and_an_approximate_replay_record(
    engine: Engine, artifact_root: Path, queued: tuple[str, captures.JobRecord]
) -> None:
    invocation_id, job = queued
    result = run_once(engine, artifact_root, _factory(GoodAdapter()))
    assert (result.job_id, result.status) == (job.id, "completed")
    with engine.connect() as conn:
        finished = captures.get_job(conn, job.id)
        measurements = captures.list_measurements(conn, invocation_id)
        assert [(m.token_position, m.layer, m.shape) for m in measurements] == [
            (0, 3, [512]),
            (4, 3, [512]),
        ]
        assert get_verified(conn, artifact_root, measurements[0].artifact_id) == b"tensor-bytes"
    assert finished is not None and finished.replay is not None
    assert finished.replay["status"] == "approximate"
    assert finished.replay["greedy_consistent"] is True
    assert run_once(engine, artifact_root, _factory(GoodAdapter())).status == "idle"


@pytest.mark.parametrize(
    ("adapter", "status", "error"),
    [
        (FailingAdapter(), "failed", "NUMERICAL_MISMATCH"),
        (MemoryHogAdapter(), "blocked_resource", "OUT_OF_MEMORY"),
    ],
)
def test_failed_job_records_a_structured_error_and_no_measurements(
    engine: Engine,
    artifact_root: Path,
    queued: tuple[str, captures.JobRecord],
    adapter: CaptureAdapter,
    status: str,
    error: str,
) -> None:
    invocation_id, job = queued
    result = run_once(engine, artifact_root, _factory(adapter))
    assert (result.status, result.error_type) == (status, error)
    with engine.connect() as conn:
        finished = captures.get_job(conn, job.id)
        assert finished is not None and (finished.status, finished.error_type) == (status, error)
        assert conn.execute(text("SELECT count(*) FROM measurements")).scalar_one() == 0


def test_missing_model_assets_fail_the_job_without_a_stack_trace(
    engine: Engine, artifact_root: Path, queued: tuple[str, captures.JobRecord]
) -> None:
    def unavailable() -> CaptureAdapter:
        raise OSError("no network")

    result = run_once(engine, artifact_root, unavailable)
    assert (result.status, result.error_type) == ("failed", "MODEL_ASSET_UNAVAILABLE")


def test_duplicate_requests_share_one_active_job_and_a_finished_one_allows_a_retry(
    engine: Engine, artifact_root: Path, queued: tuple[str, captures.JobRecord]
) -> None:
    invocation_id, job = queued
    with engine.begin() as conn:
        again, created = captures.create_job(
            conn, invocation_id=invocation_id, plan_json=job.plan_json, plan_sha256=job.plan_sha256
        )
    assert (again.id, created) == (job.id, False)
    run_once(engine, artifact_root, _factory(FailingAdapter()))
    with engine.begin() as conn:
        retry, created = captures.create_job(
            conn, invocation_id=invocation_id, plan_json=job.plan_json, plan_sha256=job.plan_sha256
        )
    assert created and retry.id != job.id  # a new job; the failed attempt is kept


def test_a_job_left_running_by_a_dead_worker_is_marked_interrupted(
    engine: Engine, artifact_root: Path, queued: tuple[str, captures.JobRecord]
) -> None:
    _, job = queued
    with engine.begin() as conn:
        assert captures.claim_next(conn) is not None
        assert captures.claim_next(conn) is None  # nothing left to claim twice
    result = run_once(engine, artifact_root, _factory(GoodAdapter()))
    assert result.status == "idle"
    with engine.connect() as conn:
        stale = captures.get_job(conn, job.id)
    assert stale is not None and (stale.status, stale.error_type) == (
        "failed",
        "WORKER_INTERRUPTED",
    )


def test_replay_record_is_json_serialisable(
    engine: Engine, artifact_root: Path, queued: tuple[str, captures.JobRecord]
) -> None:
    run_once(engine, artifact_root, _factory(GoodAdapter()))
    with engine.connect() as conn:
        raw = conn.execute(text("SELECT replay_json FROM capture_jobs")).scalar_one()
    assert json.loads(raw)["device"] == "cpu"
