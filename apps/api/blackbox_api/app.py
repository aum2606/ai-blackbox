"""FastAPI app for the /v1 endpoints. Bind to 127.0.0.1 only (ADD §10.2)."""

import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from demo_rag.fixtures import FixtureError, load_case
from demo_rag.pipeline import run_case
from evidence_store.artifacts import ArtifactError, get_verified
from evidence_store.engine import artifact_root, database_url, make_engine, max_capture_bytes
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from instrumentation.plan import CaptureError, CapturePlan, build_plan
from run_registry import captures
from run_registry import queries as registry_queries
from sqlalchemy import Engine
from trace_ingest.exporter import make_tracer_provider
from trace_ingest.queries import list_spans

from blackbox_api.schemas import (
    AnswerOut,
    ArtifactOut,
    CaptureJobOut,
    CapturePlanOut,
    CaptureRequest,
    ErrorOut,
    HealthOut,
    InvocationOut,
    MeasurementOut,
    RetrievalOut,
    RunCreate,
    RunCreated,
    RunOut,
    SpanOut,
)

_log = logging.getLogger(__name__)
_CAPTURE_STATUS = {
    "NEURAL_ACCESS_UNAVAILABLE": 409,
    "PRIVACY_RESTRICTED": 403,
    "HOOK_UNSUPPORTED": 422,
    "POSITION_OUT_OF_RANGE": 422,
    "CAPTURE_TOO_LARGE": 422,
}
_ERRORS: dict[int | str, dict[str, Any]] = {404: {"model": ErrorOut}}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    body = ErrorOut.model_validate(
        {"error": {"code": code, "message": message, "request_id": request_id}}
    )
    return JSONResponse(body.model_dump(), status_code=status, headers={"X-Request-ID": request_id})


def create_app(engine: Engine, store_root: Path) -> FastAPI:
    app = FastAPI(title="AI Black Box Lab API", version="0.1.0")
    tracer = make_tracer_provider(engine).get_tracer("blackbox_api")

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Always generated here: a client-supplied ID is not trusted.
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, err: ApiError) -> JSONResponse:
        return _error(request, err.status, err.code, str(err))

    @app.exception_handler(FixtureError)
    async def _fixture_error(request: Request, err: FixtureError) -> JSONResponse:
        status = 404 if err.code == "CASE_NOT_FOUND" else 500
        return _error(request, status, err.code, str(err))

    @app.exception_handler(CaptureError)
    async def _capture_error(request: Request, err: CaptureError) -> JSONResponse:
        return _error(request, _CAPTURE_STATUS.get(err.code, 422), err.code, str(err))

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, err: RequestValidationError) -> JSONResponse:
        # Field paths and messages only; submitted values are not echoed back.
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors())
        return _error(request, 422, "VALIDATION_ERROR", detail)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, err: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        _log.error("unhandled error request_id=%s", request_id, exc_info=err)
        return _error(request, 500, "INTERNAL_ERROR", "unexpected server error")

    @app.get("/v1/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(status="ok")  # liveness only; makes no model or GPU claim

    @app.post("/v1/runs", status_code=201, response_model=RunCreated, responses=_ERRORS)
    def create_run(body: RunCreate) -> RunCreated:
        result = run_case(
            engine,
            tracer,
            store_root,
            body.case_id,
            body.retrieval_fault,
            capture_content=body.capture_content,
        )
        return RunCreated(
            run_id=result.run_id,
            trace_id=result.trace_id,
            status=result.status,
            error_type=result.error_type,
        )

    @app.get("/v1/runs/{run_id}", response_model=RunOut, responses=_ERRORS)
    def get_run(run_id: str) -> RunOut:
        with engine.connect() as conn:
            run = registry_queries.get_run(conn, run_id)
            if run is None:
                raise ApiError(404, "RUN_NOT_FOUND", f"unknown run {run_id!r}")
            spans = list_spans(conn, run_id)
        expected = load_case(run.dataset_case_id).expected_answer
        output_sha256 = run.invocations[0].output_sha256 if run.invocations else None
        expected_sha256 = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        return RunOut(
            run_id=run.id,
            task_id=run.task_id,
            case_id=run.dataset_case_id,
            retrieval_fault=run.retrieval_fault,
            trace_id=run.trace_id,
            status=run.status,
            error_type=run.error_type,
            started_at=run.started_at,
            ended_at=run.ended_at,
            answer=AnswerOut(
                captured=run.capture_content, text=run.answer_text, sha256=output_sha256
            ),
            expected_answer=expected,
            # Compared by hash so correctness is observable even without content capture.
            is_correct=None if output_sha256 is None else output_sha256 == expected_sha256,
            invocations=[InvocationOut.model_validate(i) for i in run.invocations],
            retrievals=[RetrievalOut.model_validate(r) for r in run.retrievals],
            spans=[SpanOut.model_validate(s) for s in spans],
        )

    @app.get("/v1/invocations/{invocation_id}", response_model=InvocationOut, responses=_ERRORS)
    def get_invocation(invocation_id: str) -> InvocationOut:
        with engine.connect() as conn:
            invocation = registry_queries.get_invocation(conn, invocation_id)
        if invocation is None:
            raise ApiError(404, "INVOCATION_NOT_FOUND", f"unknown invocation {invocation_id!r}")
        return InvocationOut.model_validate(invocation)

    @app.get("/v1/artifacts/{artifact_id}", response_model=ArtifactOut, responses=_ERRORS)
    def get_artifact(artifact_id: str) -> ArtifactOut:
        """Metadata and a fresh checksum check. Raw bytes are never returned."""
        with engine.connect() as conn:
            artifact = registry_queries.get_artifact(conn, artifact_id)
            if artifact is None:
                raise ApiError(404, "ARTIFACT_NOT_FOUND", f"unknown artifact {artifact_id!r}")
            try:
                get_verified(conn, store_root, artifact_id)
                integrity_error = None
            except ArtifactError as err:
                integrity_error = err.code  # reported, not hidden
        return ArtifactOut(
            id=artifact.id,
            sha256=artifact.sha256,
            mime_type=artifact.mime_type,
            nbytes=artifact.nbytes,
            category=artifact.category,
            parents=list(artifact.parents),
            verified=integrity_error is None,
            integrity_error=integrity_error,
        )

    def _job_out(job: captures.JobRecord) -> CaptureJobOut:
        plan = CapturePlan.from_json(job.plan_json)
        return CaptureJobOut(
            id=job.id,
            invocation_id=job.invocation_id,
            status=job.status,  # type: ignore[arg-type]  # constrained by the table CHECK
            error_type=job.error_type,
            plan=CapturePlanOut(
                hooks=list(plan.hooks),
                positions=list(plan.positions),
                include_logits=plan.include_logits,
            ),
            replay=job.replay,
            created_at=job.created_at,
            started_at=job.started_at,
            ended_at=job.ended_at,
        )

    @app.post(
        "/v1/invocations/{invocation_id}/captures",
        status_code=202,
        response_model=CaptureJobOut,
        responses=_ERRORS,
    )
    def create_capture(invocation_id: str, body: CaptureRequest) -> CaptureJobOut:
        """Validate and queue a capture; the worker process does the model work."""
        with engine.begin() as conn:
            invocation = registry_queries.get_invocation(conn, invocation_id)
            if invocation is None:
                raise ApiError(404, "INVOCATION_NOT_FOUND", f"unknown invocation {invocation_id!r}")
            if invocation.neural_access != "available":
                raise CaptureError(
                    "NEURAL_ACCESS_UNAVAILABLE", "this invocation has no model internals to read"
                )
            sequences = captures.token_sequences(conn, invocation_id)
            if "prompt" not in sequences or "output" not in sequences:
                raise CaptureError(
                    "PRIVACY_RESTRICTED", "token IDs were not stored; the run did not opt in"
                )
            plan = build_plan(
                body.hooks,
                body.token_positions,
                include_logits=body.include_logits,
                n_tokens=sequences["prompt"].n_tokens + sequences["output"].n_tokens,
                max_bytes=max_capture_bytes(),
            )
            job, _ = captures.create_job(
                conn,
                invocation_id=invocation_id,
                plan_json=plan.canonical_json(),
                plan_sha256=plan.sha256,
            )
        return _job_out(job)

    @app.get("/v1/captures/{job_id}", response_model=CaptureJobOut, responses=_ERRORS)
    def get_capture(job_id: str) -> CaptureJobOut:
        with engine.connect() as conn:
            job = captures.get_job(conn, job_id)
        if job is None:
            raise ApiError(404, "CAPTURE_NOT_FOUND", f"unknown capture {job_id!r}")
        return _job_out(job)

    @app.get(
        "/v1/invocations/{invocation_id}/measurements",
        response_model=list[MeasurementOut],
        responses=_ERRORS,
    )
    def list_measurements(invocation_id: str) -> list[MeasurementOut]:
        """Summaries only; each measurement points to its verified tensor artifact."""
        with engine.connect() as conn:
            if registry_queries.get_invocation(conn, invocation_id) is None:
                raise ApiError(404, "INVOCATION_NOT_FOUND", f"unknown invocation {invocation_id!r}")
            records = captures.list_measurements(conn, invocation_id)
        return [MeasurementOut.model_validate(r) for r in records]

    return app


def app_from_env() -> FastAPI:
    """Uvicorn factory: ``uvicorn blackbox_api.app:app_from_env --factory --host 127.0.0.1``."""
    return create_app(make_engine(database_url()), artifact_root())
