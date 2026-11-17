"""FastAPI app for the /v1 endpoints. Bind to 127.0.0.1 only (ADD §10.2)."""

import hashlib
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from demo_rag.fixtures import FixtureError, load_case
from demo_rag.pipeline import run_case
from evidence_store.engine import make_engine
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from run_registry import queries as registry_queries
from sqlalchemy import Engine
from trace_ingest.exporter import make_tracer_provider
from trace_ingest.queries import list_spans

from blackbox_api.schemas import (
    AnswerOut,
    ErrorOut,
    HealthOut,
    InvocationOut,
    RetrievalOut,
    RunCreate,
    RunCreated,
    RunOut,
    SpanOut,
)

_log = logging.getLogger(__name__)
DEFAULT_URL = "sqlite:///./.local/blackbox.db"
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


def create_app(engine: Engine) -> FastAPI:
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

    return app


def app_from_env() -> FastAPI:
    """Uvicorn factory: ``uvicorn blackbox_api.app:app_from_env --factory --host 127.0.0.1``."""
    return create_app(make_engine(os.environ.get("DATABASE_URL", DEFAULT_URL)))
