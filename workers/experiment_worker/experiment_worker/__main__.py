"""Run one queued capture job: ``python -m experiment_worker`` (needs `uv sync --group neural`)."""

import os

from evidence_store.engine import artifact_root, database_url, make_engine
from instrumentation.plan import CaptureAdapter

from experiment_worker.worker import DEFAULT_TIMEOUT_SECONDS, run_once


def _load_adapter() -> CaptureAdapter:
    from instrumentation.pythia import PythiaAdapter  # imports torch

    return PythiaAdapter()


def main() -> None:
    timeout = int(os.environ.get("WORKER_JOB_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    result = run_once(
        make_engine(database_url()), artifact_root(), _load_adapter, timeout_seconds=timeout
    )
    print(f"job:    {result.job_id or '-'}")
    print(f"status: {result.status}" + (f" ({result.error_type})" if result.error_type else ""))


if __name__ == "__main__":
    main()
