"""CLI: ``python -m demo_rag --case case_0007 --fault stale_version --capture-content``.

Needs a migrated database (``alembic upgrade head``); the URL comes from DATABASE_URL.
"""

import argparse

from evidence_store.engine import database_url, make_engine
from trace_ingest.exporter import make_tracer_provider

from demo_rag.fixtures import FixtureError
from demo_rag.pipeline import run_case


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo_rag", description="Run one stale-document case")
    parser.add_argument("--case", required=True)
    parser.add_argument("--fault", choices=["none", "stale_version"], default="none")
    parser.add_argument("--capture-content", action="store_true", help="store the answer text")
    args = parser.parse_args()

    engine = make_engine(database_url())
    tracer = make_tracer_provider(engine).get_tracer("demo_rag")
    try:
        result = run_case(
            engine, tracer, args.case, args.fault, capture_content=args.capture_content
        )
    except FixtureError as err:
        raise SystemExit(f"{err.code}: {err}") from err
    print(f"run_id:   {result.run_id}")
    print(f"trace_id: {result.trace_id}")
    print(f"status:   {result.status}" + (f" ({result.error_type})" if result.error_type else ""))
    print(f"view:     http://127.0.0.1:5173/?run={result.run_id}")


if __name__ == "__main__":
    main()
