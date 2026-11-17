import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from evidence_store.engine import make_engine
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from run_registry import registry
from sqlalchemy import Engine, text
from trace_ingest.exporter import RUN_ID_ATTR, make_tracer_provider

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    url = f"sqlite:///{tmp_path / 'blackbox.db'}"
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    engine = make_engine(url)
    yield engine
    engine.dispose()


def test_spans_are_persisted_when_they_end_with_parent_links_and_run_id(engine: Engine) -> None:
    with engine.begin() as conn:
        run_id = registry.create_run(
            conn,
            task_id="stale_document",
            dataset_case_id="case_0007",
            retrieval_fault="none",
            capture_content=False,
        )
    tracer = make_tracer_provider(engine).get_tracer("test")
    chain = OpenInferenceSpanKindValues.CHAIN.value
    retriever = OpenInferenceSpanKindValues.RETRIEVER.value

    with tracer.start_as_current_span(
        "rag.run", attributes={RUN_ID_ATTR: run_id, SpanAttributes.OPENINFERENCE_SPAN_KIND: chain}
    ):
        with tracer.start_as_current_span(
            "retriever.retrieve",
            attributes={RUN_ID_ATTR: run_id, SpanAttributes.OPENINFERENCE_SPAN_KIND: retriever},
        ):
            pass
        # A span that has not ended yet must not be stored.
        with engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM spans")).scalar_one() == 1

    with engine.connect() as conn:
        rows = {r.name: r for r in conn.execute(text("SELECT * FROM spans"))}
    root, child = rows["rag.run"], rows["retriever.retrieve"]
    assert root.parent_span_id is None
    assert child.parent_span_id == root.span_id
    assert child.trace_id == root.trace_id and len(root.trace_id) == 32
    assert (root.run_id, child.run_id) == (run_id, run_id)
    assert (root.openinference_span_kind, child.openinference_span_kind) == (chain, retriever)
    assert root.status_code == "UNSET"
    assert json.loads(child.attributes_json)[RUN_ID_ATTR] == run_id
