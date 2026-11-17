import pytest
from demo_rag import model
from demo_rag.fixtures import load_case
from demo_rag.retriever import retrieve


def test_retriever_is_deterministic_and_fault_selects_the_version() -> None:
    case = load_case("case_0007")
    current, stale = retrieve(case, "none"), retrieve(case, "stale_version")
    assert (current.version, stale.version) == (2, 1)
    assert retrieve(case, "none") == current
    assert retrieve(case, "stale_version") == stale


def test_model_extracts_the_cap_from_each_version() -> None:
    case = load_case("case_0007")
    assert model.generate(retrieve(case, "none").text) == "800 units"
    assert model.generate(retrieve(case, "stale_version").text) == "500 units"


def test_model_fails_when_no_cap_is_present() -> None:
    with pytest.raises(model.ModelError) as err:
        model.generate("Plan Delta has no stated limit.")
    assert err.value.code == "ANSWER_NOT_EXTRACTABLE"
