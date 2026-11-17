"""Needs `uv sync --group neural` and the pinned model; run with `uv run pytest -m neural`."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from instrumentation.plan import D_MODEL, VOCAB_SIZE, build_plan  # noqa: E402
from instrumentation.pythia import READ_ONLY_TOLERANCE, PythiaAdapter  # noqa: E402

pytestmark = pytest.mark.neural
GOLDEN = json.loads((Path(__file__).with_name("golden_pythia_70m.json")).read_text("utf-8"))


@pytest.fixture(scope="module")
def adapter() -> PythiaAdapter:
    return PythiaAdapter()


def test_tokenization_matches_golden(adapter: PythiaAdapter) -> None:
    assert adapter.tokenize(GOLDEN["prompt"]) == GOLDEN["input_ids"]


def test_golden_logits_at_the_last_position(adapter: PythiaAdapter) -> None:
    last = adapter.logits(GOLDEN["input_ids"])[-1]
    assert last.shape == (VOCAB_SIZE,)
    top = torch.topk(last, 10)
    assert top.indices.tolist() == GOLDEN["top10_token_ids"]
    torch.testing.assert_close(
        top.values, torch.tensor(GOLDEN["top10_logits"]), rtol=1e-4, atol=1e-4
    )


def test_capture_is_read_only_and_has_the_expected_shapes(adapter: PythiaAdapter) -> None:
    ids = GOLDEN["input_ids"]
    plan = build_plan(
        ["blocks.3.hook_resid_post", "blocks.5.hook_resid_post"],
        [0, -1],
        include_logits=True,
        n_tokens=len(ids),
    )
    result = adapter.capture(ids, plan)
    assert result.max_logit_diff <= READ_ONLY_TOLERANCE
    assert set(result.activations) == {
        (hook, position)
        for hook in ("blocks.3.hook_resid_post", "blocks.5.hook_resid_post")
        for position in (0, len(ids) - 1)
    }
    assert all(t.shape == (D_MODEL,) for t in result.activations.values())
    assert set(result.logits) == {0, len(ids) - 1}
    assert len(result.argmax_ids) == len(ids)


def test_greedy_generation_is_deterministic(adapter: PythiaAdapter) -> None:
    first = adapter.generate(GOLDEN["input_ids"], 4)
    assert first == adapter.generate(GOLDEN["input_ids"], 4) and len(first) == 4
