import pytest
from instrumentation.plan import (
    D_MODEL,
    VOCAB_SIZE,
    CaptureError,
    build_plan,
    estimate_bytes,
    supported_hooks,
)

HOOK = "blocks.3.hook_resid_post"


def test_plan_resolves_negative_positions_and_is_canonical() -> None:
    a = build_plan([HOOK], [-1, 0, -1], include_logits=False, n_tokens=10)
    b = build_plan([HOOK], [9, 0], include_logits=False, n_tokens=10)
    assert a.positions == (0, 9) and a.sha256 == b.sha256


@pytest.mark.parametrize("hook", ["blocks.6.hook_resid_post", "blocks.3.hook_mlp_out", "x", ""])
def test_unsupported_hook_is_rejected(hook: str) -> None:
    with pytest.raises(CaptureError) as err:
        build_plan([hook], [0], include_logits=False, n_tokens=10)
    assert err.value.code == "HOOK_UNSUPPORTED"


@pytest.mark.parametrize("position", [10, -11, 100])
def test_position_outside_the_sequence_is_rejected(position: int) -> None:
    with pytest.raises(CaptureError) as err:
        build_plan([HOOK], [position], include_logits=False, n_tokens=10)
    assert err.value.code == "POSITION_OUT_OF_RANGE"


def test_byte_estimate_and_cap() -> None:
    plan = build_plan([HOOK], [0, 1], include_logits=True, n_tokens=10)
    assert estimate_bytes(plan) == 2 * (D_MODEL + VOCAB_SIZE) * 4
    with pytest.raises(CaptureError) as err:
        build_plan(
            list(supported_hooks()), [0, 1], include_logits=True, n_tokens=10, max_bytes=1000
        )
    assert err.value.code == "CAPTURE_TOO_LARGE"
