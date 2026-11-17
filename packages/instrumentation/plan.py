"""Capture plans and the static capability registry. Pure Python: no torch import here.

The registry is the single list of what the supported model can capture; requests are validated
against it before any job is queued (ADD §6.4, §7.3).
"""

import hashlib
import json
import re
from dataclasses import dataclass

MODEL_ID = "EleutherAI/pythia-70m-deduped"
MODEL_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
PRECISION = "float32"
N_LAYERS = 6
D_MODEL = 512
VOCAB_SIZE = 50304  # logits width of the pinned checkpoint; checked by the neural tests
BYTES_PER_FLOAT = 4
DEFAULT_MAX_CAPTURE_BYTES = 64 * 1024 * 1024

_HOOK = re.compile(r"blocks\.(\d+)\.hook_resid_post")


class CaptureError(Exception):
    """Invalid or failed capture; ``code`` is the structured error type reported by the API."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    tokenizer_revision: str
    precision: str


@dataclass(frozen=True)
class CapturePlan:
    hooks: tuple[str, ...]
    positions: tuple[int, ...]  # absolute token positions, sorted and unique
    include_logits: bool

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "hooks": list(self.hooks),
                "positions": list(self.positions),
                "include_logits": self.include_logits,
            },
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def supported_hooks() -> tuple[str, ...]:
    return tuple(f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS))


def hook_layer(hook_name: str) -> int:
    match = _HOOK.fullmatch(hook_name)
    if match is None or int(match.group(1)) >= N_LAYERS:
        raise CaptureError("HOOK_UNSUPPORTED", f"{hook_name!r} is not a supported hook")
    return int(match.group(1))


def estimate_bytes(plan: CapturePlan) -> int:
    """Tensor payload only; model weights, caches and clones also use memory (ADD §7.3)."""
    per_position = len(plan.hooks) * D_MODEL + (VOCAB_SIZE if plan.include_logits else 0)
    return per_position * len(plan.positions) * BYTES_PER_FLOAT


def build_plan(
    hooks: list[str],
    positions: list[int],
    *,
    include_logits: bool,
    n_tokens: int,
    max_bytes: int = DEFAULT_MAX_CAPTURE_BYTES,
) -> CapturePlan:
    """Validate against the registry and token count; negative positions count from the end."""
    if not hooks or not positions:
        raise CaptureError("HOOK_UNSUPPORTED", "at least one hook and one position are required")
    for hook in hooks:
        hook_layer(hook)
    resolved = set()
    for position in positions:
        absolute = position + n_tokens if position < 0 else position
        if not 0 <= absolute < n_tokens:
            raise CaptureError(
                "POSITION_OUT_OF_RANGE", f"position {position} is outside 0..{n_tokens - 1}"
            )
        resolved.add(absolute)
    plan = CapturePlan(tuple(sorted(set(hooks))), tuple(sorted(resolved)), include_logits)
    if estimate_bytes(plan) > max_bytes:
        raise CaptureError(
            "CAPTURE_TOO_LARGE", f"estimated {estimate_bytes(plan)} bytes exceeds {max_bytes}"
        )
    return plan
