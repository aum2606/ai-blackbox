"""Model backends for a run: the Phase 1 fixture (default) or a real model (CLI only, ADR-011)."""

from dataclasses import dataclass
from typing import Any, Protocol

from demo_rag import model


@dataclass(frozen=True)
class Generation:
    answer: str
    # Token IDs exist only for real-model backends; they are stored only on content opt-in.
    prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] | None = None


class Backend(Protocol):
    model_id: str
    model_revision: str
    tokenizer_revision: str
    neural_access: str  # "available" only if activations can be captured later
    sampling_config: dict[str, Any]

    def generate(self, prompt: str, document_text: str) -> Generation: ...


class FixtureBackend:
    model_id = model.MODEL_ID
    model_revision = model.MODEL_REVISION
    tokenizer_revision = model.TOKENIZER_REVISION
    neural_access = model.NEURAL_ACCESS
    sampling_config: dict[str, Any] = {"deterministic": True}  # noqa: RUF012 - read-only

    def generate(self, prompt: str, document_text: str) -> Generation:
        return Generation(answer=model.generate(document_text))


FIXTURE_BACKEND = FixtureBackend()
