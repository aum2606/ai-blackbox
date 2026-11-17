"""Real-model backend (pythia-70m-deduped). Needs the ``neural`` group; import it lazily."""

from typing import Any

from instrumentation.plan import MODEL_ID, MODEL_REVISION
from instrumentation.pythia import PythiaAdapter

from demo_rag.backends import Generation

MAX_NEW_TOKENS = 4


class PythiaBackend:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    tokenizer_revision = MODEL_REVISION  # pinned to the same repository commit as the weights
    neural_access = "available"
    sampling_config: dict[str, Any] = {  # noqa: RUF012 - read-only
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": False,
    }

    def __init__(self) -> None:
        self.adapter = PythiaAdapter()

    def generate(self, prompt: str, document_text: str) -> Generation:
        prompt_ids = self.adapter.tokenize(prompt)
        output_ids = self.adapter.generate(prompt_ids, MAX_NEW_TOKENS)
        return Generation(self.adapter.decode(output_ids), prompt_ids, output_ids)
