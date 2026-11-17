"""pythia-70m-deduped adapter: tokenize, greedy generate, and read-only activation capture.

Needs the ``neural`` dependency group. Replay is teacher-forced on given token IDs with
``use_cache=False`` so hooks see every position (ADD §4.2, §7.2).
"""

from typing import Any

import torch
import transformers
from safetensors.torch import save
from transformers import AutoModelForCausalLM, AutoTokenizer

from instrumentation.plan import (
    MODEL_ID,
    MODEL_REVISION,
    PRECISION,
    CaptureData,
    CapturedMeasurement,
    CaptureError,
    CapturePlan,
    ModelIdentity,
    hook_layer,
)

READ_ONLY_TOLERANCE = 1e-6  # max |hooked - unhooked| logits; recorded with every capture


def _hidden(output: Any) -> torch.Tensor:
    """A GPT-NeoX layer returns the residual stream, bare or as the first tuple item."""
    hidden = output[0] if isinstance(output, tuple) else output
    assert isinstance(hidden, torch.Tensor)  # noqa: S101 - guards a library-version assumption
    return hidden


class PythiaAdapter:
    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.float32,  # the checkpoint is fp16; transformers would keep that by default
            use_safetensors=True,
            trust_remote_code=False,
        ).eval()  # type: ignore[no-untyped-call]  # transformers' eval() is untyped

    def identity(self) -> ModelIdentity:
        # The tokenizer is pinned to the same repository commit as the weights.
        return ModelIdentity(MODEL_ID, MODEL_REVISION, MODEL_REVISION, PRECISION)

    def tokenize(self, text: str) -> list[int]:
        return [int(t) for t in self._tokenizer(text)["input_ids"]]

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.decode(token_ids))

    def generate(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedy continuation; returns only the new tokens."""
        ids = torch.tensor([token_ids])
        with torch.no_grad():
            out = self._model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False
            )
        return [int(t) for t in out[0, len(token_ids) :]]

    def _forward(self, token_ids: list[int]) -> torch.Tensor:
        with torch.no_grad():
            logits: torch.Tensor = self._model(torch.tensor([token_ids]), use_cache=False).logits
        return logits[0]

    def logits(self, token_ids: list[int]) -> torch.Tensor:
        return self._forward(token_ids)

    def capture(self, token_ids: list[int], plan: CapturePlan) -> CaptureData:
        baseline = self._forward(token_ids)
        positions = list(plan.positions)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in {hook_layer(h) for h in plan.hooks}:

            def read_only(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer] = _hidden(output)[0, positions, :].detach().clone()

            handles.append(self._model.gpt_neox.layers[layer].register_forward_hook(read_only))
        try:
            hooked = self._forward(token_ids)
        finally:
            for handle in handles:
                handle.remove()
        max_diff = float((hooked - baseline).abs().max().item())
        if max_diff > READ_ONLY_TOLERANCE:
            raise CaptureError(
                "NUMERICAL_MISMATCH", f"hooks changed logits by {max_diff:.3g} (> tolerance)"
            )
        measurements = [
            _measurement(
                "activation", hook, hook_layer(hook), position, captured[hook_layer(hook)][i]
            )
            for hook in plan.hooks
            for i, position in enumerate(positions)
        ]
        if plan.include_logits:
            measurements += [
                _measurement("logits", "unembed.logits", None, position, baseline[position])
                for position in positions
            ]
        return CaptureData(
            measurements=measurements,
            argmax_ids=[int(t) for t in baseline.argmax(-1)],
            max_logit_diff=max_diff,
            runtime={
                "mode": "teacher_forced",
                "device": "cpu",
                "dtype": PRECISION,
                "use_cache": False,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        )


def _measurement(
    kind: str, hook_name: str, layer: int | None, position: int, tensor: torch.Tensor
) -> CapturedMeasurement:
    flat = tensor.detach().contiguous()
    return CapturedMeasurement(
        kind=kind,
        hook_name=hook_name,
        layer=layer,
        token_position=position,
        dtype=str(flat.dtype).removeprefix("torch."),
        shape=list(flat.shape),
        payload=save({"data": flat}),
        summary={
            "l2_norm": float(flat.norm().item()),
            "mean": float(flat.mean().item()),
            "max_abs": float(flat.abs().max().item()),
        },
    )
