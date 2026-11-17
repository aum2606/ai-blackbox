"""Phase 0 compatibility gate for pythia-70m-deduped (ADD §7.2). Needs `uv sync --group neural`.

Run: ``uv run python scripts/compat_pythia.py``. It downloads the pinned model on first use, checks
that hooks are read-only and patchable, and writes the report plus a small golden-logit file.
"""

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-70m-deduped"
REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
# Fixed synthetic prompt in the Phase 1 rendered-prompt format.
PROMPT = (
    "Context: Plan Delta cap is 800 units. Effective 2026-09-01.\n"
    "Question: What is the approved reimbursement cap for plan Delta?\nAnswer:"
)
READ_ONLY_TOLERANCE = 1e-6
PATCH_LAYER = 3
REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "docs" / "compatibility" / "pythia-70m-deduped.md"
GOLDEN = REPO / "tests" / "neural" / "golden_pythia_70m.json"


def peak_rss_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [  # noqa: RUF012 - ctypes layout
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                *[(n, ctypes.c_size_t) for n in ("a", "b", "c", "d", "e", "f", "g")],
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi  # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 2**20
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def layer_hidden(output: object) -> torch.Tensor:
    """A GPT-NeoX layer returns the residual stream, bare or as the first tuple item."""
    return output[0] if isinstance(output, tuple) else output  # type: ignore[return-value]


def main() -> int:
    results: list[tuple[str, str, str]] = []

    def record(check: str, ok: bool, detail: str) -> bool:
        results.append((check, "pass" if ok else "FAIL", detail))
        return ok

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, use_safetensors=True, trust_remote_code=False
    ).eval()
    cold_start = time.perf_counter() - started
    layers = model.gpt_neox.layers
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    n_tokens = ids.shape[1]

    def forward() -> torch.Tensor:
        with torch.no_grad():
            return model(ids, use_cache=False).logits

    baseline = forward()
    repeats = [forward() for _ in range(10)]
    record(
        "10 repeats bit-identical (CPU)",
        all(torch.equal(baseline, r) for r in repeats),
        "torch.equal on all 10",
    )

    shapes: dict[str, list[int]] = {}
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out, name=f"blocks.{i}.hook_resid_post": shapes.__setitem__(
                name, list(layer_hidden(out).shape)
            )
        )
        for i, layer in enumerate(layers)
    ]
    hooked = forward()
    for handle in handles:
        handle.remove()
    max_diff = (hooked - baseline).abs().max().item()
    record("hook points enumerated", len(shapes) == len(layers), f"{len(shapes)} layers: {shapes}")
    record(
        f"read-only hooks within {READ_ONLY_TOLERANCE:g}",
        max_diff <= READ_ONLY_TOLERANCE,
        f"max abs diff {max_diff:.3g}",
    )

    def patch(_m: object, _i: object, out: object) -> object:
        hidden = layer_hidden(out).clone()
        hidden[:, -1, :] += 1.0
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

    handle = layers[PATCH_LAYER].register_forward_hook(patch)
    patched = forward()
    handle.remove()
    patch_effect = (patched[0, -1] - baseline[0, -1]).abs().max().item()
    record(
        "patching one residual position changes logits",
        patched.shape == baseline.shape and patch_effect > 1e-4,
        f"layer {PATCH_LAYER}, last position, max abs change {patch_effect:.4g}",
    )

    generated = model.generate(ids, max_new_tokens=4, do_sample=False, use_cache=False)
    answer = tokenizer.decode(generated[0, n_tokens:])
    record("greedy generation runs", generated.shape[1] == n_tokens + 4, f"output {answer!r}")

    last = baseline[0, -1]
    top = torch.topk(last, 10)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "prompt": PROMPT,
                "input_ids": ids[0].tolist(),
                "top10_token_ids": top.indices.tolist(),
                "top10_logits": [round(v, 6) for v in top.values.tolist()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    passed = all(status == "pass" for _, status, _ in results)
    lines = [
        "# Compatibility report: pythia-70m-deduped",
        "",
        f"**Result:** {'passed' if passed else 'FAILED'} · generated {datetime.now(UTC):%Y-%m-%d}"
        " by `scripts/compat_pythia.py`",
        "",
        f"- Model: `{MODEL_ID}` at commit `{REVISION}` (safetensors, float32, CPU, eval mode)",
        f"- Software: torch {torch.__version__}, transformers {transformers.__version__},"
        f" Python {platform.python_version()} on {platform.platform()}",
        f"- Prompt: {n_tokens} tokens; replay uses `use_cache=False`",
        f"- Cold start (download cached or not): {cold_start:.1f} s",
        f"- Peak RSS: {peak_rss_mb():.0f} MB",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        *[f"| {c} | {s} | {d} |" for c, s, d in results],
        "| SAE compatibility | not_run | Phase 4 |",
        "",
        "Hook names map to `model.gpt_neox.layers[i]` outputs (the residual stream after layer i).",
        "Bit-identical repeats are claimed for this CPU and software only, not across machines.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
