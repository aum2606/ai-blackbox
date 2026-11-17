# Compatibility report: pythia-70m-deduped

**Result:** passed · generated 2026-11-17 by `scripts/compat_pythia.py`

- Model: `EleutherAI/pythia-70m-deduped` at commit `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (safetensors, float32, CPU, eval mode)
- Software: torch 2.15.0+cpu, transformers 5.23.0, Python 3.12.13 on Windows-11-10.0.26200-SP0
- Prompt: 33 tokens; replay uses `use_cache=False`
- Cold start (download cached or not): 12.7 s
- Peak RSS: 767 MB

| Check | Result | Detail |
|---|---|---|
| model dtype is float32 | pass | torch.float32 |
| 10 repeats bit-identical (CPU) | pass | torch.equal on all 10 |
| hook points enumerated | pass | 6 layers: {'blocks.0.hook_resid_post': [1, 33, 512], 'blocks.1.hook_resid_post': [1, 33, 512], 'blocks.2.hook_resid_post': [1, 33, 512], 'blocks.3.hook_resid_post': [1, 33, 512], 'blocks.4.hook_resid_post': [1, 33, 512], 'blocks.5.hook_resid_post': [1, 33, 512]} |
| read-only hooks within 1e-06 | pass | max abs diff 0 |
| patching one residual position changes logits | pass | layer 3, last position, max abs change 0.001221 |
| greedy generation runs | pass | output ' The approved reimbursement cap' |
| SAE compatibility | not_run | Phase 4 |

Hook names map to `model.gpt_neox.layers[i]` outputs (the residual stream after layer i).
Bit-identical repeats are claimed for this CPU and software only, not across machines.
