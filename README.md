# AI Black Box Lab

Local-first, ₹0 observability + interpretability + causal debugging for AI systems.

- Architecture: [ADD v1.1](docs/architecture/AI_Black_Box_Lab_Architecture_v1.1_Free_Tier.md) · Decisions: [docs/adr/](docs/adr/README.md)
- Engineering rules (mandatory): [rules_v1.1.md](rules_v1.1.md)
- Phase 1 plan: [docs/phase1_implementation_plan.md](docs/phase1_implementation_plan.md)

## Quick start

See the [Phase 1 runbook](docs/runbooks/phase1_demo.md) (plus [Phase 2](docs/runbooks/phase2_evidence.md) for artifacts and trace events and [Phase 3](docs/runbooks/phase3_neural.md) for neural capture) for setup, the demo, the exit-test walkthrough, checks, backup and limitations. No cloud account or credentials are needed.

```bash
uv sync && npm ci --prefix apps/dashboard && uv run alembic upgrade head
uv run pytest
```
