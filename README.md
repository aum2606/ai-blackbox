# AI Black Box Lab

Local-first, ₹0 observability + interpretability + causal debugging for AI systems.

- Architecture: [ADD v1.1](docs/architecture/AI_Black_Box_Lab_Architecture_v1.1_Free_Tier.md) · Decisions: [docs/adr/](docs/adr/README.md)
- Engineering rules (mandatory): [rules_v1.1.md](rules_v1.1.md)
- Phase 1 plan: [docs/phase1_implementation_plan.md](docs/phase1_implementation_plan.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/). No cloud account or credentials are needed.

```bash
uv sync
uv run ruff format --check && uv run ruff check && uv run pytest
```

Do not keep the SQLite database (`.local/`) in a cloud-synced folder.
