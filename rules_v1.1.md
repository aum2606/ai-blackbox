# AI Black Box Lab — Engineering Rules

> **Status:** Mandatory for all contributors and AI coding agents  
> **Version:** 1.1  
> **Applies to:** All source code, tests, schemas, database changes, scripts, infrastructure, and documentation in this repository.  
> **Architecture reference:** `AI_Black_Box_Lab_Architecture_v1.1_Free_Tier.md` (ADD v1.1).  
> **Rule of precedence:** Safety and correctness > reproducibility and evidence integrity > simplicity and maintainability > performance > convenience. Where a rule conflicts with the accepted architecture, stop and propose an ADR rather than silently diverging.

## 1. Prime directive: build only what is needed

**Write the smallest correct, readable, tested solution that meets the current acceptance criteria.** Prefer deleting complexity over adding complexity.

- **MUST** implement only requested behavior and dependencies necessary to support it. Do not add speculative features, generic frameworks, wrappers, or compatibility layers.
- **MUST** inspect existing modules, conventions, and tests before writing code. Reuse an existing solution when it fits.
- **MUST** keep each change narrowly scoped. Do not refactor unrelated code or reformat entire files.
- **MUST** choose clear, boring, maintainable code over clever abstractions or dense one-liners.
- **SHOULD** use standard-library capabilities before adding a dependency. Add third-party packages only for a demonstrated need; pin and document them.
- **SHOULD** extract a shared abstraction only after a real, repeated pattern or an unavoidable boundary justifies it. Two similar lines are usually cheaper than a premature abstraction.
- **MUST NOT** build for imagined future scale. Follow **KISS, YAGNI, and pragmatic DRY**; do not pursue DRY at the expense of clarity.
- **MUST NOT** add unused functions, dead code, TODO-driven scaffolding, duplicate validators, redundant configuration, excessive defensive branches, or interfaces with only hypothetical implementations.
- **MUST NOT** create a new service, database, queue, cache, or language boundary without measured need and an architecture decision.

**Optimization rule:** First make it correct and measurable. Profile representative workloads; optimize confirmed bottlenecks. Prefer algorithmic and data-access improvements before adding infrastructure or rewriting in Rust. Record the benchmark and tradeoff for nontrivial optimizations.

## 2. Respect architectural boundaries

The approved baseline is **local Python-first modular monolith + separate bounded experiment CLI**, **React/Vite + TypeScript dashboard**, **SQLite + local checked artifacts**, **Rust deferred**. Do not replace it with microservices or a new stack without an ADR.

| Module | Owns | Must not own |
|---|---|---|
| `trace_ingest` | OpenTelemetry/OpenInference event normalization and correlation | Tensor interpretation |
| `run_registry` | Run IDs, invocation IDs, versions, prompt/output hashes | Model inference |
| `evidence_store` | Metadata, artifact references, checksums, retention | Scientific claims |
| `instrumentation` | Model hooks, token positions, activation capture | HTTP or UI logic |
| `feature_analysis` | SAE encoding, feature measurements, candidate labels | Unverified causal conclusions |
| `experiment_engine` | Manifests, interventions, controls, metrics, attempts | Arbitrary user-authored code execution |
| `reporting` | Evidence-linked reports and limitations | Fabricated explanations |
| `dashboard` | Visualization and user interaction through published APIs | Direct database/tensor access |

- **MUST** keep business/research logic independent of FastAPI routes, SQL models, and UI components.
- **MUST** use explicit, typed contracts at real boundaries. Do not introduce interfaces for their own sake.
- **MUST** keep GPU experiments out of HTTP request handlers. Heavy work runs in a worker process/CLI.
- **MUST** store large tensors in an artifact store, not relational rows or API JSON.
- **MUST** use OpenTelemetry/OpenInference for operational traces; AI Trace IR is a companion research record, not a competing tracing format.
- **MUST** keep the dashboard on documented `/v1` APIs; no direct access to internal databases.
- **MUST NOT** introduce Rust until a reproducible profile demonstrates a performance bottleneck and a bounded Rust comparison is approved.

## 3. Readable, idiomatic code

### Python

- Follow PEP 8, use **Ruff** for formatting/linting, and run the repository's configured type checker. Do not silence warnings with blanket ignores.
- Use descriptive identifiers, small cohesive modules, and functions with one clear responsibility. Optimize for understanding, not arbitrary line-count limits.
- Add type hints to public functions and meaningful internal boundaries. Use Pydantic for external data validation; dataclasses or plain Python types for simple internal domain objects.
- Prefer straightforward control flow, early returns where they improve clarity, and explicit failure states.
- Use context managers for resources and parameterized SQL/ORM queries for database access.
- Catch only exceptions you can handle meaningfully. Preserve the underlying cause when wrapping errors; never use bare `except`, silent `pass`, or unconditional success fallbacks.
- Use async only for truly asynchronous I/O. Do not use concurrency as decoration or block an event loop with GPU/CPU-heavy work.
- Write comments explaining **why**, assumptions, numerical constraints, or non-obvious decisions—not narrating obvious code. Docstrings belong on public APIs and complex research functions.
- **MUST NOT** use wildcard imports, mutable default arguments, untyped dictionaries across module boundaries, excessive inheritance, or needless design patterns.

### TypeScript / frontend

- Use TypeScript strict mode and the repository's ESLint/formatter configuration.
- Keep components focused; separate data fetching, domain types, and rendering when complexity warrants it.
- Use existing UI primitives; do not add an entire component library for one feature.
- Represent loading, empty, failed, and unavailable-data states honestly. Never invent neural measurements for display.

### Naming and structure

- Match existing conventions before introducing new ones. Use predictable names for files, functions, classes, API routes, experiments, and database entities.
- Keep configuration in one documented place; do not duplicate constants or scatter magic values.
- Remove obsolete code and imports during the change that makes them obsolete. Do not preserve unused compatibility paths without a documented consumer.

## 4. Scientific integrity is non-negotiable

- **MUST** distinguish the evidence classes: `observed`, `measured`, `inferred`, and `intervened`.
- **MUST** attach provenance to any neural measurement: model and tokenizer revisions, code/config versions, exact hook, token position, capture mode, seed, dtype/precision, and artifact checksum as applicable.
- **MUST** distinguish a feature label or attribution hypothesis from an experimentally supported result. Feature activity is **not** a transcript of the model's thinking.
- **MUST** define intervention metrics, clean/corrupt conditions, token alignment, controls, repeats, and exclusion criteria **before** running a causal experiment.
- **MUST** include sham/random controls as appropriate, preserve null/negative results, and state limitations.
- **MUST** refuse or flag invalid clean/corrupt token alignment rather than silently patching mismatched positions.
- **MUST** version experiment manifests and store attempts separately; completed manifests and results are immutable.
- **MUST** verify instrumentation is non-perturbing within documented tolerance before interpreting captured activations.
- **MUST NOT** claim that correlation, attention weight, an SAE label, or a single intervention proves a complete internal reasoning process.
- **MUST NOT** claim exact replay if sampling, quantization, precision, kernels, or execution conditions prevent it. Document what the replay reproduces.

## 5. Data contracts and persistence

- **MUST** keep `run_id`, `trace_id`, `span_id`, and `model_invocation_id` distinct and correctly linked.
- **MUST** version AI Trace IR and external API schemas. Validate inputs at ingress; return structured errors, not stack traces.
- **MUST** use migrations (Alembic) for database changes. Never silently change historical records or edit production schema manually.
- **MUST** store binary artifacts with verified hashes and explicit type/shape/size metadata. Prefer safe tensor/array formats over executable pickle payloads.
- **MUST** preserve document versions and hashes in retrieval experiments; never mutate the reference fixture during a run.
- **MUST** make writes idempotent where retries are possible and model job state transitions explicitly.
- **SHOULD** choose simple queries and indexes based on actual access patterns; avoid speculative denormalization and caching.

## 6. Security and privacy by default

- **MUST** default to synthetic research fixtures and **opt-in** prompt, response, document-body, and raw-tensor capture as defined by policy.
- **MUST** keep secrets out of code, commits, trace attributes, error output, and screenshots. Use environment variables or a secrets manager; provide `.env.example` with placeholders only.
- **MUST** validate all untrusted input; enforce file size, tensor size, job runtime, and artifact retention limits.
- **MUST** use safe file paths and permissions; do not allow path traversal or untrusted deserialization.
- **MUST NOT** execute arbitrary Python from dashboard submissions or treat retrieved documents/tool responses as instructions for the control plane.
- **MUST** require authentication and appropriate access checks before exposing real-user data or public deployments. Local-only MVP is not permission for an unsecured public endpoint.
- **MUST** redact sensitive data in logs and define retention/deletion behavior for persisted content.
- **MUST NOT** perform destructive operations, change infrastructure credentials, publish data, or modify production environments without explicit authorization.

## 7. Tests and quality gates

**No feature is complete merely because its code runs once.** Tests should be proportional to the actual risk; avoid test duplication or brittle assertions.

- **MUST** add or update tests for changed behavior and for every bug fix. A bug fix should include a regression test that fails before the fix where practical.
- **MUST** test domain logic with small, deterministic fixtures. Mock external services at true boundaries, not the code being tested.
- **MUST** verify schema parsing/rejection, model/run ID correlation, and error paths for affected modules.
- **MUST** run formatter/linter, type checker, and relevant tests before declaring completion. If a check cannot run, state exactly why and do not imply it passed.
- **MUST** add scientific controls for new causal methods: baseline, sham/negative control, directional effect and documented tolerance.
- **MUST** verify `capture` preserves reference logits within tolerance for model compatibility changes.
- **MUST** test artifact checksums, retry attempt history, and privacy defaults when changing those features.
- **SHOULD** keep the test suite fast by separating unit, integration, GPU compatibility, and costly research benchmark jobs.
- **MUST NOT** hardcode expected results to make a test pass, disable failing tests without explanation, or hide nondeterminism by claiming exact numerical equality where unsupported.

## 8. Dependencies, performance, and operations

- Prefer existing dependencies and pinned compatible versions. Explain why each newly added dependency is required.
- Keep local setup reproducible with documented commands and explicit environment requirements.
- Do not add Redis, PostgreSQL, Kubernetes, an event bus, a graph database, mandatory Phoenix, paid hosted services, or object-storage infrastructure to the MVP. Follow the ADD's MVP simplifications.
- Bound activation capture by selected hooks, positions, batch size, and storage quotas. Never eagerly hold all model activations in API memory.
- Use structured logs with correlation IDs and meaningful errors. Do not log raw prompts or tensors by default.
- Measure throughput, p95 latency, memory, GPU memory, and storage when proposing a performance change. Include before/after benchmark conditions.
- Prefer the simplest reliable failure handling: explicit status, actionable error, safe retry, and idempotent writes. Avoid silent fallback behavior that contaminates experiments.

## 9. How an AI coding agent must work

For **every task**, follow this sequence:

1. **Read context:** Inspect the relevant ADD section, existing code, tests, current ADRs, and the requested acceptance criteria.
2. **Scope the change:** State the smallest intended implementation and which files will change. Identify any architectural conflict before coding.
3. **Implement minimally:** Make the smallest coherent change. Reuse existing utilities. Avoid unrelated cleanup, rewrites, and speculative extensibility.
4. **Verify:** Run relevant formatting/linting, typing, tests, and scientific/integration checks. Inspect the diff for unnecessary code and security or provenance regressions.
5. **Report truthfully:** Explain what changed, why it was needed, what was tested, what could not be tested, and any important limitation. Do not claim successful validation without evidence.

### Mandatory self-review before completion

- [ ] Does each new file, dependency, class, function, and configuration key have a current, concrete purpose?
- [ ] Could this be simpler while remaining clear, correct, and testable?
- [ ] Did I duplicate existing logic, introduce speculative abstractions, or add unrelated changes?
- [ ] Are the public interfaces typed, error cases handled, and naming consistent?
- [ ] Are observed data, interpreted hypotheses, and intervention effects kept separate?
- [ ] Are model/version/token/provenance identifiers preserved and research outputs reproducible?
- [ ] Did I avoid sensitive data exposure, unsafe deserialization, and unnecessary resource use?
- [ ] Did I run the relevant checks and accurately report their results?
- [ ] Does the final diff contain only necessary changes and appropriate tests/docs?

## 10. Version control and change management

- Keep commits small, coherent, and descriptive. Commit working increments that correspond to accepted milestones.
- Prefer conventional-style commit titles such as `feat(trace): link invocations to spans` or `fix(capture): reject invalid token alignment`.
- Do not overwrite others' work, amend unrelated commits, force push, or remove failing tests to obtain a green check.
- Update the relevant README/runbook when setup, APIs, environment, or user-visible behavior changes.
- Record a new ADR for meaningful deviations in language choice, services, storage design, data semantics, or research claims. An ADR must state context, decision, alternatives, consequences, validation, and rollback.
- Treat approved architecture and these rules as living documents, but change them deliberately—not opportunistically during a coding task.

## 11. Definition of done

A task is **done** only when its acceptance criteria are met, the smallest reasonable implementation is committed or ready for review, relevant checks pass (or failures are explicitly disclosed), documentation/contracts are updated where needed, and scientific/security invariants remain intact.

**Final instruction to every coding agent:** Do not optimize for code volume or visual impressiveness. Optimize for **correctness, clarity, reproducibility, measured performance, and the smallest maintainable diff**.


## 12. Mandatory ₹0 / free-tier constraints (Architecture v1.1)

- **MUST** preserve a functional local-first, offline-capable core: FastAPI, React/Vite, SQLite, local artifacts, tiny CPU model and bounded experiment CLI.
- **MUST NOT** require credit-card signup, paid LLM APIs, managed databases, cloud GPUs, paid storage, premium CI, or always-on cloud hosting for development, tests, demonstration or baseline research.
- **MUST NOT** provision any potentially billable cloud resource or enable a paid API without explicit owner approval and an ADR describing charges, upper bounds, and rollback.
- **MUST** treat free Colab/Kaggle accelerator access as optional and quota-dependent; never assume permanent availability or guaranteed resource sizes.
- **MUST** provide a CPU/small-model fallback and label resource-blocked experiments accurately; never fabricate neural or SAE findings.
- **MUST** use bounded capture sizes, local disk quotas, timeouts and one active experiment by default, with measured adjustments when needed.
- **MUST** use versioned manifest export/import and checksum verification for manual notebook runs. Do not require notebooks to access local SQLite over the network.
- **MUST** keep local traces and artifacts private by default and never expose them through unauthenticated public tunnels.
- **SHOULD** prefer single-process/CLI solutions and local standard-library facilities before optional infrastructure or new language boundaries.
- **MUST** verify that a clean checkout runs the core system without cloud credentials. The zero-cost acceptance checks in ADD §20.2 are mandatory for every deployment-related change.

**Cost interpretation:** ₹0 means no incremental payments for external services. It does not mean existing hardware, electricity, bandwidth or time have zero real-world cost.
