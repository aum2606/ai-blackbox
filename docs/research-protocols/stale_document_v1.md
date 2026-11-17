# Research protocol: stale_document_v1

**Version:** 1.0 (draft, Phase 1 scope) · **Governs:** `experiments/stale_document/` · **ADD:** §8

## Question

When the retriever returns a superseded version of a policy document, does the system give the outdated answer, and can that failure be traced from the answer to the exact document version and model invocation?

## Fixtures (synthetic only)

- 10 cases, `case_0001`..`case_0010`, one question each: "What is the approved reimbursement cap for plan `<Name>`?"
- Each case has two versions of one document (`doc_key`, version 1 and 2). Version 2 is current from its `valid_from` date. Version 1 has `valid_to` equal to that date.
- Caps are 3-digit numbers in both versions, so a later phase can use matched-length tokenizations (ADD §8.3). Current and stale caps differ in every case.
- `case_0007` is Plan Delta: current cap 800 units (effective 2026-09-01), stale cap 500 units.
- `manifest.json` lists every document file with its SHA-256. The loader verifies it and refuses mismatches. Fixtures are never modified by a run.
- Ground truth (`expected_answer`) is stored in the case file, separate from any system output.

## Conditions in Phase 1

| Condition | Retrieval fault | Expected answer |
|---|---|---|
| A: clean | `none` (current version) | current cap |
| B: corrupt | `stale_version` (superseded version) | stale cap, marked incorrect |

Conditions C–F (input correction, activation patch, sham, random-site) belong to Phase 5 and are not run here.

## Measurements and evidence classes

| Item | Class |
|---|---|
| Retrieved document ID, version, hash, rank; answer text and hash | `observed` |
| `is_correct` (answer equals `expected_answer`) | `observed` |
| Span timings | `observed` |

Phase 1 makes **no** `measured`, `inferred` or `intervened` claims. The answering component is a deterministic fixture model (ADR-009), so Phase 1 shows trace and provenance correlation, not model behaviour.

## Pass criteria

- Condition A answers correctly and condition B answers with the stale value, for all 10 cases.
- Every run's `run_id`, `trace_id`, `span_id` and `model_invocation_id` link consistently, and the run links to the exact document version retrieved.

## Exclusions and limitations

- No case is excluded in Phase 1, because the fixture model is deterministic. When a real model is introduced (Phase 3), cases must be pretested for answer reliability and every exclusion recorded (ADD §8.2).
- The causal hypothesis (a mid-layer residual site carries the document's cap value) is deferred to Phase 5 and will be pre-registered there with its metric, controls and repeats.
