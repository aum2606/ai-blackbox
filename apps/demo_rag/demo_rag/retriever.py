"""Deterministic retrieval from frozen fixtures, with an injectable stale-version fault."""

from typing import Literal

from demo_rag.fixtures import Case, DocumentVersion

RetrievalFault = Literal["none", "stale_version"]


def retrieve(case: Case, fault: RetrievalFault) -> DocumentVersion:
    """Return the current version, or the superseded one when the fault is injected."""
    want_current = fault == "none"
    (document,) = (d for d in case.documents if (d.valid_to is None) == want_current)
    return document
