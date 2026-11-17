"""Deterministic extractive 'model' used in Phase 1 (ADR-009).

It reads the cap out of the retrieved text. It has no weights, so neural access is unavailable.
"""

import hashlib
import re
from pathlib import Path

MODEL_ID = "fixture-extractive"
# shortcut: hashes the source bytes, so the revision can differ if line endings differ between
# checkouts; replace with a git commit once the project has releases.
MODEL_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TOKENIZER_REVISION = "not_applicable"
NEURAL_ACCESS = "unavailable"

_CAP = re.compile(r"cap is (\d+ units)")


class ModelError(Exception):
    """The model could not answer; ``code`` is the structured error type stored on the run."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def render_prompt(context: str, question: str) -> str:
    return f"Context: {context.strip()}\nQuestion: {question}\nAnswer:"


def generate(context: str) -> str:
    match = _CAP.search(context)
    if match is None:
        raise ModelError("ANSWER_NOT_EXTRACTABLE", "no cap found in the retrieved document")
    return match.group(1)
