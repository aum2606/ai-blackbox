"""Frozen stale-document fixtures: loading with SHA-256 and path verification.

Every byte read from the fixture tree is checked against ``manifest.json`` so a run can never
silently use a modified reference document (rules §5).
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# apps/demo_rag/demo_rag/fixtures.py -> repository root is three levels above the package
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "stale_document"
MANIFEST_NAME = "manifest.json"
_CASE_ID = re.compile(r"case_\d{4}")


class FixtureError(Exception):
    """Fixture problem; ``code`` is the structured error type reported by the API."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DocumentVersion:
    version: int
    relative_path: str
    valid_from: str
    valid_to: str | None  # None marks the current version
    sha256: str
    text: str


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    doc_key: str
    expected_answer: str
    documents: tuple[DocumentVersion, ...]


def build_manifest(root: Path = DEFAULT_ROOT) -> dict[str, str]:
    """Hash every file under ``cases/`` and ``documents/``.

    To refresh after an intentional fixture change, write the result to ``manifest.json``.
    """
    files = sorted(
        p for sub in ("cases", "documents") for p in (root / sub).rglob("*") if p.is_file()
    )
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }


def _read_verified(root: Path, relative_path: str) -> bytes:
    root = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise FixtureError("FIXTURE_PATH_INVALID", f"{relative_path!r} escapes the fixture root")
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
    data = path.read_bytes()
    expected = manifest.get(path.relative_to(root).as_posix())
    if expected != hashlib.sha256(data).hexdigest():
        raise FixtureError(
            "FIXTURE_HASH_MISMATCH",
            f"{relative_path} is not listed in or differs from the manifest",
        )
    return data


def load_case(case_id: str, root: Path = DEFAULT_ROOT) -> Case:
    # Validate the ID shape first so it can never be used as a path component.
    if not _CASE_ID.fullmatch(case_id) or not (root / "cases" / f"{case_id}.json").is_file():
        raise FixtureError("CASE_NOT_FOUND", f"unknown case {case_id!r}")
    raw = json.loads(_read_verified(root, f"cases/{case_id}.json"))
    documents = []
    for entry in raw["documents"]:
        data = _read_verified(root, entry["path"])
        documents.append(
            DocumentVersion(
                version=entry["version"],
                relative_path=entry["path"],
                valid_from=entry["valid_from"],
                valid_to=entry["valid_to"],
                sha256=hashlib.sha256(data).hexdigest(),
                text=data.decode("utf-8"),
            )
        )
    return Case(
        case_id=raw["case_id"],
        question=raw["question"],
        doc_key=raw["doc_key"],
        expected_answer=raw["expected_answer"],
        documents=tuple(documents),
    )
