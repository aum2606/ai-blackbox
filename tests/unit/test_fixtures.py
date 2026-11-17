import json
import shutil
from pathlib import Path

import pytest
from demo_rag.fixtures import DEFAULT_ROOT, MANIFEST_NAME, FixtureError, build_manifest, load_case

CASE_IDS = [f"case_{n:04d}" for n in range(1, 11)]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Writable copy of the fixture tree for tamper tests."""
    dest = tmp_path / "stale_document"
    shutil.copytree(DEFAULT_ROOT, dest)
    return dest


def test_committed_manifest_matches_files_on_disk() -> None:
    committed = json.loads((DEFAULT_ROOT / MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
    assert committed == build_manifest(DEFAULT_ROOT)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_each_case_has_one_current_and_one_stale_version(case_id: str) -> None:
    case = load_case(case_id)
    current = [d for d in case.documents if d.valid_to is None]
    stale = [d for d in case.documents if d.valid_to is not None]
    assert len(current) == 1 and len(stale) == 1
    assert case.expected_answer in current[0].text
    assert case.expected_answer not in stale[0].text
    assert stale[0].valid_to == current[0].valid_from


def test_case_0007_is_plan_delta() -> None:
    case = load_case("case_0007")
    assert case.expected_answer == "800 units"
    assert [d.text for d in case.documents] == [
        "Plan Delta cap is 500 units. Superseded 2026-09-01.\n",
        "Plan Delta cap is 800 units. Effective 2026-09-01.\n",
    ]


@pytest.mark.parametrize("case_id", ["case_9999", "../cases/case_0001", "case_0001.json", ""])
def test_unknown_or_malformed_case_id_is_rejected(case_id: str) -> None:
    with pytest.raises(FixtureError) as err:
        load_case(case_id)
    assert err.value.code == "CASE_NOT_FOUND"


def test_modified_document_is_rejected(tree: Path) -> None:
    (tree / "documents" / "plan_delta" / "v2.txt").write_bytes(b"Plan Delta cap is 999 units.\n")
    with pytest.raises(FixtureError) as err:
        load_case("case_0007", tree)
    assert err.value.code == "FIXTURE_HASH_MISMATCH"


def test_document_path_escaping_the_root_is_rejected(tree: Path) -> None:
    (tree.parent / "outside.txt").write_bytes(b"secret\n")
    case_file = tree / "cases" / "case_0007.json"
    case = json.loads(case_file.read_text(encoding="utf-8"))
    case["documents"][0]["path"] = "../outside.txt"
    case_file.write_text(json.dumps(case), encoding="utf-8")
    # Re-sign the manifest so only the traversal check can reject the case.
    manifest = {"schema_version": "1", "files": build_manifest(tree)}
    (tree / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FixtureError) as err:
        load_case("case_0007", tree)
    assert err.value.code == "FIXTURE_PATH_INVALID"
