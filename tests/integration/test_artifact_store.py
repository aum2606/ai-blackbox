import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from evidence_store.artifacts import ArtifactError, get_verified, put_bytes
from sqlalchemy import Connection, Engine, text


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "store"  # the engine fixture keeps its database file in tmp_path


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    with engine.begin() as connection:
        yield connection


def _put(conn: Connection, root: Path, payload: bytes, **kwargs: object) -> str:
    return put_bytes(conn, root, payload, mime_type="text/plain", category="raw", **kwargs)  # type: ignore[arg-type]


def test_round_trip_and_metadata(conn: Connection, root: Path) -> None:
    artifact_id = _put(conn, root, b"hello")
    assert get_verified(conn, root, artifact_id) == b"hello"
    row = conn.execute(
        text("SELECT sha256, nbytes FROM artifacts WHERE id = :i"), {"i": artifact_id}
    )
    assert tuple(row.one()) == (hashlib.sha256(b"hello").hexdigest(), 5)


def test_identical_bytes_give_one_artifact_and_one_file(conn: Connection, root: Path) -> None:
    assert _put(conn, root, b"same") == _put(conn, root, b"same")
    assert len([p for p in root.rglob("*") if p.is_file()]) == 1


def test_modified_or_missing_file_is_an_integrity_error(conn: Connection, root: Path) -> None:
    artifact_id = _put(conn, root, b"original")
    path = next(p for p in root.rglob("*") if p.is_file())
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactError) as err:
        get_verified(conn, root, artifact_id)
    assert err.value.code == "ARTIFACT_INTEGRITY_ERROR"
    path.unlink()
    with pytest.raises(ArtifactError) as err:
        get_verified(conn, root, artifact_id)
    assert err.value.code == "ARTIFACT_INTEGRITY_ERROR"


def test_quota_rejects_before_writing_anything(conn: Connection, root: Path) -> None:
    _put(conn, root, b"12345", max_bytes=8)
    with pytest.raises(ArtifactError) as err:
        _put(conn, root, b"6789a", max_bytes=8)
    assert err.value.code == "ARTIFACT_QUOTA_EXCEEDED"
    assert len([p for p in root.rglob("*") if p.is_file()]) == 1


def test_path_escaping_the_store_root_is_rejected(conn: Connection, root: Path) -> None:
    artifact_id = _put(conn, root, b"x")
    conn.execute(text("UPDATE artifacts SET relative_path = '../outside'"))
    with pytest.raises(ArtifactError) as err:
        get_verified(conn, root, artifact_id)
    assert err.value.code == "ARTIFACT_PATH_INVALID"


def test_unknown_artifact_and_lineage(conn: Connection, root: Path) -> None:
    with pytest.raises(ArtifactError) as err:
        get_verified(conn, root, "art_missing")
    assert err.value.code == "ARTIFACT_NOT_FOUND"
    parent = _put(conn, root, b"parent")
    child = _put(conn, root, b"child", parents=[parent])
    _put(conn, root, b"child", parents=[parent])  # repeating must not duplicate lineage
    rows = conn.execute(text("SELECT child_id, parent_id, relation FROM artifact_lineage")).all()
    assert [tuple(r) for r in rows] == [(child, parent, "derived_from")]
