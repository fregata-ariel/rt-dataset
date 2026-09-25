"""Tests for the content-addressed bundle store and SQLite index."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path

import pytest
import viewer_bundle_fixtures as vbf

import plateau_rt.viewer.store as store_module
from plateau_rt.viewer.extract import ExtractLimits, UnsafeArchiveError, safe_extract
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store, StoreVersionError, compute_digest

TREE: dict[str, bytes] = {
    "bundle.json": b'{"bundle_format_version": 1}\n',
    "dataset/dataset_manifest.json": b'{"schema_version": 3}\n',
    "dataset/views/v0/a.npy": bytes(range(256)),
    "scene/scene.xml": b"<scene/>\n",
}


def _settings(tmp_path: Path) -> ViewerSettings:
    """Build test settings with a small data dir."""
    return ViewerSettings(
        data_dir=tmp_path / "data", max_files=1000, max_extracted_bytes=10 * 1024 * 1024
    )


def _write_tree(root: Path, mapping: dict[str, bytes]) -> None:
    """Write ``mapping`` of relpath to bytes under ``root``."""
    for rel, data in mapping.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _stage(store: Store, mapping: dict[str, bytes]) -> Path:
    """Create a staging dir in ``store`` holding ``mapping``."""
    staging = store.new_staging()
    _write_tree(staging, mapping)
    return staging


def test_zip_and_targz_of_the_same_bundle_commit_to_the_same_digest(tmp_path: Path) -> None:
    """Zip and tar.gz of one tree share a digest across stores."""
    src = tmp_path / "src"
    _write_tree(src, TREE)
    zip_path = tmp_path / "bundle.zip"
    tgz_path = tmp_path / "bundle.tar.gz"
    vbf.make_archive(src, zip_path, "zip", root_name="bundle")
    vbf.make_archive(src, tgz_path, "tar.gz")
    expected = compute_digest(src)[0]
    store_a = Store(
        ViewerSettings(
            data_dir=tmp_path / "a", max_files=1000, max_extracted_bytes=10 * 1024 * 1024
        )
    )
    store_b = Store(
        ViewerSettings(
            data_dir=tmp_path / "b", max_files=1000, max_extracted_bytes=10 * 1024 * 1024
        )
    )
    staging_a = store_a.new_staging()
    report_a = safe_extract(zip_path, staging_a, store_a.settings.extract_limits)
    digest_a, created_a = store_a.commit(report_a.root, name="zip", members=[])
    assert created_a is True
    staging_b = store_b.new_staging()
    report_b = safe_extract(tgz_path, staging_b, store_b.settings.extract_limits)
    digest_b, created_b = store_b.commit(report_b.root, name="tgz", members=[])
    assert created_b is True
    assert digest_a == digest_b == expected
    store_c = Store(
        ViewerSettings(
            data_dir=tmp_path / "c", max_files=1000, max_extracted_bytes=10 * 1024 * 1024
        )
    )
    staging_c1 = store_c.new_staging()
    report_c1 = safe_extract(zip_path, staging_c1, store_c.settings.extract_limits)
    digest_c1, created_c1 = store_c.commit(report_c1.root, name="first", members=[])
    assert (digest_c1, created_c1) == (expected, True)
    staging_c2 = store_c.new_staging()
    report_c2 = safe_extract(tgz_path, staging_c2, store_c.settings.extract_limits)
    digest_c2, created_c2 = store_c.commit(report_c2.root, name="second", members=[])
    assert (digest_c2, created_c2) == (expected, False)


def test_commit_twice_keeps_one_raw(tmp_path: Path) -> None:
    """A second identical commit keeps one raw and drops its staging."""
    store = Store(_settings(tmp_path))
    first = _stage(store, TREE)
    digest, created = store.commit(first, name="first", members=[])
    assert created is True
    second = _stage(store, TREE)
    digest2, created2 = store.commit(second, name="second", members=[])
    assert created2 is False
    assert digest2 == digest
    assert [entry.name for entry in store.bundles_dir.iterdir()] == [digest]
    with store.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM bundles").fetchone()[0]
    assert count == 1
    assert list(store.staging_dir.iterdir()) == []
    assert store.get(digest) is not None and store.get(digest).name == "first"  # type: ignore[union-attr]


@pytest.mark.parametrize("shared_store", [True, False])
def test_concurrent_commits_of_the_same_content(tmp_path: Path, shared_store: bool) -> None:
    """Concurrent identical commits create exactly one bundle."""
    for round_index in range(5):
        data_dir = tmp_path / f"round{round_index}"
        base = ViewerSettings(
            data_dir=data_dir, max_files=1000, max_extracted_bytes=10 * 1024 * 1024
        )
        if shared_store:
            store = Store(base)
            stores = (store, store)
        else:
            stores = (Store(base), Store(base))
        primary = stores[0]
        staging_a = _stage(primary, TREE)
        staging_b = _stage(primary, TREE)
        barrier = threading.Barrier(2)
        outcomes: list[tuple[str, bool]] = []
        errors: list[BaseException] = []

        def _work(store: Store, staging: Path) -> None:
            try:
                barrier.wait(timeout=30)
                outcomes.append(store.commit(staging, name="x", members=[]))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_work, args=(stores[0], staging_a)),
            threading.Thread(target=_work, args=(stores[1], staging_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert errors == []
        assert sorted(created for _, created in outcomes) == [False, True]
        assert outcomes[0][0] == outcomes[1][0]
        assert len(list(primary.bundles_dir.iterdir())) == 1
        with primary.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM bundles").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == len(TREE)
        assert list(primary.staging_dir.iterdir()) == []


def test_find_by_file_sha256(tmp_path: Path) -> None:
    """File sha256 lookup finds bundles sharing content."""
    store = Store(_settings(tmp_path))
    tree_a = {"shared.bin": b"shared", "only_a.bin": b"aaa"}
    tree_b = {"shared.bin": b"shared", "only_b.bin": b"bbb"}
    digest_a, _ = store.commit(_stage(store, tree_a), name="a", members=[])
    digest_b, _ = store.commit(_stage(store, tree_b), name="b", members=[])
    sha_a = hashlib.sha256(b"aaa").hexdigest()
    sha_shared = hashlib.sha256(b"shared").hexdigest()
    found_a = store.find_by_file_sha256(sha_a)
    assert len(found_a) == 1
    assert found_a[0].digest == digest_a and found_a[0].relpath == "only_a.bin"
    shared = store.find_by_file_sha256(sha_shared)
    assert [record.digest for record in shared] == sorted([digest_a, digest_b])
    assert all(record.relpath == "shared.bin" for record in shared)
    assert store.find_by_file_sha256(hashlib.sha256(b"missing").hexdigest()) == []
    assert store.find_by_file_sha256(sha_a.upper()) == found_a
    with pytest.raises(ValueError):
        store.find_by_file_sha256("xyz")


def test_delete_removes_raw_derived_and_rows(tmp_path: Path) -> None:
    """Delete drops the directory, derived outputs and all index rows."""
    store = Store(_settings(tmp_path))
    digest_a, _ = store.commit(_stage(store, {"a.txt": b"a"}), name="a", members=[])
    digest_b, _ = store.commit(_stage(store, {"b.txt": b"b"}), name="b", members=[])
    derived_file = store.derived_dir(digest_a) / "dataset" / "overview" / "v1" / "p" / "l"
    derived_file.mkdir(parents=True, exist_ok=True)
    (derived_file / "overview.json").write_bytes(b"{}")
    now = "2026-09-25T12:00:00.000000Z"
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO derived(digest, member, deriver, version, params_key, links_key,"
            " status, error, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (digest_a, "dataset", "overview", 1, "p", "l", "ready", None, now),
        )
        conn.execute(
            "INSERT INTO jobs(job_id, kind, digest, member, deriver, params_key, status,"
            " stage, done_bytes, total_bytes, error, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "job-a",
                "derive",
                digest_a,
                "dataset",
                "overview",
                "p",
                "done",
                None,
                1,
                1,
                None,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO jobs(job_id, kind, digest, member, deriver, params_key, status,"
            " stage, done_bytes, total_bytes, error, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("job-b", "derive", digest_b, None, None, None, "done", None, 1, 1, None, now, now),
        )
        conn.execute("COMMIT")
    assert store.delete(digest_a) is True
    assert not store.bundle_dir(digest_a).exists()
    with store.connect() as conn:
        for table in ("bundles", "files", "derived", "jobs"):
            rows = conn.execute(f"SELECT * FROM {table} WHERE digest = ?", (digest_a,)).fetchall()
            assert rows == []
        assert (
            conn.execute("SELECT COUNT(*) FROM bundles WHERE digest = ?", (digest_b,)).fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM jobs WHERE job_id = ?", ("job-b",)).fetchone()[0]
            == 1
        )
    assert (store.raw_dir(digest_b) / "b.txt").read_bytes() == b"b"
    assert list(store.staging_dir.iterdir()) == []
    assert store.get(digest_a) is None
    assert store.delete(digest_a) is False


def test_new_store_reopens_as_version_1(tmp_path: Path) -> None:
    """A fresh store has version 1 metadata, WAL mode and the full schema."""
    settings = _settings(tmp_path)
    store = Store(settings)
    assert store.schema_version == 1
    with store.connect() as conn:
        meta = {row[0]: row[1] for row in conn.execute("SELECT key, value FROM store_meta")}
    from plateau_rt.viewer import VIEWER_VERSION as _vv

    assert meta["store_schema_version"] == "1"
    assert meta["created_by_viewer"] == _vv == meta["migrated_by_viewer"]
    with sqlite3.connect(str(store.index_path)) as raw:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        expected: dict[str, list[str]] = {
            "store_meta": ["key", "value"],
            "bundles": [
                "digest",
                "name",
                "members_json",
                "created_at",
                "total_bytes",
                "file_count",
                "status",
                "error",
                "validated_with",
            ],
            "files": ["digest", "relpath", "size", "sha256"],
            "derived": [
                "digest",
                "member",
                "deriver",
                "version",
                "params_key",
                "links_key",
                "status",
                "error",
                "updated_at",
            ],
            "jobs": [
                "job_id",
                "kind",
                "digest",
                "member",
                "deriver",
                "params_key",
                "status",
                "stage",
                "done_bytes",
                "total_bytes",
                "error",
                "created_at",
                "updated_at",
            ],
        }
        for table, columns in expected.items():
            info = raw.execute(f"PRAGMA table_info({table})").fetchall()
            assert [row[1] for row in info] == columns
        sql = raw.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'files_sha256'"
        ).fetchone()[0]
        assert "files" in sql and "sha256" in sql
    digest, _ = store.commit(_stage(store, TREE), name="n", members=[])
    reopened = Store(settings)
    assert reopened.schema_version == 1
    assert reopened.get(digest) is not None


def test_newer_store_version_is_refused(tmp_path: Path) -> None:
    """A store with a newer schema version is refused untouched."""
    settings = _settings(tmp_path)
    Store(settings)
    index = settings.data_dir.absolute() / "index.sqlite"
    with sqlite3.connect(str(index)) as raw:
        raw.execute("UPDATE store_meta SET value = '999' WHERE key = 'store_schema_version'")
        raw.execute("UPDATE store_meta SET value = '9.9.9' WHERE key = 'migrated_by_viewer'")
        raw.commit()
    with pytest.raises(StoreVersionError) as excinfo:
        Store(settings)
    err = excinfo.value
    assert "999" in str(err) and "9.9.9" in str(err)
    assert err.found == 999 and err.supported == 1
    with sqlite3.connect(str(index)) as raw:
        version = raw.execute(
            "SELECT value FROM store_meta WHERE key = 'store_schema_version'"
        ).fetchone()[0]
    assert version == "999"


def test_missing_or_garbage_version_is_refused(tmp_path: Path) -> None:
    """Missing or non-integer schema versions are refused."""
    settings = _settings(tmp_path)
    Store(settings)
    index = settings.data_dir.absolute() / "index.sqlite"
    with sqlite3.connect(str(index)) as raw:
        raw.execute("DELETE FROM store_meta WHERE key = 'store_schema_version'")
        raw.commit()
    with pytest.raises(StoreVersionError) as excinfo:
        Store(settings)
    assert excinfo.value.found is None
    with sqlite3.connect(str(index)) as raw:
        raw.execute("INSERT INTO store_meta(key, value) VALUES ('store_schema_version', 'abc')")
        raw.commit()
    with pytest.raises(StoreVersionError):
        Store(settings)


def test_migrations_are_applied_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Migrations run once in order, including on fresh stores."""
    settings = _settings(tmp_path)
    Store(settings)
    calls: list[str] = []

    def _fake(conn: sqlite3.Connection, data_dir: Path) -> None:
        calls.append("m2")
        conn.execute("CREATE TABLE extra (id INTEGER PRIMARY KEY)")

    monkeypatch.setattr(store_module, "MIGRATIONS", {**store_module.MIGRATIONS, 2: _fake})
    reopened = Store(settings)
    assert calls == ["m2"]
    assert reopened.schema_version == 2
    with reopened.connect() as conn:
        version = conn.execute(
            "SELECT value FROM store_meta WHERE key = 'store_schema_version'"
        ).fetchone()[0]
        assert version == "2"
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'extra'"
            ).fetchone()
            is not None
        )
    Store(settings)
    assert calls == ["m2"]
    order: list[str] = []
    original = store_module._migrate_to_1

    def _wrapper(conn: sqlite3.Connection, data_dir: Path) -> None:
        order.append("m1")
        original(conn, data_dir)

    def _fake2(conn: sqlite3.Connection, data_dir: Path) -> None:
        order.append("m2")
        conn.execute("CREATE TABLE extra (id INTEGER PRIMARY KEY)")

    fresh = ViewerSettings(
        data_dir=tmp_path / "fresh", max_files=1000, max_extracted_bytes=10 * 1024 * 1024
    )
    monkeypatch.setattr(store_module, "MIGRATIONS", {1: _wrapper, 2: _fake2})
    created = Store(fresh)
    assert created.schema_version == 2
    assert order == ["m1", "m2"]


@pytest.mark.parametrize("layout", ["top-file", "nested-file", "dir"])
def test_stage_from_directory_rejects_symlinks(tmp_path: Path, layout: str) -> None:
    """Symlinks anywhere in a server directory are rejected."""
    store = Store(_settings(tmp_path))
    src = tmp_path / "src"
    (src / "sub" / "deep").mkdir(parents=True)
    (src / "ok.txt").write_bytes(b"ok")
    (src / "sub" / "deep" / "ok2.txt").write_bytes(b"ok2")
    target = tmp_path / "target.txt"
    target.write_bytes(b"t")
    other = tmp_path / "other"
    other.mkdir()
    (other / "f.txt").write_bytes(b"x")
    if layout == "top-file":
        os.symlink(target, src / "link.txt")
    elif layout == "nested-file":
        os.symlink(target, src / "sub" / "deep" / "link.txt")
    else:
        os.symlink(other, src / "sub" / "linkdir")
    with pytest.raises(UnsafeArchiveError) as excinfo:
        store.stage_from_directory(src, store.settings.extract_limits)
    assert excinfo.value.reason == "symlink"
    assert list(store.staging_dir.iterdir()) == []
    link_src = tmp_path / "linksrc"
    os.symlink(src, link_src)
    with pytest.raises(ValueError):
        store.stage_from_directory(link_src, store.settings.extract_limits)


def test_stage_from_directory_rejects_fifo(tmp_path: Path) -> None:
    """FIFOs in a server directory are rejected as devices."""
    store = Store(_settings(tmp_path))
    src = tmp_path / "src"
    src.mkdir()
    os.mkfifo(src / "pipe")
    with pytest.raises(UnsafeArchiveError) as excinfo:
        store.stage_from_directory(src, store.settings.extract_limits)
    assert excinfo.value.reason == "device"


def test_stage_from_directory_enforces_limits(tmp_path: Path) -> None:
    """File-count and byte caps are enforced on server directories."""
    store = Store(_settings(tmp_path))
    src = tmp_path / "src"
    _write_tree(src, {"a.txt": b"a", "b.txt": b"b", "c.txt": b"c"})
    with pytest.raises(UnsafeArchiveError) as excinfo:
        store.stage_from_directory(src, ExtractLimits(max_files=2, max_extracted_bytes=10 * 1024))
    assert excinfo.value.reason == "too_many_files"
    assert list(store.staging_dir.iterdir()) == []
    src2 = tmp_path / "src2"
    _write_tree(src2, {"big.bin": b"x" * 11})
    with pytest.raises(UnsafeArchiveError) as excinfo2:
        store.stage_from_directory(src2, ExtractLimits(max_files=1000, max_extracted_bytes=10))
    assert excinfo2.value.reason == "too_large"
    assert list(store.staging_dir.iterdir()) == []


def test_stage_from_directory_copies_and_commits(tmp_path: Path) -> None:
    """A wrapped server directory stages, digests and commits cleanly."""
    store = Store(_settings(tmp_path))
    src = tmp_path / "src"
    _write_tree(src / "mybundle", TREE)
    before = {
        path.relative_to(src).as_posix(): path.read_bytes()
        for path in sorted(src.rglob("*"))
        if path.is_file()
    }
    root = store.stage_from_directory(src, store.settings.extract_limits)
    assert root.parent.name != "" and root.name == "mybundle"
    assert compute_digest(root)[0] == compute_digest(src / "mybundle")[0]
    digest, created = store.commit(root, name="srv", members=[])
    assert created is True
    assert store.get(digest) is not None
    assert list(store.staging_dir.iterdir()) == []
    after = {
        path.relative_to(src).as_posix(): path.read_bytes()
        for path in sorted(src.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_compute_digest_matches_the_spec(tmp_path: Path) -> None:
    """Digest lines, order and insensitivity match the specification."""
    root = tmp_path / "root"
    _write_tree(root, {"a.txt": b"1", "a/c": b"22", "a-b": b"333"})
    (root / "empty").mkdir(parents=True)
    sha_b = hashlib.sha256(b"333").hexdigest()
    sha_txt = hashlib.sha256(b"1").hexdigest()
    sha_c = hashlib.sha256(b"22").hexdigest()
    text = f"a-b\t3\t{sha_b}\na.txt\t1\t{sha_txt}\na/c\t2\t{sha_c}\n"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    digest, records = compute_digest(root)
    assert digest == expected
    assert [record.relpath for record in records] == ["a-b", "a.txt", "a/c"]
    assert [record.size for record in records] == [3, 1, 2]
    assert [record.sha256 for record in records] == [sha_b, sha_txt, sha_c]
    assert all(record.digest == digest for record in records)
    target = root / "a.txt"
    os.chmod(target, 0o600)
    os.utime(target, (1, 1))
    assert compute_digest(root)[0] == digest
    os.symlink(target, root / "link")
    with pytest.raises(ValueError):
        compute_digest(root)


def test_raw_is_read_only(tmp_path: Path) -> None:
    """Committed raw files and dirs are read-only; derived stays writable."""
    store = Store(_settings(tmp_path))
    digest, _ = store.commit(_stage(store, TREE), name="n", members=[])
    raw = store.raw_dir(digest)
    for dirpath, dirnames, filenames in os.walk(raw, followlinks=False):
        assert stat.S_IMODE(os.lstat(dirpath).st_mode) == 0o555
        for name in dirnames + filenames:
            mode = stat.S_IMODE(os.lstat(os.path.join(dirpath, name)).st_mode)
            assert mode == (0o555 if name in dirnames else 0o444)
    assert store.derived_dir(digest).is_dir()
    probe = store.derived_dir(digest) / "probe.txt"
    probe.write_bytes(b"x")
    assert probe.read_bytes() == b"x"
    if os.geteuid() != 0:  # root ignores permission bits
        with pytest.raises(PermissionError):
            open(raw / "bundle.json", "w").close()


def test_commit_rejects_paths_outside_staging(tmp_path: Path) -> None:
    """Commits outside staging, of staging itself, or via symlink fail."""
    store = Store(_settings(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.txt").write_bytes(b"x")
    with pytest.raises(ValueError):
        store.commit(outside, name="n", members=[])
    with pytest.raises(ValueError):
        store.commit(store.staging_dir, name="n", members=[])
    real = _stage(store, TREE)
    link = store.staging_dir / "link"
    os.symlink(real, link)
    with pytest.raises(ValueError):
        store.commit(link, name="n", members=[])
    link.unlink()
    empty = store.new_staging()
    with pytest.raises(ValueError, match="empty bundle"):
        store.commit(empty, name="n", members=[])


def test_invalid_digest_arguments(tmp_path: Path) -> None:
    """Non-hex digests are rejected; uppercase hex is accepted."""
    store = Store(_settings(tmp_path))
    with pytest.raises(ValueError):
        store.get("../etc")
    with pytest.raises(ValueError):
        store.delete("")
    with pytest.raises(ValueError):
        store.raw_dir("g" * 64)
    assert store.get("A" * 64) is None


def test_commit_records_metadata(tmp_path: Path) -> None:
    """Commit metadata round-trips through get and list in order."""
    store = Store(_settings(tmp_path))
    members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    digest_a, _ = store.commit(
        _stage(store, {"a.txt": b"aa"}), name="alpha", members=members, validated_with="reader-1"
    )
    time.sleep(0.02)
    digest_b, _ = store.commit(_stage(store, {"b.txt": b"bbb"}), name="beta", members=[])
    record = store.get(digest_a)
    assert record is not None
    assert record.name == "alpha"
    assert record.members == tuple(members)
    assert record.total_bytes == 2 and record.file_count == 1
    assert record.status == "ready" and record.error is None
    assert record.validated_with == "reader-1"
    assert record.created_at.endswith("Z")
    listed = store.list()
    assert [item.digest for item in listed] == [digest_a, digest_b]
    assert listed[0].created_at < listed[1].created_at


def test_orphan_bundle_dir_is_replaced(tmp_path: Path) -> None:
    """A leftover directory without a row is replaced by the real commit."""
    store = Store(_settings(tmp_path))
    src = tmp_path / "src"
    _write_tree(src, TREE)
    digest = compute_digest(src)[0]
    orphan_raw = store.bundles_dir / digest / "raw"
    orphan_raw.mkdir(parents=True)
    (orphan_raw / "junk.txt").write_bytes(b"junk")
    staging = store.new_staging()
    _write_tree(staging, TREE)
    result, created = store.commit(staging, name="n", members=[])
    assert (result, created) == (digest, True)
    assert (store.raw_dir(digest) / "bundle.json").read_bytes() == TREE["bundle.json"]
    assert not (store.raw_dir(digest) / "junk.txt").exists()


def test_disk_usage(tmp_path: Path) -> None:
    """Disk usage counts raw, derived, staging and index bytes."""
    store = Store(_settings(tmp_path))
    digest, _ = store.commit(_stage(store, TREE), name="n", members=[])
    derived = store.derived_dir(digest) / "out.bin"
    derived.write_bytes(b"12345")
    usage = store.disk_usage()
    assert usage.raw_bytes == sum(len(data) for data in TREE.values())
    assert usage.derived_bytes == 5
    assert usage.staging_bytes == 0
    assert usage.index_bytes > 0
    assert usage.free_bytes > 0


def test_store_doc_lists_every_table_and_column(tmp_path: Path) -> None:
    """Every table and column is documented in docs/viewer.md."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "viewer.md"
    lines = doc.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Store format")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    section = "\n".join(lines[start:end])
    with Store(_settings(tmp_path)).connect() as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        columns: dict[str, list[str]] = {}
        for table in tables:
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            columns[table] = [row[1] for row in info]
    assert tables != []
    for table in tables:
        assert f"`{table}`" in section
        for column in columns[table]:
            assert f"`{column}`" in section, f"missing `{column}` of {table}"
    for key in ("store_schema_version", "created_by_viewer", "migrated_by_viewer"):
        assert f"`{key}`" in section


def test_staging_paths_cannot_escape(tmp_path: Path) -> None:
    """Dot-dot and symlinked staging paths never reach commit or removal."""
    store = Store(_settings(tmp_path))
    victim = tmp_path / "victim"
    _write_tree(victim, {"keep.txt": b"keep"})
    staging = _stage(store, TREE)
    digest, _ = store.commit(staging, name="n", members=[])
    escape = store.staging_dir / "x" / ".." / ".." / ".." / "victim"
    with pytest.raises(ValueError):
        store.commit(escape, name="n", members=[])
    for path in (
        store.staging_dir / "..",
        store.staging_dir / ".." / "bundles",
        store.staging_dir / "x" / ".." / "..",
        store.staging_dir,
    ):
        with pytest.raises(ValueError):
            store.discard_staging(path)
    assert (victim / "keep.txt").read_bytes() == b"keep"
    assert store.get(digest) is not None and store.raw_dir(digest).is_dir()
    other = store.new_staging()
    _write_tree(other, TREE)
    link = store.staging_dir / "link"
    os.symlink(victim, link)
    with pytest.raises(ValueError):
        store.discard_staging(link / "keep.txt")
    with pytest.raises(ValueError):
        store.commit(other / ".." / "link", name="n", members=[])
    assert (victim / "keep.txt").read_bytes() == b"keep"
    store.discard_staging(other / "dataset")
    assert not other.exists()


def test_failed_commit_leaves_no_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An index failure removes the half-created bundle directory."""
    store = Store(_settings(tmp_path))
    staging = _stage(store, TREE)
    digest = compute_digest(staging)[0]
    real_connect = store.connect

    class _Boom:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: object) -> object:
            if sql.startswith("INSERT INTO bundles"):
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name: str) -> object:
            return getattr(self._conn, name)

    from contextlib import contextmanager

    @contextmanager
    def _connect():  # type: ignore[no-untyped-def]
        with real_connect() as conn:
            yield _Boom(conn)

    monkeypatch.setattr(store, "connect", _connect)
    with pytest.raises(sqlite3.OperationalError):
        store.commit(staging, name="n", members=[])
    monkeypatch.undo()
    assert not store.bundle_dir(digest).exists()
    assert store.get(digest) is None
