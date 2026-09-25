"""Tests for the deriver registry, parameter validation and versioned cache."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plateau_rt.viewer.derive import (
    BadParams,
    DeriveError,
    DeriveNotFound,
    DeriverSpec,
    LinkTarget,
    ParamSpec,
    derive_eager,
    get_deriver,
    get_or_derive,
    links_key,
    open_context,
    params_key,
    register,
    registered,
    registered_derivers,
    unregister,
)
from plateau_rt.viewer.safeio import UnsafeArrayError, UnsafePathError
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store

VALID = {"view": "v0", "freq_bin": "3", "threshold_db": "-30", "mode": "mag"}


@pytest.fixture(autouse=True)
def _empty_registry() -> Any:
    """Leave the global deriver registry empty after every test."""
    yield
    for deriver in list(registered_derivers()):
        unregister(deriver.spec.name)


class ToyDeriver:
    """Small configurable deriver used across these tests."""

    def __init__(
        self,
        name: str = "toy",
        version: int = 1,
        kinds: tuple[str, ...] = ("rf_dataset",),
        params: tuple[ParamSpec, ...] = (),
        *,
        eager: bool = False,
        space: Any = None,
        make: Any = None,
    ) -> None:
        """Build a toy with the given spec and behaviour."""
        self.spec = DeriverSpec(name, version, kinds, params, eager=eager)
        self._space = space
        self._make = make
        self.calls = 0

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return the configured space (a callable may use ``ctx``)."""
        if callable(self._space):
            return self._space(ctx)
        return [{}] if self._space is None else [dict(entry) for entry in self._space]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call the configured ``make`` or return the typed params as JSON."""
        self.calls += 1
        if self._make is None:
            return {"value.json": dict(params)}
        return self._make(ctx, params)


def open_store(tmp_path: Path, **kwargs: Any) -> Store:
    """Open a store under ``tmp_path`` with optional settings overrides."""
    return Store(ViewerSettings(data_dir=tmp_path / "store", **kwargs))


def write_file(path: Path, data: Any) -> None:
    """Write bytes, a JSON mapping/list or a numpy array at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, np.ndarray):
        np.save(path, data)
    elif isinstance(data, (dict, list)):
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_bytes(data)


def raise_derive_error(ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
    """A deriver body that always fails."""
    raise DeriveError("boom")


def make_bundle(
    tmp_path: Path,
    members: Sequence[Mapping[str, Any]],
    files: Mapping[str, Any],
    *,
    name: str = "bundle",
) -> Path:
    """Write a bundle root with ``bundle.json`` and member files."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle.json").write_text(
        json.dumps({"bundle_format_version": 1, "members": [dict(m) for m in members]}) + "\n",
        encoding="utf-8",
    )
    for relative, data in files.items():
        write_file(root / relative, data)
    return root


def commit_bundle(store: Store, root: Path, members: Sequence[Mapping[str, Any]]) -> str:
    """Stage and commit ``root``; return its digest."""
    staged = store.stage_from_directory(root, store.settings.extract_limits)
    digest, _ = store.commit(staged, name="t", members=[dict(m) for m in members])
    return digest


def toy_params() -> tuple[ParamSpec, ...]:
    """Return the four parameters of the cache toy deriver."""
    return (
        ParamSpec("view", "view"),
        ParamSpec("freq_bin", "int", min=0, max=15),
        ParamSpec("threshold_db", "float", min=-60, max=0, step=1),
        ParamSpec("mode", "enum", values=("mag", "phase")),
    )


def toy_space(ctx: Any) -> list[dict[str, str]]:
    """Return the view/mode combinations declared in the member's ``views.json``."""
    views = json.loads(ctx.read_bytes(ctx.member_path("views.json")))
    return [{"view": view, "mode": mode} for view in views for mode in ("mag", "phase")]


def toy_make(ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one npy, one json and one bytes output depending on the params."""
    return {
        "image.npy": np.full((2, 2), params["freq_bin"], dtype=np.float32),
        "info.json": {
            "mode": params["mode"],
            "threshold_db": params["threshold_db"],
            "view": params["view"],
        },
        "note.txt": b"note",
    }


def cache_fixture(tmp_path: Path) -> tuple[Store, str, ToyDeriver]:
    """Open a store with one rf_dataset member declaring views v0 and v1."""
    store = open_store(tmp_path)
    members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    root = make_bundle(
        tmp_path,
        members,
        {
            "dataset/views.json": ["v0", "v1"],
            "dataset/dataset_manifest.json": {"schema_version": 3},
        },
    )
    digest = commit_bundle(store, root, members)
    deriver = ToyDeriver(params=toy_params(), space=toy_space, make=toy_make)
    return store, digest, deriver


def test_param_spec_validation() -> None:
    with pytest.raises(ValueError):
        ParamSpec("x", "float", min=0.0, max=1.0, step=0.3)
    with pytest.raises(ValueError):
        ParamSpec("x", "int", min=True, max=2)
    with pytest.raises(ValueError):
        ParamSpec("x", "enum")
    with pytest.raises(ValueError):
        ParamSpec("x", "bogus")
    with pytest.raises(ValueError):
        ParamSpec("Bad", "int", min=0, max=1)


def test_deriver_spec_validation() -> None:
    with pytest.raises(ValueError):
        DeriverSpec("toy", 0, ("rf_dataset",))
    with pytest.raises(ValueError):
        DeriverSpec("toy", 1, ("bogus",))
    with pytest.raises(ValueError):
        DeriverSpec("toy", 1, ("rf_dataset",), (ParamSpec("a", "int", min=0, max=1),) * 2)
    with pytest.raises(ValueError):
        DeriverSpec(
            "toy",
            1,
            ("rf_dataset",),
            (ParamSpec("a", "int", min=0, max=1),),
            eager=True,
        )


def test_register_replace_and_registered_restore() -> None:
    first = ToyDeriver()
    second = ToyDeriver()
    register(first)
    try:
        with pytest.raises(ValueError):
            register(second)
        assert get_deriver("toy") is first
        register(second, replace=True)
        assert get_deriver("toy") is second
    finally:
        unregister("toy")
    with registered(first):
        assert get_deriver("toy") is first
    assert registered_derivers() == ()
    with pytest.raises(KeyError):
        unregister("toy")


def test_ac1_cache_hit_and_layout(tmp_path: Path) -> None:
    store, digest, deriver = cache_fixture(tmp_path)
    with registered(deriver):
        result = get_or_derive(store, digest, "dataset", "toy", dict(VALID))
        assert result.cached is False
        assert deriver.calls == 1
        expected = (
            store.derived_dir(digest)
            / "dataset"
            / "toy"
            / "v1"
            / "freq_bin=3,mode=mag,threshold_db=-30,view=v0"
            / "nolink"
        )
        assert result.directory == expected
        assert {record.name for record in result.files} == {
            "image.npy",
            "info.json",
            "note.txt",
        }
        for record in result.files:
            data = (result.directory / record.name).read_bytes()
            assert record.size == len(data)
            assert record.sha256 == hashlib.sha256(data).hexdigest()
        again = get_or_derive(
            store,
            digest,
            "dataset",
            "toy",
            {"view": "v0", "freq_bin": "+3", "threshold_db": "-30.0", "mode": "mag"},
        )
        assert again.cached is True
        assert again.directory == result.directory
        assert deriver.calls == 1
    assert registered_derivers() == ()
    with store.connect() as conn:
        row = conn.execute(
            "SELECT status FROM derived WHERE digest = ? AND member = ? AND deriver = ?",
            (digest, "dataset", "toy"),
        ).fetchone()
    assert row is not None
    assert row["status"] == "ready"


def test_ac1_version_bump(tmp_path: Path) -> None:
    store, digest, first = cache_fixture(tmp_path)
    with registered(first):
        old = get_or_derive(store, digest, "dataset", "toy", dict(VALID))
    second = ToyDeriver(version=2, params=toy_params(), space=toy_space, make=toy_make)
    with registered(second):
        new = get_or_derive(store, digest, "dataset", "toy", dict(VALID))
        assert new.cached is False
        assert new.directory.parent.parent.name == "v2"
        assert new.directory != old.directory
        assert second.calls == 1
        cached = get_or_derive(store, digest, "dataset", "toy", dict(VALID))
        assert cached.cached is True
        assert second.calls == 1


def test_ac1_complex_output_rejected(tmp_path: Path) -> None:
    store, digest, _ = cache_fixture(tmp_path)
    deriver = ToyDeriver(
        name="cmplx",
        make=lambda ctx, params: {"x.npy": np.zeros(4, dtype=np.complex64)},
    )
    with registered(deriver):
        with pytest.raises(DeriveError) as excinfo:
            get_or_derive(store, digest, "dataset", "cmplx", {})
        assert "complex" in str(excinfo.value)
    assert not (store.derived_dir(digest) / "dataset" / "cmplx").exists()
    assert list(store.staging_dir.iterdir()) == []
    with store.connect() as conn:
        row = conn.execute(
            "SELECT status, error FROM derived WHERE deriver = ?", ("cmplx",)
        ).fetchone()
    assert row is not None
    assert row["status"] == "failed"
    assert row["error"]


def test_output_rules(tmp_path: Path) -> None:
    store, digest, _ = cache_fixture(tmp_path)
    with registered(ToyDeriver(name="f64", make=lambda c, p: {"x.npy": np.zeros(2)})):
        with pytest.raises(DeriveError):
            get_or_derive(store, digest, "dataset", "f64", {})
    with registered(ToyDeriver(name="npyjson", make=lambda c, p: {"x.npy": {"a": 1}})):
        with pytest.raises(DeriveError):
            get_or_derive(store, digest, "dataset", "npyjson", {})
    with registered(
        ToyDeriver(name="nan", make=lambda c, p: {"x.json": {"b": float("nan"), "a": 1}})
    ):
        result = get_or_derive(store, digest, "dataset", "nan", {})
    assert (result.directory / "x.json").read_bytes() == b'{"a":1,"b":null}'
    with registered(
        ToyDeriver(name="be", make=lambda c, p: {"x.npy": np.array([1.5, -2.5], dtype=">f4")})
    ):
        result = get_or_derive(store, digest, "dataset", "be", {})
    loaded = np.load(result.directory / "x.npy")
    assert loaded.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(loaded, np.array([1.5, -2.5], dtype=np.float32))
    with registered(ToyDeriver(name="raw", make=lambda c, p: {"x.bin": b"\x00\xff"})):
        result = get_or_derive(store, digest, "dataset", "raw", {})
    assert (result.directory / "x.bin").read_bytes() == b"\x00\xff"


@pytest.mark.parametrize(
    "params",
    [
        dict(VALID, freq_bin="16"),
        dict(VALID, freq_bin="-1"),
        dict(VALID, freq_bin="1.5"),
        dict(VALID, freq_bin="x"),
        dict(VALID, threshold_db="-30.5"),
        dict(VALID, threshold_db="1"),
        dict(VALID, threshold_db="-61"),
        dict(VALID, threshold_db="nan"),
        dict(VALID, threshold_db="inf"),
        dict(VALID, threshold_db=" -3"),
        dict(VALID, view="v9"),
        dict(VALID, mode="abs"),
        {**VALID, "foo": "1"},
        {key: value for key, value in VALID.items() if key != "mode"},
    ],
)
def test_ac2_bad_params(tmp_path: Path, params: dict[str, str]) -> None:
    store, digest, deriver = cache_fixture(tmp_path)
    with registered(deriver):
        with pytest.raises(BadParams):
            get_or_derive(store, digest, "dataset", "toy", params)
    assert deriver.calls == 0
    assert list(store.derived_dir(digest).iterdir()) == []
    with store.connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM derived").fetchone()["n"]
    assert count == 0


def test_bad_params_from_derive(tmp_path: Path) -> None:
    store, digest, _ = cache_fixture(tmp_path)

    def make(ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        raise BadParams("data-dependent limit")

    deriver = ToyDeriver(name="dp", make=make)
    with registered(deriver):
        with pytest.raises(BadParams):
            get_or_derive(store, digest, "dataset", "dp", {})
    assert list(store.derived_dir(digest).iterdir()) == []
    with store.connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM derived").fetchone()["n"]
    assert count == 0


def test_params_key_and_links_key() -> None:
    assert params_key({}) == "noparams"
    assert params_key({"a": "x/y,z=w"}) == "a=x%2Fy%2Cz%3Dw"
    with pytest.raises(BadParams):
        params_key({"a": "x" * 300})
    assert links_key([]) == "nolink"
    assert links_key(["none"]) == "nolink"
    digest = "a" * 64
    assert links_key([digest]) == digest
    assert links_key(["none", digest]) == "none_" + digest
    many = ["b" * 64, "c" * 64, "d" * 64, "e" * 64]
    hashed = links_key(many)
    assert hashed.startswith("links-")
    assert len(hashed) == len("links-") + 64


def test_ac3_link_resolution(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    content = b"CONTENT"
    source_sha = hashlib.sha256(content).hexdigest()
    a_members = [{"id": "run", "kind": "tomo_run", "path": "run"}]
    a_root = make_bundle(
        tmp_path,
        a_members,
        {"run/run.json": {"source_sha256": source_sha}},
        name="bundle_a",
    )
    digest_a = commit_bundle(store, a_root, a_members)
    b_members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    b_root = make_bundle(
        tmp_path,
        b_members,
        {"dataset/data.bin": content, "dataset/dataset_manifest.json": {"schema_version": 3}},
        name="bundle_b",
    )

    def make(ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        config = json.loads(ctx.read_bytes(ctx.member_path("run.json")))
        linked = ctx.link(LinkTarget(sha256=config["source_sha256"], kind="rf_dataset"))
        if linked is None:
            return {"link.json": {"digest": None}}
        assert ctx.read_bytes(linked.path(linked.relpath)) == content
        return {"link.json": {"digest": linked.digest}}

    deriver = ToyDeriver(name="linker", kinds=("tomo_run",), make=make)
    with registered(deriver):
        first = get_or_derive(store, digest_a, "run", "linker", {})
        assert first.links_key == "nolink"
        assert json.loads((first.directory / "link.json").read_text())["digest"] is None
        assert deriver.calls == 1

        digest_b = commit_bundle(store, b_root, b_members)
        second = get_or_derive(store, digest_a, "run", "linker", {})
        assert second.links_key == digest_b
        assert second.cached is False
        assert json.loads((second.directory / "link.json").read_text())["digest"] == digest_b
        assert first.directory.exists() and second.directory.exists()
        assert deriver.calls == 2

        third = get_or_derive(store, digest_a, "run", "linker", {})
        assert third.cached is True
        assert third.links_key == digest_b
        assert deriver.calls == 2

        store.delete(digest_b)
        fourth = get_or_derive(store, digest_a, "run", "linker", {})
        assert fourth.links_key == "nolink"
        assert fourth.cached is True
        assert json.loads((fourth.directory / "link.json").read_text())["digest"] is None
        assert deriver.calls == 2


def test_link_same_bundle_self(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    root = make_bundle(
        tmp_path,
        members,
        {"dataset/dataset_manifest.json": {"schema_version": 3}},
    )
    digest = commit_bundle(store, root, members)

    def make(ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        linked = ctx.link(LinkTarget(member="dataset"))
        assert linked is not None
        return {"x.json": {"same": linked.same_bundle, "token": ctx.links[0][1]}}

    deriver = ToyDeriver(name="selflink", make=make)
    with registered(deriver):
        result = get_or_derive(store, digest, "dataset", "selflink", {})
    assert result.links_key == "self"
    payload = json.loads((result.directory / "x.json").read_text())
    assert payload == {"same": True, "token": "self"}


def test_link_prefers_earlier_bundle_and_kind_filter(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    content = b"SHARED"
    sha = hashlib.sha256(content).hexdigest()
    run_members = [{"id": "run", "kind": "tomo_run", "path": "run"}]
    run_root = make_bundle(
        tmp_path,
        run_members,
        {"run/run.json": {"source_sha256": sha}},
        name="run_bundle",
    )
    digest_run = commit_bundle(store, run_root, run_members)

    first_members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    first_root = make_bundle(
        tmp_path,
        first_members,
        {
            "dataset/shared.bin": content,
            "dataset/extra.txt": b"first",
            "dataset/dataset_manifest.json": {},
        },
        name="first",
    )
    digest_first = commit_bundle(store, first_root, first_members)
    time.sleep(0.01)
    second_members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    second_root = make_bundle(
        tmp_path,
        second_members,
        {
            "dataset/shared.bin": content,
            "dataset/extra.txt": b"second",
            "dataset/dataset_manifest.json": {},
        },
        name="second",
    )
    digest_second = commit_bundle(store, second_root, second_members)
    assert len({digest_run, digest_first, digest_second}) == 3

    def make(ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        config = json.loads(ctx.read_bytes(ctx.member_path("run.json")))
        linked = ctx.link(LinkTarget(sha256=config["source_sha256"], kind="rf_dataset"))
        assert linked is not None
        return {"x.json": {"digest": linked.digest, "member": linked.member}}

    deriver = ToyDeriver(name="picker", kinds=("tomo_run",), make=make)
    with registered(deriver):
        result = get_or_derive(store, digest_run, "run", "picker", {})
    payload = json.loads((result.directory / "x.json").read_text())
    assert payload["digest"] == digest_first
    assert payload["digest"] != digest_second
    assert payload["member"] == "dataset"

    mixed_members = [
        {"id": "dataset", "kind": "rf_dataset", "path": "."},
        {"id": "run", "kind": "tomo_run", "path": "."},
    ]
    mixed_root = make_bundle(
        tmp_path,
        mixed_members,
        {"shared.bin": content, "dataset_manifest.json": {}, "run_manifest.json": {}},
        name="mixed",
    )
    commit_bundle(store, mixed_root, mixed_members)

    def filtered(ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        config = json.loads(ctx.read_bytes(ctx.member_path("run.json")))
        linked = ctx.link(LinkTarget(sha256=config["source_sha256"], kind="tomo_run"))
        assert linked is not None
        return {"x.json": {"member": linked.member}}

    filtered_deriver = ToyDeriver(name="filtered", kinds=("tomo_run",), make=filtered)
    with registered(filtered_deriver):
        result = get_or_derive(store, digest_run, "run", "filtered", {})
    assert json.loads((result.directory / "x.json").read_text())["member"] == "run"


def test_loader_containment_and_max_bytes(tmp_path: Path) -> None:
    store = open_store(tmp_path, max_array_bytes=32)
    members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    array = np.arange(64, dtype=np.float32)
    root = make_bundle(
        tmp_path,
        members,
        {
            "dataset/dataset_manifest.json": {},
            "dataset/big.npy": array,
        },
    )
    digest = commit_bundle(store, root, members)
    ctx = open_context(store, digest, "dataset")
    with pytest.raises(UnsafePathError):
        ctx.load_npy("../x.npy")
    with pytest.raises(UnsafePathError):
        ctx.load_npy(tmp_path / "outside.npy")
    with pytest.raises(UnsafeArrayError):
        ctx.load_npy("dataset/big.npy")


def test_derive_eager(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    members = [
        {"id": "d1", "kind": "rf_dataset", "path": "d1"},
        {"id": "d2", "kind": "rf_dataset", "path": "d2"},
    ]
    root = make_bundle(
        tmp_path,
        members,
        {
            "d1/dataset_manifest.json": {},
            "d2/dataset_manifest.json": {},
        },
    )
    digest = commit_bundle(store, root, members)
    good = ToyDeriver(
        name="egood",
        params=(ParamSpec("v", "enum", values=("a",)),),
        eager=True,
        space=[{"v": "a"}],
    )
    bad = ToyDeriver(name="ebad", eager=True, make=raise_derive_error)
    lazy = ToyDeriver(name="lazy")
    with registered(good, bad, lazy):
        outcomes = derive_eager(store, digest)
    assert [(o.deriver, o.member) for o in outcomes] == [
        ("ebad", "d1"),
        ("ebad", "d2"),
        ("egood", "d1"),
        ("egood", "d2"),
    ]
    assert all(o.error for o in outcomes if o.deriver == "ebad")
    assert all(o.result is not None and o.error is None for o in outcomes if o.deriver == "egood")
    assert lazy.calls == 0
    with store.connect() as conn:
        failed = conn.execute(
            "SELECT COUNT(*) AS n FROM derived WHERE deriver = ? AND status = ?",
            ("ebad", "failed"),
        ).fetchone()["n"]
    assert failed == 2


def test_not_found(tmp_path: Path) -> None:
    store, digest, _ = cache_fixture(tmp_path)
    scene = ToyDeriver(name="scenery", kinds=("scene",))
    with registered(scene):
        with pytest.raises(DeriveNotFound):
            get_or_derive(store, digest, "dataset", "scenery", {})
        with pytest.raises(DeriveNotFound):
            get_or_derive(store, digest, "dataset", "missing", {})
        with pytest.raises(DeriveNotFound):
            open_context(store, "0" * 64, "dataset")
        with pytest.raises(DeriveNotFound):
            open_context(store, digest, "missing")
        with pytest.raises(DeriveNotFound):
            open_context(store, "not-a-digest", "dataset")
        with pytest.raises(DeriveNotFound):
            derive_eager(store, "not-a-digest")


def test_param_space_failure_is_derive_error(tmp_path: Path) -> None:
    store, digest, _ = cache_fixture(tmp_path)

    def broken_space(ctx: Any) -> list[dict[str, str]]:
        raise OSError("manifest unreadable")

    deriver = ToyDeriver(name="broken", space=broken_space)
    with registered(deriver):
        with pytest.raises(DeriveError) as excinfo:
            get_or_derive(store, digest, "dataset", "broken", {})
    assert "manifest unreadable" in str(excinfo.value)
    assert deriver.calls == 0
    assert list(store.derived_dir(digest).iterdir()) == []
