"""Tests for determinism and golden helpers (``plateau_rt.viewer.testing``)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from viewer_bundle_fixtures import write_fixture_bundle

from plateau_rt.viewer.derive import (
    DeriverSpec,
    ParamSpec,
    registered_derivers,
    unregister,
)
from plateau_rt.viewer.testing import (
    GoldenMismatch,
    assert_deterministic,
    bundle_members,
    check_all_goldens,
    golden_check,
    representative_cases,
)

GOLDEN_PATH = Path(__file__).parent / "viewer_golden" / "derivers.json"


@pytest.fixture(autouse=True)
def _empty_registry() -> Any:
    """Leave the global deriver registry empty after every test."""
    yield
    for deriver in list(registered_derivers()):
        unregister(deriver.spec.name)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the v3 and v2 fixture bundles lazily."""
    base = tmp_path_factory.mktemp("golden-bundles")
    version3 = base / "v3"
    version2 = base / "v2"
    write_fixture_bundle(version3, schema_version=3)
    write_fixture_bundle(version2, schema_version=2)
    return {"v3": version3, "v2": version2}


class SceneDeriver:
    """A toy deriver over a ``scene`` member returning tags and mutable salt bytes."""

    def __init__(self, salt: bytes = b"salt", version: int = 1) -> None:
        """Configure the salt (output bytes) and spec version."""
        self.salt = salt
        self.spec = DeriverSpec("tags", version, ("scene",))

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """The scene deriver has no space parameters."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the sorted XML tags and the configured salt bytes."""
        root = ctx.parse_xml(ctx.member_path())
        return {"tags.json": sorted(element.tag for element in root.iter()), "salt.bin": self.salt}


class RandomDeriver:
    """A deliberately non-deterministic deriver."""

    spec = DeriverSpec("random", 1, ("rf_dataset",))

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """No parameters."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return random bytes so two runs differ."""
        return {"x.bin": os.urandom(8)}


class FixedDeriver:
    """A deterministic deriver returning fixed bytes."""

    spec = DeriverSpec("fixed", 1, ("rf_dataset",))

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """No parameters."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return fixed bytes."""
        return {"x.bin": b"abc"}


class ViewDeriver:
    """A deriver with a view space and an int range parameter."""

    spec = DeriverSpec(
        "views",
        1,
        ("rf_dataset",),
        (ParamSpec("view", "view"), ParamSpec("n", "int", min=0, max=5)),
    )

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Two views."""
        return [{"view": "a"}, {"view": "b"}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a small deterministic output."""
        return {"x.json": dict(params)}


def synth_bundle(root: Path) -> Path:
    """Write a minimal rf_dataset bundle root and return it."""
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_format_version": 1,
                "members": [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_registered_derivers_match_golden(
    request: pytest.FixtureRequest, bundles: dict[str, Path]
) -> None:
    check_all_goldens(
        bundles,
        golden_path=GOLDEN_PATH,
        update=request.config.getoption("--update-viewer-golden"),
    )
    assert json.loads(GOLDEN_PATH.read_text(encoding="utf-8")) == {}


def test_golden_scene_update_and_mismatch(tmp_path: Path, bundles: dict[str, Path]) -> None:
    golden = tmp_path / "golden.json"
    deriver = SceneDeriver()
    golden_check(deriver, bundles["v3"], {}, case="v3", golden_path=golden, update=True)
    assert golden.is_file()
    assert golden_check(deriver, bundles["v3"], {}, case="v3", golden_path=golden)["salt.bin"]

    changed = SceneDeriver(b"salt-two")
    with pytest.raises(GoldenMismatch) as excinfo:
        golden_check(changed, bundles["v3"], {}, case="v3", golden_path=golden)
    message = str(excinfo.value)
    assert "changed but its version is still v1" in message
    assert "bump DeriverSpec.version" in message

    bumped = SceneDeriver(b"salt-two", version=2)
    with pytest.raises(GoldenMismatch) as version_info:
        golden_check(bumped, bundles["v3"], {}, case="v3", golden_path=golden)
    assert "update the golden" in str(version_info.value)
    golden_check(bumped, bundles["v3"], {}, case="v3", golden_path=golden, update=True)
    payload = json.loads(golden.read_text(encoding="utf-8"))
    assert payload["tags"]["version"] == 2


def test_check_all_goldens_failures(tmp_path: Path, bundles: dict[str, Path]) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(GoldenMismatch) as excinfo:
        check_all_goldens(
            {"v3": bundles["v3"]},
            golden_path=missing,
            derivers=[SceneDeriver()],
        )
    assert "no golden for deriver" in str(excinfo.value)

    class RunDeriver:
        spec = DeriverSpec("runr", 1, ("tomo_run",))

        def param_space(self, ctx: Any) -> list[dict[str, str]]:
            return [{}]

        def derive(self, ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
            return {"x.json": {}}

    with pytest.raises(GoldenMismatch) as run_info:
        check_all_goldens(
            {"v3": bundles["v3"]},
            golden_path=tmp_path / "run.json",
            derivers=[RunDeriver()],
        )
    assert "has no representative case" in str(run_info.value)

    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"ghost": {"version": 1, "cases": {}}}) + "\n", encoding="utf-8")
    with pytest.raises(GoldenMismatch) as stale_info:
        check_all_goldens({}, golden_path=stale, derivers=[])
    assert "stale golden entry" in str(stale_info.value)


def test_assert_deterministic(tmp_path: Path) -> None:
    root = synth_bundle(tmp_path / "bundle")
    with pytest.raises(AssertionError):
        assert_deterministic(RandomDeriver(), root, {})
    shas = assert_deterministic(FixedDeriver(), root, {})
    assert shas == {"x.bin": hashlib.sha256(b"abc").hexdigest()}


def test_representative_cases(tmp_path: Path) -> None:
    root = synth_bundle(tmp_path / "bundle")
    cases = representative_cases(ViewDeriver(), root)
    assert cases == [
        ("dataset", {"view": "a", "n": "0"}),
        ("dataset", {"view": "b", "n": "5"}),
    ]


def test_bundle_members_fallback(tmp_path: Path) -> None:
    root = tmp_path / "fallback"
    root.mkdir()
    (root / "run_manifest.json").write_text("{}", encoding="utf-8")
    assert bundle_members(root) == [{"id": "run", "kind": "tomo_run", "path": "."}]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        bundle_members(empty)


def test_update_golden_keeps_only_checked_derivers(tmp_path: Path) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({"ghost": {"version": 1, "cases": {}}, "other": {"version": 1, "cases": {}}})
        + "\n",
        encoding="utf-8",
    )
    root = synth_bundle(tmp_path / "bundle")
    check_all_goldens({"s": root}, golden_path=golden, derivers=[FixedDeriver()], update=True)
    assert set(json.loads(golden.read_text(encoding="utf-8"))) == {"fixed"}
