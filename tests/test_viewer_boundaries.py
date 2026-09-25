"""Boundary guards: viewer modules must be registered and stay import-light."""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from test_rf_camera_boundaries import SIONNA_FREE_MODULES

HEAVY_MODULES = ("sionna", "mitsuba", "drjit", "matplotlib")


def list_package_modules(package: str) -> list[str]:
    """Return ``package`` and all its submodules (recursively) sorted."""
    module = importlib.import_module(package)
    names = [package]
    if hasattr(module, "__path__"):
        names.extend(
            info.name for info in pkgutil.walk_packages(module.__path__, prefix=package + ".")
        )
    return sorted(names)


def missing_from_registry(modules: Sequence[str], registry: Sequence[str]) -> list[str]:
    """Return the modules not listed in ``registry``, sorted."""
    known = set(registry)
    return sorted(name for name in modules if name not in known)


def heavy_modules_loaded(modules: Sequence[str], extra_path: Sequence[Path] = ()) -> list[str]:
    """Import ``modules`` in a fresh interpreter; return the HEAVY_MODULES that get loaded."""
    code = (
        "import importlib, sys\n"
        f"for name in {list(modules)!r}:\n"
        "    importlib.import_module(name)\n"
        "loaded = sorted(m for m in "
        f"{list(HEAVY_MODULES)!r} if m in sys.modules)\n"
        "print(','.join(loaded))\n"
    )
    env = dict(os.environ)
    prefix = os.pathsep.join(str(path) for path in extra_path)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = prefix if existing is None else prefix + os.pathsep + existing
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    output = result.stdout.strip()
    return output.split(",") if output else []


def _drop_modules(prefix: str) -> None:
    """Remove a fake package and its submodules from ``sys.modules``."""
    for name in [name for name in sys.modules if name == prefix or name.startswith(prefix + ".")]:
        del sys.modules[name]


def _write_package(path: Path, modules: Sequence[str]) -> None:
    """Create ``path/__init__.py`` and the given relative module files."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").write_text("", encoding="utf-8")
    for relative in modules:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")


def test_every_viewer_module_is_registered_sionna_free() -> None:
    modules = list_package_modules("plateau_rt.viewer")
    assert "plateau_rt.viewer.extract" in modules
    assert "plateau_rt.viewer.settings" in modules
    missing = missing_from_registry(modules, SIONNA_FREE_MODULES)
    assert missing == [], (
        f"add {missing!r} to SIONNA_FREE_MODULES in tests/test_rf_camera_boundaries.py"
    )


def test_viewer_modules_do_not_import_heavy_packages() -> None:
    assert heavy_modules_loaded(list_package_modules("plateau_rt.viewer")) == []


def test_enumeration_finds_nested_modules_and_detects_unregistered(
    tmp_path: Path, monkeypatch, request
) -> None:
    _write_package(tmp_path / "fakeviewer", ["a.py", "sub/__init__.py", "sub/b.py"])
    monkeypatch.syspath_prepend(str(tmp_path))
    request.addfinalizer(lambda: _drop_modules("fakeviewer"))
    modules = list_package_modules("fakeviewer")
    assert modules == ["fakeviewer", "fakeviewer.a", "fakeviewer.sub", "fakeviewer.sub.b"]
    assert missing_from_registry(modules, ["fakeviewer", "fakeviewer.a", "fakeviewer.sub"]) == [
        "fakeviewer.sub.b"
    ]


def test_heavy_import_detection_can_fail(tmp_path: Path, monkeypatch, request) -> None:
    _write_package(tmp_path / "fakeheavy", ["mod.py"])
    (tmp_path / "fakeheavy" / "mod.py").write_text("import matplotlib\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    request.addfinalizer(lambda: _drop_modules("fakeheavy"))
    assert heavy_modules_loaded(["fakeheavy.mod"], extra_path=[tmp_path]) == ["matplotlib"]
