"""Test derivers used by the viewer job tests (importable by spawn children)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.viewer.derive import DeriverSpec, ParamSpec

MARKS_ENV = "VIEWER_TEST_MARKS"


def marks_directory() -> Path | None:
    """Return the directory named by ``VIEWER_TEST_MARKS``, or None when unset."""
    raw = os.environ.get(MARKS_ENV)
    return Path(raw) if raw else None


def read_marks(directory: Path) -> list[dict[str, Any]]:
    """Parse every ``*.json`` mark below ``directory``, sorted by ``start``."""
    marks: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            marks.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    marks.sort(key=lambda mark: mark.get("start", 0.0))
    return marks


def _mark_start(deriver: str, tag: int) -> Path | None:
    """Write the start marker and return its path."""
    directory = marks_directory()
    if directory is None:
        return None
    path = directory / f"{deriver}-{tag}-{os.getpid()}.json"
    path.write_text(
        json.dumps(
            {"deriver": deriver, "tag": tag, "pid": os.getpid(), "start": time.time(), "end": None}
        ),
        encoding="utf-8",
    )
    return path


def _mark_end(path: Path | None) -> None:
    """Rewrite a start marker with its end time."""
    if path is None:
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["end"] = time.time()
    path.write_text(json.dumps(data), encoding="utf-8")


class SlowDeriver:
    """A deriver that sleeps for ``ms`` milliseconds and writes a mark at start and end."""

    spec = DeriverSpec(
        "tslow",
        1,
        ("rf_dataset",),
        params=(
            ParamSpec("ms", "int", min=0, max=600000),
            ParamSpec("tag", "int", min=0, max=1000),
        ),
    )

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return the single empty parameter combination."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Mark, sleep and return a JSON output carrying the tag."""
        tag = int(params["tag"])
        path = _mark_start("tslow", tag)
        try:
            time.sleep(int(params["ms"]) / 1000.0)
        finally:
            _mark_end(path)
        return {"out.json": {"tag": tag}}


class EagerSlowDeriver:
    """The eager variant of :class:`SlowDeriver` with a fixed 300 ms sleep."""

    spec = DeriverSpec("teagerslow", 1, ("rf_dataset",), eager=True)

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return the single empty parameter combination."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Mark for 300 ms and return a JSON output."""
        path = _mark_start("teagerslow", -1)
        try:
            time.sleep(0.3)
        finally:
            _mark_end(path)
        return {"out.json": {"tag": -1}}


class MemDeriver:
    """A deriver that allocates ``mib`` mebibytes before returning."""

    spec = DeriverSpec(
        "tmem", 1, ("rf_dataset",), params=(ParamSpec("mib", "int", min=1, max=65536),)
    )

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return the single empty parameter combination."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Allocate the requested buffer and return a small JSON summary."""
        array = np.ones(int(params["mib"]) << 20, dtype=np.uint8)
        return {"out.json": {"sum": int(array[:16].sum())}}


class FlakyDeriver:
    """A deriver that fails while a sentinel file exists."""

    spec = DeriverSpec("tflaky", 1, ("rf_dataset",))

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return the single empty parameter combination."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Raise while ``<marks>/flaky-fail`` exists, else succeed."""
        directory = marks_directory()
        if directory is not None and (directory / "flaky-fail").exists():
            raise RuntimeError("flaky failure")
        return {"out.json": {"ok": True}}


class EnumDeriver:
    """A deriver with one enum space parameter."""

    spec = DeriverSpec(
        "tenum", 1, ("rf_dataset",), params=(ParamSpec("mode", "enum", values=("a", "b")),)
    )

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return both enum values."""
        return [{"mode": "a"}, {"mode": "b"}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return the selected mode."""
        return {"out.json": {"mode": params["mode"]}}


class Versioned:
    """A test deriver whose outputs depend on its spec version."""

    def __init__(self, version: int) -> None:
        """Build the spec for ``version`` with one enum parameter."""
        self.spec = DeriverSpec(
            "tversion",
            version,
            ("rf_dataset",),
            params=(ParamSpec("mode", "enum", values=("a/b", "c d")),),
        )

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return both enum values."""
        return [{"mode": "a/b"}, {"mode": "c d"}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return a JSON file and a binary blob tagged with the version."""
        return {
            "out.json": {"mode": params["mode"], "version": self.spec.version},
            "blob.bin": bytes([self.spec.version]) * 16,
        }


class Boom:
    """A test deriver that always fails."""

    spec = DeriverSpec("tboom", 1, ("rf_dataset",))

    def param_space(self, ctx: Any) -> list[dict[str, str]]:
        """Return the single empty parameter combination."""
        return [{}]

    def derive(self, ctx: Any, params: Mapping[str, Any]) -> dict[str, Any]:
        """Raise a runtime error."""
        raise RuntimeError("boom")
