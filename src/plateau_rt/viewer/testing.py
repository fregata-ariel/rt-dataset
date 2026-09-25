"""Determinism and golden helpers for derivers (used by the test suite).

This module intentionally does not import ``pytest``: it is plain library code that the test
suite (and later tooling) calls. Golden files are keyed by deriver name and version.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from plateau_rt.viewer.derive import (
    RANGE_KINDS,
    DerivedResult,
    Deriver,
    get_or_derive,
    open_context,
    registered,
    registered_derivers,
)
from plateau_rt.viewer.kinds import detect_members
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store

DEFAULT_GOLDEN_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "viewer_golden" / "derivers.json"
)


class GoldenMismatch(AssertionError):
    """A deriver's outputs do not match its recorded golden."""


def bundle_members(root: Path) -> list[dict[str, Any]]:
    """Return the members of a bundle root, from ``bundle.json`` or fallback detection."""
    return [member.to_dict() for member in detect_members(Path(root))]


def _matching_member(members: Sequence[Mapping[str, Any]], deriver: Deriver) -> str:
    """Return the single member id whose kind the deriver supports."""
    matching = [str(m["id"]) for m in members if m.get("kind") in deriver.spec.kinds]
    if len(matching) != 1:
        raise ValueError(
            f"expected one member for deriver {deriver.spec.name!r}, found {matching!r}"
        )
    return matching[0]


def derive_in_temp_store(
    deriver: Deriver,
    bundle_dir: Path,
    params: Mapping[str, str],
    *,
    member: str | None = None,
    members: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[DerivedResult, dict[str, bytes]]:
    """Commit ``bundle_dir`` into a temp store, derive, and return result and file bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(ViewerSettings(data_dir=Path(tmp) / "store"))
        root = store.stage_from_directory(Path(bundle_dir), store.settings.extract_limits)
        member_list = list(members) if members is not None else bundle_members(root)
        digest, _ = store.commit(root, name="test", members=member_list)
        if member is None:
            member = _matching_member(member_list, deriver)
        with registered(deriver):
            result = get_or_derive(store, digest, member, deriver.spec.name, params)
        data = {
            record.name: (result.directory / record.name).read_bytes() for record in result.files
        }
        return result, data


def assert_deterministic(
    deriver: Deriver,
    bundle_dir: Path,
    params: Mapping[str, str],
    *,
    member: str | None = None,
    members: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Derive twice in separate temp stores and assert identical names, bytes and keys."""
    _, shas = _derive_twice(deriver, bundle_dir, params, member=member, members=members)
    return shas


def _derive_twice(
    deriver: Deriver,
    bundle_dir: Path,
    params: Mapping[str, str],
    *,
    member: str | None,
    members: Sequence[Mapping[str, Any]] | None,
) -> tuple[DerivedResult, dict[str, str]]:
    """Derive twice, assert identical outputs and return the first result with its sha256s."""
    first, first_data = derive_in_temp_store(
        deriver, bundle_dir, params, member=member, members=members
    )
    second, second_data = derive_in_temp_store(
        deriver, bundle_dir, params, member=member, members=members
    )
    if first.params_key != second.params_key or first.links_key != second.links_key:
        raise AssertionError(
            f"deriver {deriver.spec.name!r} is not deterministic: "
            f"params_key {first.params_key!r} != {second.params_key!r} or "
            f"links_key {first.links_key!r} != {second.links_key!r}"
        )
    if set(first_data) != set(second_data):
        raise AssertionError(
            f"deriver {deriver.spec.name!r} is not deterministic: file names differ "
            f"({sorted(first_data)} != {sorted(second_data)})"
        )
    for name in first_data:
        if first_data[name] != second_data[name]:
            raise AssertionError(f"deriver {deriver.spec.name!r} is not deterministic: {name!r}")
    return first, {name: hashlib.sha256(data).hexdigest() for name, data in first_data.items()}


def representative_cases(
    deriver: Deriver,
    bundle_dir: Path,
    *,
    members: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[str, dict[str, str]]]:
    """Return the first and last representative parameter case per supported member."""
    cases: list[tuple[str, dict[str, str]]] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(ViewerSettings(data_dir=Path(tmp) / "store"))
        root = store.stage_from_directory(Path(bundle_dir), store.settings.extract_limits)
        member_list = list(members) if members is not None else bundle_members(root)
        digest, _ = store.commit(root, name="test", members=member_list)
        for info in member_list:
            if info.get("kind") not in deriver.spec.kinds:
                continue
            ctx = open_context(store, digest, str(info["id"]))
            space = sorted(deriver.param_space(ctx), key=lambda entry: sorted(entry.items()))
            if not space:
                continue
            first = dict(space[0])
            last = dict(space[-1])
            for param in deriver.spec.params:
                if param.kind not in RANGE_KINDS:
                    continue
                if param.kind == "int":
                    first[param.name] = str(param.min)
                    last[param.name] = str(param.max)
                else:
                    first[param.name] = format(param.min, ".12g")
                    last[param.name] = format(param.max, ".12g")
            seen: list[dict[str, str]] = []
            for entry in (first, last):
                if entry not in seen:
                    seen.append(entry)
            for entry in seen:
                cases.append((str(info["id"]), entry))
    return cases


def _load_golden(path: Path) -> dict[str, Any]:
    """Read a golden file (a missing file is an empty mapping)."""
    golden_path = Path(path)
    if not golden_path.is_file():
        return {}
    text = golden_path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise GoldenMismatch(f"golden file {golden_path} is not a JSON object")
    return payload


def _write_golden(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a golden file with sorted keys, two-space indent and a trailing newline."""
    golden_path = Path(path)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _changed_files(expected: Mapping[str, str], actual: Mapping[str, str]) -> list[str]:
    """Return the sorted file names whose hash changed, appeared or disappeared."""
    return sorted(
        name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name)
    )


def golden_check(
    deriver: Deriver,
    bundle_dir: Path,
    params: Mapping[str, str],
    *,
    member: str | None = None,
    case: str = "default",
    members: Sequence[Mapping[str, Any]] | None = None,
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    update: bool = False,
) -> dict[str, str]:
    """Check (or update) one golden case after proving the deriver deterministic."""
    result, shas = _derive_twice(deriver, bundle_dir, params, member=member, members=members)
    key = f"{case}/{result.member}/{result.params_key}"
    name = deriver.spec.name
    version = deriver.spec.version
    golden = _load_golden(golden_path)
    if update:
        record = golden.get(name)
        if not isinstance(record, dict) or record.get("version") != version:
            golden[name] = {"version": version, "cases": {key: shas}}
        else:
            record.setdefault("cases", {})[key] = shas
        _write_golden(golden_path, golden)
        return shas
    record = golden.get(name)
    if record is None:
        raise GoldenMismatch(f"no golden for deriver {name!r}; run with --update-viewer-golden")
    recorded_version = record.get("version")
    if recorded_version != version:
        raise GoldenMismatch(
            f"deriver {name!r} is v{version} but the golden is v{recorded_version}; "
            "update the golden with --update-viewer-golden"
        )
    cases = record.get("cases", {})
    if key not in cases:
        raise GoldenMismatch(
            f"no golden case {key!r} for deriver {name!r}; run with --update-viewer-golden"
        )
    expected = cases[key]
    if expected != shas:
        changed = _changed_files(expected, shas)
        raise GoldenMismatch(
            f"outputs of deriver {name!r} changed but its version is still v{version}; "
            f"bump DeriverSpec.version (changed files: {changed})"
        )
    return shas


def check_all_goldens(
    bundles: Mapping[str, Path],
    *,
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    update: bool = False,
    derivers: Sequence[Deriver] | None = None,
) -> None:
    """Check every representative case of every deriver against the golden file."""
    checked = list(derivers) if derivers is not None else list(registered_derivers())
    checked_names = {deriver.spec.name for deriver in checked}
    golden = _load_golden(golden_path)
    failures: list[str] = []
    has_case = {deriver.spec.name: False for deriver in checked}
    for deriver in checked:
        for label, bundle_dir in bundles.items():
            for member_id, params in representative_cases(deriver, bundle_dir):
                has_case[deriver.spec.name] = True
                try:
                    golden_check(
                        deriver,
                        bundle_dir,
                        params,
                        member=member_id,
                        case=label,
                        golden_path=golden_path,
                        update=update,
                    )
                except GoldenMismatch as exc:
                    failures.append(str(exc))
    for deriver in checked:
        if not has_case[deriver.spec.name]:
            failures.append(
                f"deriver {deriver.spec.name!r} has no representative case in the fixture bundles"
            )
    if not update:
        for name in sorted(golden):
            if name not in checked_names:
                failures.append(f"stale golden entry for {name!r}")
    else:
        reloaded = _load_golden(golden_path)
        _write_golden(
            golden_path, {name: value for name, value in reloaded.items() if name in checked_names}
        )
    if failures:
        raise GoldenMismatch("\n".join(failures))
