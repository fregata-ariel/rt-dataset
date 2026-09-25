"""Bundle kind detection (docs/viewer_bundle.md) and member validation."""

from __future__ import annotations

import json
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    RFDatasetManifest,
    parse_rf_dataset_manifest,
)
from plateau_rt.application.scene_files import (
    FILE_SHAPE_TYPES,
    SceneFileError,
    scene_file_references,
)
from plateau_rt.viewer import safeio
from plateau_rt.viewer.safeio import DEFAULT_MAX_XML_BYTES, UnsafePathError
from plateau_rt.viewer.settings import DEFAULT_MAX_ARRAY_BYTES

BUNDLE_JSON = "bundle.json"
BUNDLE_FORMAT_VERSION = 1
MAX_BUNDLE_JSON_BYTES = 1024 * 1024
MAX_MEMBERS = 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MEMBER_KINDS = ("rf_dataset", "rf_partial", "tomo_run", "scene")
DIRECTORY_KINDS = ("rf_dataset", "rf_partial", "tomo_run")
MARKER_FILES = {
    "rf_dataset": "dataset_manifest.json",
    "rf_partial": "partial_manifest.json",
    "tomo_run": "run_manifest.json",
}
FALLBACK_IDS = {"rf_dataset": "dataset", "rf_partial": "partial", "tomo_run": "run"}
PARTIAL_SCHEMA_VERSION = 1
PARTIAL_MODE = "rf_camera_partial_observation"

_TOP_LEVEL_KEYS = frozenset({"bundle_format_version", "members", "created_by"})
_MEMBER_KEYS = frozenset({"id", "kind", "path", "for", "source"})
_MEMBER_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_DRIVE_RE = re.compile(r"[A-Za-z]:")
_FALLBACK_MARKERS = (
    ("dataset_manifest.json", "dataset", "rf_dataset"),
    ("partial_manifest.json", "partial", "rf_partial"),
    ("run_manifest.json", "run", "tomo_run"),
)


class BundleValidationError(ValueError):
    """A bundle member failed kind detection or validation."""

    def __init__(self, member_id: str | None, message: str) -> None:
        """Store the member id and message, prefixing str() with the id."""
        self.member_id = member_id
        self.message = message
        super().__init__(message if member_id is None else f"member {member_id!r}: {message}")


@dataclass(frozen=True)
class Member:
    """One detected bundle member: id, kind, bundle-relative path and links."""

    id: str
    kind: str
    path: str
    links: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the store record for this member."""
        return {"id": self.id, "kind": self.kind, "path": self.path, "links": dict(self.links)}


@dataclass(frozen=True)
class MemberInfo:
    """A validated member with its schema facts and checked files."""

    member: Member
    schema_version: int | None
    mode: str | None
    files: tuple[str, ...]
    summary: dict[str, Any]


def member_relpath(member: Member, relpath: str) -> str:
    """Return the bundle-root-relative POSIX path of ``relpath`` in ``member``."""
    if member.path == ".":
        return relpath
    return posixpath.join(member.path, relpath)


def _member_path_relpath(member_path: str, relpath: str) -> str:
    """Join ``relpath`` onto a member path string without normalisation."""
    if member_path == ".":
        return relpath
    return posixpath.join(member_path, relpath)


def dataset_manifest_from_bytes(data: bytes, member_path: str) -> RFDatasetManifest:
    """Parse dataset manifest ``data`` with bundle-root-relative artifact paths."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"dataset_manifest.json is not valid JSON: {exc}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"dataset_manifest.json is not valid JSON: {exc}") from exc
    return parse_rf_dataset_manifest(
        obj,
        root=Path(member_path),
        manifest_path=Path(_member_path_relpath(member_path, "dataset_manifest.json")),
    )


def validate_bundle(
    raw_root: Path | str, *, max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES
) -> list[MemberInfo]:
    """Detect and validate every member of the bundle at ``raw_root``."""
    members = detect_members(raw_root)
    return [
        validate_member(member, raw_root, max_array_bytes=max_array_bytes) for member in members
    ]


def detect_members(raw_root: Path | str) -> list[Member]:
    """Detect bundle members from ``bundle.json`` or the fallback markers."""
    root = Path(raw_root)
    if os.path.lexists(root / BUNDLE_JSON):
        return _detect_from_bundle_json(root)
    return _detect_fallback(root)


def _detect_from_bundle_json(root: Path) -> list[Member]:
    """Detect members declared in ``bundle.json``."""
    bundle_path = root / BUNDLE_JSON
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise BundleValidationError(None, "bundle.json is not a regular file")
    try:
        data = safeio.read_bytes(bundle_path, max_bytes=MAX_BUNDLE_JSON_BYTES)
    except ValueError as exc:
        raise BundleValidationError(None, "bundle.json is larger than 1048576 bytes") from exc
    except OSError as exc:
        raise BundleValidationError(None, f"bundle.json could not be read: {exc}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleValidationError(None, f"bundle.json is not valid UTF-8 JSON: {exc}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BundleValidationError(None, f"bundle.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise BundleValidationError(None, "bundle.json must be a JSON object")
    unknown_top = sorted(key for key in obj if key not in _TOP_LEVEL_KEYS)
    if unknown_top:
        raise BundleValidationError(None, f"unknown key(s) in bundle.json: {unknown_top}")
    version = obj.get("bundle_format_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != BUNDLE_FORMAT_VERSION
    ):
        raise BundleValidationError(
            None, f"unsupported bundle_format_version {version!r} (expected 1)"
        )
    members_raw = obj.get("members")
    if not isinstance(members_raw, list) or not members_raw or len(members_raw) > MAX_MEMBERS:
        raise BundleValidationError(
            None, "bundle.json 'members' must be a non-empty list of at most 1024 entries"
        )
    if "created_by" in obj:
        created_by = obj["created_by"]
        if not isinstance(created_by, dict):
            raise BundleValidationError(None, "'created_by' must be an object")
        tool = created_by.get("tool")
        tool_version = created_by.get("tool_version")
        if ("tool" in created_by and not isinstance(tool, str)) or (
            "tool_version" in created_by
            and (isinstance(tool_version, bool) or not isinstance(tool_version, int))
        ):
            raise BundleValidationError(
                None, "'created_by' has an invalid 'tool' or 'tool_version'"
            )
    raw_members: list[dict[str, Any]] = []
    for index, entry in enumerate(members_raw):
        if not isinstance(entry, dict):
            raise BundleValidationError(None, f"member #{index} must be an object")
        entry_id = entry.get("id")
        label: str | None = entry_id if isinstance(entry_id, str) else None
        unknown = sorted(key for key in entry if key not in _MEMBER_KEYS)
        if unknown:
            raise BundleValidationError(label, f"unknown key(s) {unknown}")
        member_id = entry.get("id")
        if (
            not isinstance(member_id, str)
            or _MEMBER_ID_RE.fullmatch(member_id) is None
            or member_id in (".", "..")
        ):
            raise BundleValidationError(
                None, f"member #{index}: missing or invalid id {member_id!r}"
            )
        if any(existing.get("id") == member_id for existing in raw_members):
            raise BundleValidationError(member_id, "duplicate member id")
        kind = entry.get("kind")
        if kind not in MEMBER_KINDS:
            raise BundleValidationError(
                member_id,
                f"invalid kind {kind!r}; expected one of rf_dataset, rf_partial, tomo_run, scene",
            )
        member_path = entry.get("path")
        _check_member_path(root, member_id, str(kind), member_path)
        raw_members.append(dict(entry))
    _check_directory_collisions(raw_members)
    _check_links(raw_members)
    dataset_by_path = {
        str(entry["path"]): str(entry["id"])
        for entry in raw_members
        if entry["kind"] == "rf_dataset"
    }
    members: list[Member] = []
    for entry in raw_members:
        member_id = str(entry["id"])
        kind = str(entry["kind"])
        member_path = str(entry["path"])
        links: dict[str, str] = {}
        if kind == "scene":
            if "for" in entry:
                links = {"for": str(entry["for"])}
        elif kind in ("rf_partial", "tomo_run") and "source" in entry:
            links = {"source": str(entry["source"])}
        members.append(Member(id=member_id, kind=kind, path=member_path, links=links))
    for index, member in enumerate(members):
        if member.kind == "rf_partial" and "source" not in member.links:
            linked = _partial_link_from_manifest(root, member, dataset_by_path)
            if linked is not None:
                members[index] = Member(member.id, member.kind, member.path, {"source": linked})
    return members


def _check_member_path(root: Path, member_id: str, kind: str, member_path: Any) -> None:
    """Validate one member path lexically, on disk and through resolve_inside."""
    if (
        not isinstance(member_path, str)
        or not member_path
        or member_path == "."
        or member_path.startswith("/")
        or "\\" in member_path
        or "\x00" in member_path
        or _DRIVE_RE.match(member_path) is not None
        or any(segment in ("", ".", "..") for segment in member_path.split("/"))
    ):
        raise BundleValidationError(member_id, f"invalid path {member_path!r}")
    current = Path(root)
    for segment in member_path.split("/"):
        current = current / segment
        if not os.path.lexists(current):
            raise BundleValidationError(member_id, f"path {member_path!r} does not exist")
        if os.path.islink(current):
            raise BundleValidationError(member_id, f"path {member_path!r} goes through a symlink")
    try:
        resolved = safeio.resolve_inside(root, member_path)
    except UnsafePathError as exc:
        raise BundleValidationError(member_id, f"invalid path {member_path!r}: {exc}") from exc
    if kind in DIRECTORY_KINDS:
        if not resolved.is_dir():
            raise BundleValidationError(
                member_id, f"path {member_path!r} must be a directory for kind {kind}"
            )
        marker = MARKER_FILES[kind]
        marker_path = resolved / marker
        if marker_path.is_symlink() or not marker_path.is_file():
            raise BundleValidationError(member_id, f"missing {marker} in {member_path!r}")
    else:
        if not member_path.endswith(".xml") or not resolved.is_file():
            raise BundleValidationError(
                member_id, f"scene path {member_path!r} must be an .xml file"
            )
        try:
            element = safeio.parse_xml(resolved, max_bytes=DEFAULT_MAX_XML_BYTES)
        except safeio.UnsafeXmlError as exc:
            raise BundleValidationError(member_id, str(exc)) from exc
        if element.tag != "scene":
            raise BundleValidationError(member_id, f"root element is <{element.tag}>, not <scene>")


def _check_directory_collisions(raw_members: list[dict[str, Any]]) -> None:
    """Reject duplicate or nested directory member paths (scenes are exempt)."""
    dirs = [
        (str(entry["id"]), str(entry["path"]))
        for entry in raw_members
        if entry["kind"] in DIRECTORY_KINDS
    ]
    for index, (first_id, first) in enumerate(dirs):
        for second_id, second in dirs[index + 1 :]:
            if first == second:
                raise BundleValidationError(
                    second_id, f"same directory path as member {first_id!r}: {second!r}"
                )
            if second.startswith(first + "/"):
                raise BundleValidationError(
                    second_id, f"path {second!r} is nested inside member {first_id!r} ({first!r})"
                )
            if first.startswith(second + "/"):
                raise BundleValidationError(
                    first_id, f"path {first!r} is nested inside member {second_id!r} ({second!r})"
                )


def _check_links(raw_members: list[dict[str, Any]]) -> None:
    """Validate for/source links between members."""
    id_to_kind = {str(entry["id"]): str(entry["kind"]) for entry in raw_members}
    scenes_for: dict[str, str] = {}
    for entry in raw_members:
        member_id = str(entry["id"])
        kind = str(entry["kind"])
        if "for" in entry:
            if kind != "scene":
                raise BundleValidationError(member_id, "'for' is only allowed on scene members")
            value = entry["for"]
            if (
                not isinstance(value, str)
                or value not in id_to_kind
                or id_to_kind[value] != "rf_dataset"
            ):
                raise BundleValidationError(
                    member_id, f"'for' names {value!r}, which is not an rf_dataset member"
                )
            if value in scenes_for:
                raise BundleValidationError(member_id, f"two scenes have 'for' {value!r}")
            scenes_for[value] = member_id
        if "source" in entry:
            if kind not in ("rf_partial", "tomo_run"):
                raise BundleValidationError(
                    member_id, "'source' is only allowed on rf_partial and tomo_run members"
                )
            value = entry["source"]
            if (
                not isinstance(value, str)
                or value not in id_to_kind
                or id_to_kind[value] != "rf_dataset"
            ):
                raise BundleValidationError(
                    member_id, f"'source' names {value!r}, which is not an rf_dataset member"
                )


def _partial_link_from_manifest(
    root: Path, member: Member, dataset_by_path: dict[str, str]
) -> str | None:
    """Return the rf_dataset id a partial's ``source_dataset`` names, or None for no link."""
    try:
        relpath = _member_path_relpath(member.path, "partial_manifest.json")
        resolved = safeio.resolve_inside(root, relpath)
        data = safeio.read_bytes(resolved, max_bytes=MAX_MANIFEST_BYTES)
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    source_dataset = obj.get("source_dataset")
    if not isinstance(source_dataset, str) or not source_dataset:
        return None
    if (
        source_dataset.startswith("/")
        or source_dataset.startswith("\\")
        or _DRIVE_RE.match(source_dataset) is not None
    ):
        return None
    joined = posixpath.normpath(posixpath.join(member.path, source_dataset))
    if joined == ".." or joined.startswith("../"):
        return None
    return dataset_by_path.get(joined)


def _detect_fallback(root: Path) -> list[Member]:
    """Detect the single implicit member from root marker manifests."""
    found: list[tuple[str, str, str]] = []
    for file_name, member_id, kind in _FALLBACK_MARKERS:
        candidate = root / file_name
        if candidate.is_file() and not candidate.is_symlink():
            found.append((file_name, member_id, kind))
    if not found:
        raise BundleValidationError(
            None,
            "no bundle.json and no marker manifest at the bundle root (expected exactly one of "
            "dataset_manifest.json, partial_manifest.json, run_manifest.json)",
        )
    if len(found) > 1:
        names = ", ".join(name for name, _, _ in found)
        raise BundleValidationError(
            None,
            f"no bundle.json and several marker manifests at the bundle root: {names}; "
            "add a bundle.json that lists the members",
        )
    _, member_id, kind = found[0]
    return [Member(id=member_id, kind=kind, path=".", links={})]


def validate_member(
    member: Member, raw_root: Path | str, *, max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES
) -> MemberInfo:
    """Validate one member's data and return its MemberInfo."""
    root = Path(raw_root)
    if member.kind == "rf_dataset":
        return _validate_dataset(member, root, max_array_bytes=max_array_bytes)
    if member.kind == "scene":
        return _validate_scene(member, root)
    if member.kind == "rf_partial":
        return _validate_partial(member, root)
    if member.kind == "tomo_run":
        raise BundleValidationError(
            member.id, "unsupported member kind 'tomo_run' (supported from V3-2, #71)"
        )
    raise BundleValidationError(member.id, f"unsupported member kind {member.kind!r}")


def _read_manifest_bytes(member: Member, root: Path, relpath: str) -> bytes:
    """Read a manifest file, mapping size problems to BundleValidationError."""
    try:
        resolved = safeio.resolve_inside(root, relpath)
    except UnsafePathError as exc:
        raise BundleValidationError(member.id, f"{relpath!r}: {exc}") from exc
    try:
        return safeio.read_bytes(resolved, max_bytes=MAX_MANIFEST_BYTES)
    except ValueError as exc:
        raise BundleValidationError(member.id, str(exc)) from exc
    except OSError as exc:
        raise BundleValidationError(member.id, f"{relpath!r} does not exist: {exc}") from exc


def _validate_dataset(member: Member, root: Path, *, max_array_bytes: int) -> MemberInfo:
    """Validate an rf_dataset member's manifest and every artifact it names."""
    manifest_relpath = member_relpath(member, "dataset_manifest.json")
    data = _read_manifest_bytes(member, root, manifest_relpath)
    try:
        manifest = dataset_manifest_from_bytes(data, member.path)
    except ManifestError as exc:
        raise BundleValidationError(member.id, str(exc)) from exc
    # (label, bundle-relative path, is the view's aperture CFR)
    labeled: list[tuple[str, Path, bool]] = [("camera_model", manifest.camera_model_path, False)]
    for view in manifest.views:
        for key, artifact in view.artifacts.items():
            labeled.append((f"view {view.view_id!r} {key}", artifact, key == "aperture_cfr"))
        for entry in view.bs:
            for key, artifact in entry.artifacts.items():
                labeled.append((f"view {view.view_id!r} bs {entry.bs_id!r} {key}", artifact, False))
    if manifest.path_geometry_gt is not None:
        labeled.append(("path_geometry_gt", manifest.path_geometry_gt.path, False))
        if manifest.path_geometry_gt.schema_path is not None:
            labeled.append(("path_schema", manifest.path_geometry_gt.schema_path, False))
    optical_reference = manifest.raw.get("optical_reference")
    if isinstance(optical_reference, dict):
        pinhole = optical_reference.get("pinhole")
        if isinstance(pinhole, dict):
            transforms = pinhole.get("transforms")
            if isinstance(transforms, str):
                labeled.append(
                    ("optical_reference transforms", Path(member.path) / transforms, False)
                )
    resolved_paths: dict[str, Path] = {}
    for label, artifact, _ in labeled:
        relpath = str(artifact)
        try:
            resolved = safeio.resolve_inside(root, relpath)
        except UnsafePathError as exc:
            raise BundleValidationError(member.id, f"{label} {relpath!r}: {exc}") from exc
        if not resolved.is_file():
            raise BundleValidationError(member.id, f"{label} {relpath!r} does not exist")
        resolved_paths[relpath] = resolved
    for label, artifact, is_aperture_cfr in labeled:
        relpath = str(artifact)
        resolved = resolved_paths[relpath]
        suffix = resolved.suffix
        try:
            if is_aperture_cfr:
                array = safeio.load_npy(resolved, max_bytes=max_array_bytes, mmap=True)
                try:
                    shape = tuple(int(dim) for dim in array.shape)
                    dtype = array.dtype
                    bins = manifest.num_frequency_bins
                    if manifest.schema_version == 3:
                        expected = (
                            manifest.num_bs,
                            len(manifest.hemispheres),
                            manifest.rx_rows,
                            manifest.rx_cols,
                            bins,
                        )
                    else:
                        expected = (
                            len(manifest.hemispheres),
                            manifest.rx_rows,
                            manifest.rx_cols,
                            bins,
                        )
                    if tuple(shape) != tuple(expected) or not np.issubdtype(
                        dtype, np.complexfloating
                    ):
                        raise BundleValidationError(
                            member.id,
                            f"{label}: shape {shape} dtype {dtype} does not match "
                            f"expected {expected} (complex)",
                        )
                finally:
                    del array
            elif suffix == ".npy":
                info = safeio.read_npy_header(resolved)
                if info.dtype.hasobject:
                    raise BundleValidationError(
                        member.id, f"{label}: object dtype {info.dtype} is not allowed"
                    )
            elif suffix == ".npz":
                safeio.check_npz(resolved, max_member_bytes=max_array_bytes)
        except BundleValidationError:
            raise
        except ValueError as exc:
            raise BundleValidationError(member.id, f"{label}: {exc}") from exc
    files = sorted({manifest_relpath} | set(resolved_paths))
    summary: dict[str, Any] = {
        "num_views": manifest.num_views,
        "num_bs": manifest.num_bs,
        "num_frequency_bins": manifest.num_frequency_bins,
    }
    return MemberInfo(
        member=member,
        schema_version=manifest.schema_version,
        mode=manifest.mode,
        files=tuple(files),
        summary=summary,
    )


def _validate_scene(member: Member, root: Path) -> MemberInfo:
    """Validate a scene member's XML and every PLY file it references."""
    try:
        resolved = safeio.resolve_inside(root, member.path)
    except UnsafePathError as exc:
        raise BundleValidationError(member.id, f"{member.path!r}: {exc}") from exc
    try:
        refs = scene_file_references(resolved)
    except SceneFileError as exc:
        raise BundleValidationError(member.id, str(exc)) from exc
    checked: list[str] = []
    for ref in refs:
        if ref.unsafe:
            raise BundleValidationError(
                member.id,
                f"shape {ref.shape_id!r} filename {ref.filename!r} is unsafe (absolute or '..')",
            )
        if ref.filename is None:
            if ref.shape_type in FILE_SHAPE_TYPES:
                raise BundleValidationError(
                    member.id,
                    f"shape {ref.shape_id!r} of type {ref.shape_type!r} has no filename",
                )
            continue
        dirname = posixpath.dirname(member.path)
        relpath = posixpath.join(dirname, ref.filename) if dirname else ref.filename
        try:
            resolved_ref = safeio.resolve_inside(root, relpath)
        except UnsafePathError as exc:
            raise BundleValidationError(
                member.id, f"shape {ref.shape_id!r} filename {ref.filename!r}: {exc}"
            ) from exc
        if not resolved_ref.is_file():
            raise BundleValidationError(
                member.id, f"shape {ref.shape_id!r} file {relpath!r} does not exist"
            )
        checked.append(relpath)
    files = tuple([member.path] + sorted(set(checked)))
    return MemberInfo(
        member=member,
        schema_version=None,
        mode=None,
        files=files,
        summary={"num_shapes": len(refs)},
    )


def _validate_partial(member: Member, root: Path) -> MemberInfo:
    """Validate an rf_partial member's manifest envelope."""
    manifest_relpath = member_relpath(member, "partial_manifest.json")
    data = _read_manifest_bytes(member, root, manifest_relpath)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleValidationError(
            member.id, f"partial_manifest.json is not a valid JSON object: {exc}"
        ) from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BundleValidationError(
            member.id, f"partial_manifest.json is not a valid JSON object: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise BundleValidationError(
            member.id,
            f"partial_manifest.json is not a valid JSON object: expected object, "
            f"got {type(obj).__name__}",
        )
    schema_version = obj.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PARTIAL_SCHEMA_VERSION
    ):
        raise BundleValidationError(
            member.id,
            f"unsupported partial schema_version {schema_version!r} (supported: 1)",
        )
    mode = obj.get("mode")
    if mode != PARTIAL_MODE:
        raise BundleValidationError(
            member.id, f"unsupported partial mode {mode!r} (expected {PARTIAL_MODE!r})"
        )
    return MemberInfo(
        member=member,
        schema_version=PARTIAL_SCHEMA_VERSION,
        mode=PARTIAL_MODE,
        files=(manifest_relpath,),
        summary={},
    )
