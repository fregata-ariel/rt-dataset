"""Deriver registry, parameter validation and versioned cache.

The design is documented in ``docs/viewer.md`` ("Derivers"). A deriver declares a
:class:`DeriverSpec`, validates a request against :func:`validate_params`, and returns named
outputs that are cached under ``derived/<member>/<deriver>/v<version>/<params_key>/<links_key>/``.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import importlib
import io
import json
import math
import os
import posixpath
import re
import urllib.parse
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from plateau_rt.viewer import safeio
from plateau_rt.viewer.kinds import MEMBER_KINDS  # re-exported: one definition of the kinds
from plateau_rt.viewer.safeio import NpyInfo, UnsafePathError
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store, _make_writable_and_remove

PARAM_KINDS = ("enum", "int", "float", "view", "bs", "member_link")
RANGE_KINDS = ("int", "float")
SPACE_KINDS = ("enum", "view", "bs", "member_link")
OUTPUT_DTYPES = ("float16", "float32", "int32", "uint8", "uint32", "bool")
STATUS_READY = "ready"
STATUS_FAILED = "failed"
NO_PARAMS_KEY = "noparams"
NO_LINK_KEY = "nolink"
SELF_LINK = "self"
NONE_LINK = "none"
META_FILE = "_meta.json"
MAX_KEY_CHARS = 200

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MEMBER_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_OUTPUT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_INT_RE = re.compile(r"[+-]?[0-9]{1,18}\Z")
_FLOAT_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


class BadParams(ValueError):
    """The request parameters are invalid (HTTP 400 later)."""


class DeriveNotFound(LookupError):
    """An unknown bundle / member / deriver, or an unsupported member kind."""


class DeriveError(RuntimeError):
    """The deriver failed or returned invalid outputs."""


@dataclass(frozen=True)
class ParamSpec:
    """Declaration of one deriver parameter."""

    name: str
    kind: str
    values: tuple[str, ...] | None = None
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None

    def __post_init__(self) -> None:
        if _NAME_RE.fullmatch(self.name) is None:
            raise ValueError(f"invalid parameter name {self.name!r}")
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"invalid parameter kind {self.kind!r}")
        if self.kind == "enum":
            if (
                not isinstance(self.values, tuple)
                or not self.values
                or any(not isinstance(v, str) or not v for v in self.values)
                or len(set(self.values)) != len(self.values)
            ):
                raise ValueError(f"enum {self.name!r} needs unique non-empty string values")
            if self.min is not None or self.max is not None or self.step is not None:
                raise ValueError(f"enum {self.name!r} must not set min/max/step")
        elif self.kind == "int":
            if (
                isinstance(self.min, bool)
                or isinstance(self.max, bool)
                or not isinstance(self.min, int)
                or not isinstance(self.max, int)
            ):
                raise ValueError(f"int {self.name!r} needs integer min and max")
            if self.min > self.max:
                raise ValueError(f"int {self.name!r}: min > max")
            if self.values is not None or self.step is not None:
                raise ValueError(f"int {self.name!r} must not set values/step")
        elif self.kind == "float":
            if not all(_is_finite_number(v) for v in (self.min, self.max, self.step)):
                raise ValueError(f"float {self.name!r} needs finite min, max and step")
            if self.step <= 0:
                raise ValueError(f"float {self.name!r}: step must be positive")
            if self.min > self.max:
                raise ValueError(f"float {self.name!r}: min > max")
            quotient = (self.max - self.min) / self.step
            if abs(quotient - round(quotient)) > 1e-9:
                raise ValueError(f"float {self.name!r}: step does not divide the range")
            if self.values is not None:
                raise ValueError(f"float {self.name!r} must not set values")
        else:
            if any(v is not None for v in (self.values, self.min, self.max, self.step)):
                raise ValueError(f"{self.kind} {self.name!r} must not set values/min/max/step")


@dataclass(frozen=True)
class DeriverSpec:
    """Declaration of one deriver: name, version, supported kinds and parameters."""

    name: str
    version: int
    kinds: tuple[str, ...]
    params: tuple[ParamSpec, ...] = ()
    eager: bool = False

    def __post_init__(self) -> None:
        if _NAME_RE.fullmatch(self.name) is None:
            raise ValueError(f"invalid deriver name {self.name!r}")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError(f"deriver {self.name!r}: version must be an int >= 1")
        if not isinstance(self.kinds, tuple) or not self.kinds:
            raise ValueError(f"deriver {self.name!r}: kinds must be a non-empty tuple")
        if any(kind not in MEMBER_KINDS for kind in self.kinds):
            raise ValueError(f"deriver {self.name!r}: unknown member kind")
        if not isinstance(self.params, tuple) or any(
            not isinstance(p, ParamSpec) for p in self.params
        ):
            raise ValueError(f"deriver {self.name!r}: params must be a tuple of ParamSpec")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"deriver {self.name!r}: duplicate parameter names")
        if self.eager and any(p.kind in RANGE_KINDS for p in self.params):
            raise ValueError(f"eager deriver {self.name!r} must not have range parameters")


def _is_finite_number(value: Any) -> bool:
    """Return True for a finite int/float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class LinkTarget:
    """A link from a derivation to a member or file, in this or another bundle."""

    member: str | None = None
    sha256: str | None = None
    kind: str | None = None

    def __post_init__(self) -> None:
        if self.member is None and self.sha256 is None:
            raise ValueError("link target needs a member id or a sha256")
        if self.sha256 is not None:
            lowered = self.sha256.lower()
            if len(lowered) != 64 or any(c not in "0123456789abcdef" for c in lowered):
                raise ValueError(f"invalid sha256 {self.sha256!r}")
            object.__setattr__(self, "sha256", lowered)
        if self.kind is not None and self.kind not in MEMBER_KINDS:
            raise ValueError(f"invalid link kind {self.kind!r}")


@dataclass(frozen=True)
class LinkedBundle:
    """A resolved link target: which bundle, member and file it names."""

    digest: str
    member: str | None
    kind: str | None
    relpath: str | None
    raw_root: Path
    same_bundle: bool

    def path(self, relpath: str) -> Path:
        """Resolve ``relpath`` against the linked bundle's raw root."""
        return safeio.resolve_inside(self.raw_root, relpath)


class Deriver(Protocol):
    """The protocol every deriver implements."""

    spec: DeriverSpec

    def param_space(self, ctx: DeriveContext) -> list[dict[str, str]]:
        """Return the allowed combinations of space-kind parameters."""

    def derive(
        self, ctx: DeriveContext, params: Mapping[str, Any]
    ) -> Mapping[str, np.ndarray | dict | list | bytes]:
        """Compute the named outputs for one parameter combination."""


_REGISTRY: dict[str, Deriver] = {}


def register(deriver: Deriver, *, replace: bool = False) -> Deriver:
    """Register ``deriver`` under ``deriver.spec.name`` and return it."""
    spec = getattr(deriver, "spec", None)
    if not isinstance(spec, DeriverSpec):
        raise TypeError(f"{deriver!r}.spec is not a DeriverSpec")
    existing = _REGISTRY.get(spec.name)
    if existing is deriver:
        return deriver
    if existing is not None and not replace:
        raise ValueError(f"deriver {spec.name!r} is already registered")
    _REGISTRY[spec.name] = deriver
    return deriver


def unregister(name: str) -> None:
    """Remove the deriver registered under ``name`` (KeyError when unknown)."""
    del _REGISTRY[name]


def get_deriver(name: str) -> Deriver:
    """Return the deriver registered under ``name`` or raise DeriveNotFound."""
    deriver = _REGISTRY.get(name)
    if deriver is None:
        raise DeriveNotFound(f"unknown deriver {name!r}")
    return deriver


def registered_derivers() -> tuple[Deriver, ...]:
    """Return all registered derivers sorted by name."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


@contextlib.contextmanager
def registered(*derivers: Deriver) -> Iterator[None]:
    """Temporarily register ``derivers`` and restore the previous registry on exit."""
    snapshot = dict(_REGISTRY)
    try:
        for deriver in derivers:
            register(deriver)
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def _now() -> str:
    """Return the store timestamp for the current UTC time."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _member_contains(member_path: str, relpath: str) -> bool:
    """Return True when ``relpath`` lies inside the member directory ``member_path``."""
    if member_path in (".", ""):
        return True
    return relpath == member_path or relpath.startswith(member_path + "/")


def _resolve_link(
    store: Store,
    digest: str,
    members: Sequence[Mapping[str, Any]],
    target: LinkTarget,
) -> LinkedBundle | None:
    """Resolve ``target`` against the own bundle then the store's file index."""
    if target.member is not None:
        for member in members:
            if member.get("id") != target.member:
                continue
            kind = member.get("kind")
            if target.kind is not None and kind != target.kind:
                break
            return LinkedBundle(
                digest=digest,
                member=target.member,
                kind=kind,
                relpath=None,
                raw_root=store.raw_dir(digest),
                same_bundle=True,
            )
    if target.sha256 is None:
        return None
    candidates: list[tuple[bool, str, str, str, str, str | None]] = []
    for record in store.find_by_file_sha256(target.sha256):
        bundle = store.get(record.digest)
        if bundle is None:
            continue
        same = record.digest == digest
        contained = [
            member
            for member in bundle.members
            if _member_contains(str(member.get("path", "")), record.relpath)
            and (target.kind is None or member.get("kind") == target.kind)
        ]
        if contained:
            for member in contained:
                candidates.append(
                    (
                        same,
                        bundle.created_at,
                        record.digest,
                        str(member.get("id", "")),
                        record.relpath,
                        member.get("kind"),
                    )
                )
        elif target.kind is None:
            candidates.append((same, bundle.created_at, record.digest, "", record.relpath, None))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (not c[0], c[1], c[2], c[3], c[4]))
    same, _created, chosen, member_id, relpath, kind = candidates[0]
    return LinkedBundle(
        digest=chosen,
        member=member_id or None,
        kind=kind,
        relpath=relpath,
        raw_root=store.raw_dir(chosen),
        same_bundle=same,
    )


def _token_for(result: LinkedBundle | None) -> str:
    """Return the links-key token for a resolved link."""
    if result is None:
        return NONE_LINK
    if result.same_bundle:
        return SELF_LINK
    return result.digest


def links_key(tokens: Sequence[str]) -> str:
    """Return the cache key segment for a sequence of link tokens."""
    if not tokens or all(token == NONE_LINK for token in tokens):
        return NO_LINK_KEY
    joined = "_".join(tokens)
    if len(joined) > MAX_KEY_CHARS:
        return "links-" + hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return joined


class DeriveContext:
    """One deriver run: the bundle, the member and the safe loaders around it."""

    store: Store
    settings: ViewerSettings
    digest: str
    raw_root: Path
    member: str
    member_info: Mapping[str, Any]
    members: tuple[Mapping[str, Any], ...]

    def __init__(
        self,
        store: Store,
        digest: str,
        member_info: Mapping[str, Any],
        members: Sequence[Mapping[str, Any]],
    ) -> None:
        """Build a context for one member of the already-resolved bundle."""
        self.store = store
        self.settings = store.settings
        self.digest = digest
        self.raw_root = Path(os.path.realpath(store.raw_dir(digest)))
        self.member = str(member_info["id"])
        self.member_info = MappingProxyType(dict(member_info))
        self.members = tuple(MappingProxyType(dict(m)) for m in members)
        self._allowed_roots: list[Path] = [self.raw_root]
        self._links: list[tuple[LinkTarget, str]] = []

    def member_path(self, relpath: str = "") -> Path:
        """Resolve ``relpath`` inside this member's directory."""
        rel = str(relpath)
        safeio.resolve_inside(self.raw_root, rel)
        base = str(self.member_info.get("path", ""))
        if not rel:
            joined = base
        elif base in (".", ""):
            joined = rel
        else:
            joined = posixpath.join(base, rel)
        return safeio.resolve_inside(self.raw_root, joined)

    def path(self, relpath: str) -> Path:
        """Resolve ``relpath`` against the bundle root."""
        return safeio.resolve_inside(self.raw_root, relpath)

    def _loader_path(self, path: str | Path) -> Path:
        """Resolve a loader argument to a checked absolute path."""
        if isinstance(path, str):
            return self.path(path)
        candidate = Path(path)
        if not candidate.is_absolute():
            raise UnsafePathError(f"path is not absolute: {path}")
        real = os.path.realpath(candidate)
        for root in self._allowed_roots:
            root_real = os.path.realpath(root)
            if real == root_real or real.startswith(root_real + os.sep):
                return safeio.resolve_inside(root, os.path.relpath(real, root_real))
        raise UnsafePathError(f"path leaves the allowed roots: {path}")

    def load_npy(self, path: str | Path, *, mmap: bool = False) -> np.ndarray:
        """Load a checked ``.npy`` file bounded by the settings array cap."""
        return safeio.load_npy(
            self._loader_path(path),
            max_bytes=self.settings.max_array_bytes,
            mmap=mmap,
        )

    def check_npz(self, path: str | Path) -> dict[str, NpyInfo]:
        """Validate a checked ``.npz`` file bounded by the settings array cap."""
        return safeio.check_npz(
            self._loader_path(path), max_member_bytes=self.settings.max_array_bytes
        )

    def load_npz(
        self, path: str | Path, keys: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Load checked ``.npz`` members bounded by the settings array cap."""
        return safeio.load_npz(
            self._loader_path(path),
            keys,
            max_member_bytes=self.settings.max_array_bytes,
        )

    def parse_xml(self, path: str | Path) -> safeio.XmlElement:
        """Parse a checked XML file with DTDs, entities and external refs forbidden."""
        return safeio.parse_xml(self._loader_path(path))

    def read_bytes(self, path: str | Path, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        """Read checked bytes from a file no larger than ``max_bytes``."""
        return safeio.read_bytes(self._loader_path(path), max_bytes=max_bytes)

    def link(self, target: LinkTarget) -> LinkedBundle | None:
        """Resolve, record and allow a link; repeated targets are recorded once."""
        result = _resolve_link(self.store, self.digest, self.members, target)
        token = _token_for(result)
        if all(existing is not target and existing != target for existing, _ in self._links):
            self._links.append((target, token))
        if result is not None:
            root = Path(os.path.realpath(result.raw_root))
            if root not in self._allowed_roots:
                self._allowed_roots.append(root)
        return result

    @property
    def links(self) -> tuple[tuple[LinkTarget, str], ...]:
        """Return the recorded ``(target, token)`` pairs in call order."""
        return tuple(self._links)


def open_context(store: Store, digest: str, member: str) -> DeriveContext:
    """Open a context for ``member`` of the bundle ``digest`` or raise DeriveNotFound."""
    try:
        record = store.get(digest)
    except ValueError as exc:
        raise DeriveNotFound(f"unknown bundle {digest!r}") from exc
    if record is None:
        raise DeriveNotFound(f"unknown bundle {digest!r}")
    if not isinstance(member, str) or _MEMBER_RE.fullmatch(member) is None or member in (".", ".."):
        raise DeriveNotFound(f"unknown member {member!r}")
    for info in record.members:
        if info.get("id") == member:
            return DeriveContext(store, record.digest, info, record.members)
    raise DeriveNotFound(f"unknown member {member!r}")


def params_key(canonical: Mapping[str, str]) -> str:
    """Return the safe cache-key segment for canonical string parameters."""
    if not canonical:
        return NO_PARAMS_KEY
    joined = ",".join(
        f"{key}={urllib.parse.quote(value, safe='')}" for key, value in sorted(canonical.items())
    )
    if len(joined) > MAX_KEY_CHARS:
        raise BadParams("parameters too long")
    return joined


def validate_params(
    deriver: Deriver, ctx: DeriveContext, params: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, str], str]:
    """Return (typed params for derive, canonical string params, params_key)."""
    spec = deriver.spec
    declared = {param.name: param for param in spec.params}
    for key in params:
        if key not in declared:
            raise BadParams(f"undeclared parameter {key!r}")
    typed: dict[str, Any] = {}
    canonical: dict[str, str] = {}
    for name, param in declared.items():
        if name not in params:
            raise BadParams(f"missing parameter {name!r}")
        value = params[name]
        if not isinstance(value, str):
            raise BadParams(f"parameter {name!r} must be a string")
        if param.kind == "int":
            if _INT_RE.fullmatch(value) is None:
                raise BadParams(f"parameter {name!r} is not an integer: {value!r}")
            number = int(value)
            if number < param.min or number > param.max:
                raise BadParams(f"parameter {name!r} out of range: {value!r}")
            typed[name] = number
            canonical[name] = str(number)
        elif param.kind == "float":
            if _FLOAT_RE.fullmatch(value) is None:
                raise BadParams(f"parameter {name!r} is not a number: {value!r}")
            number = float(value)
            if number < param.min or number > param.max:
                raise BadParams(f"parameter {name!r} out of range: {value!r}")
            quotient = round((number - param.min) / param.step)
            snapped = param.min + quotient * param.step + 0.0
            if abs(snapped - number) > 1e-9 * max(1.0, abs(param.step)):
                raise BadParams(f"parameter {name!r} is not a multiple of the step: {value!r}")
            text = format(snapped, ".12g")
            typed[name] = float(text)
            canonical[name] = text
        elif param.kind == "enum":
            if value not in param.values:
                raise BadParams(f"parameter {name!r} is not one of {param.values!r}")
            typed[name] = value
            canonical[name] = value
        else:
            typed[name] = value
            canonical[name] = value
    space = {
        param.name: canonical[param.name] for param in spec.params if param.kind in SPACE_KINDS
    }
    try:
        allowed = deriver.param_space(ctx)
    except (BadParams, DeriveNotFound):
        raise
    except Exception as exc:
        raise DeriveError(
            f"deriver {spec.name!r} param_space failed: {type(exc).__name__}: {exc}"
        ) from exc
    if space not in allowed:
        raise BadParams(f"parameter combination not available: {space}")
    return typed, canonical, params_key(canonical)


def _dumps(obj: Any) -> bytes:
    """Serialise ``obj`` as canonical UTF-8 JSON bytes."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _normalize_json(value: Any, deriver: str, output: str) -> Any:
    """Normalise a JSON output value, raising DeriveError for unsupported values."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            raise DeriveError(f"deriver {deriver!r} output {output!r}: infinity is not allowed")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        return _normalize_json(value.item(), deriver, output)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeriveError(
                    f"deriver {deriver!r} output {output!r}: JSON keys must be strings"
                )
            normalized[key] = _normalize_json(item, deriver, output)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, deriver, output) for item in value]
    raise DeriveError(
        f"deriver {deriver!r} output {output!r}: unsupported type {type(value).__name__}"
    )


def _encode_output(name: str, value: Any, deriver: str) -> bytes:
    """Validate one output and return the exact bytes to write."""
    if _OUTPUT_RE.fullmatch(name) is None:
        raise DeriveError(f"deriver {deriver!r} output name {name!r} is invalid")
    if type(value) is np.ndarray:
        if not name.endswith(".npy"):
            raise DeriveError(f"deriver {deriver!r} output {name!r}: arrays must be named *.npy")
        dtype = value.dtype
        if np.issubdtype(dtype, np.complexfloating):
            raise DeriveError(
                f"deriver {deriver!r} output {name!r}: complex dtype {dtype} is not allowed; "
                "split into real arrays (e.g. magnitude and phase)"
            )
        if dtype.hasobject or dtype.name not in OUTPUT_DTYPES:
            raise DeriveError(f"deriver {deriver!r} output {name!r}: dtype {dtype} is not allowed")
        array = np.ascontiguousarray(value, dtype=dtype.newbyteorder("<"))
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()
    if isinstance(value, (dict, list)):
        if not name.endswith(".json"):
            raise DeriveError(f"deriver {deriver!r} output {name!r}: JSON must be named *.json")
        return _dumps(_normalize_json(value, deriver, name))
    if isinstance(value, bytes):
        if name.endswith(".npy") or name.endswith(".json"):
            raise DeriveError(
                f"deriver {deriver!r} output {name!r}: bytes must not end in .npy or .json"
            )
        return bytes(value)
    raise DeriveError(
        f"deriver {deriver!r} output {name!r}: unsupported type {type(value).__name__}"
    )


def _encode_outputs(outputs: Any, deriver: str) -> list[tuple[str, bytes]]:
    """Validate the whole output mapping and encode every value."""
    if not isinstance(outputs, Mapping):
        raise DeriveError(f"deriver {deriver!r} returned {type(outputs).__name__}, not a mapping")
    if not outputs:
        raise DeriveError(f"deriver {deriver!r} returned no outputs")
    return [(name, _encode_output(name, value, deriver)) for name, value in outputs.items()]


def _file_records(root: Path) -> tuple[DerivedFile, ...]:
    """Return sorted records for the output files in ``root`` (excluding META_FILE)."""
    records: list[DerivedFile] = []
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if path.name == META_FILE:
            continue
        data = path.read_bytes()
        records.append(
            DerivedFile(name=path.name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        )
    return tuple(records)


def _cache_dir(
    store: Store,
    digest: str,
    member: str,
    name: str,
    version: int,
    params_key_: str,
    links_key_: str,
) -> Path:
    """Return the cache directory for one derivation."""
    return store.derived_dir(digest) / member / name / f"v{version}" / params_key_ / links_key_


def _read_meta(directory: Path) -> dict[str, Any] | None:
    """Read a cache META_FILE, returning None when missing or unreadable."""
    try:
        raw = (directory / META_FILE).read_bytes()
        meta = json.loads(raw)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _files_match(directory: Path, files: Any) -> bool:
    """Return True when every recorded file exists with the recorded size."""
    if not isinstance(files, list):
        return False
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return False
        try:
            if (directory / item["name"]).stat().st_size != item["size"]:
                return False
        except (OSError, KeyError, TypeError):
            return False
    return True


def _record_row(
    store: Store,
    digest: str,
    member: str,
    name: str,
    version: int,
    params_key_: str,
    links_key_: str,
    status: str,
    error: str | None,
) -> None:
    """Upsert one `derived` row."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO derived(digest, member, deriver, version, params_key, links_key, "
                "status, error, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(digest, member, deriver, version, params_key, links_key) "
                "DO UPDATE SET status = excluded.status, error = excluded.error, "
                "updated_at = excluded.updated_at",
                (digest, member, name, version, params_key_, links_key_, status, error, _now()),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


def _result(
    store: Store,
    digest: str,
    member: str,
    name: str,
    version: int,
    params: Mapping[str, str],
    params_key_: str,
    links_key_: str,
    files: Sequence[DerivedFile],
    cached: bool,
) -> DerivedResult:
    """Build a DerivedResult for the cache directory implied by the keys."""
    return DerivedResult(
        deriver=name,
        version=version,
        member=member,
        params=dict(params),
        params_key=params_key_,
        links_key=links_key_,
        directory=_cache_dir(store, digest, member, name, version, params_key_, links_key_),
        files=tuple(files),
        cached=cached,
    )


def _find_cached(
    store: Store,
    digest: str,
    member: str,
    deriver: Deriver,
    params_key_: str,
) -> DerivedResult | None:
    """Return a valid cached result for the request, or None."""
    spec = deriver.spec
    record = store.get(digest)
    if record is None:
        return None
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT links_key FROM derived WHERE digest = ? AND member = ? AND deriver = ? "
            "AND version = ? AND params_key = ? AND status = ? "
            "ORDER BY updated_at DESC, links_key",
            (digest, member, spec.name, spec.version, params_key_, STATUS_READY),
        ).fetchall()
    for row in rows:
        recorded_links_key = row["links_key"]
        directory = _cache_dir(
            store, digest, member, spec.name, spec.version, params_key_, recorded_links_key
        )
        meta = _read_meta(directory)
        if meta is None:
            continue
        tokens: list[str] = []
        try:
            for item in meta.get("links", []):
                target = LinkTarget(**item["target"])
                tokens.append(_token_for(_resolve_link(store, digest, record.members, target)))
        except (KeyError, TypeError, ValueError):
            continue
        if links_key(tokens) != recorded_links_key:
            continue
        if not _files_match(directory, meta.get("files", [])):
            continue
        files = tuple(
            DerivedFile(name=item["name"], size=item["size"], sha256=item["sha256"])
            for item in meta.get("files", [])
        )
        return DerivedResult(
            deriver=meta["deriver"],
            version=meta["version"],
            member=meta["member"],
            params=dict(meta["params"]),
            params_key=meta["params_key"],
            links_key=meta["links_key"],
            directory=directory,
            files=files,
            cached=True,
        )
    return None


def get_or_derive(
    store: Store,
    digest: str,
    member: str,
    name: str,
    params: Mapping[str, str],
) -> DerivedResult:
    """Return the cached derivation for a request or compute and cache it."""
    deriver = get_deriver(name)
    ctx = open_context(store, digest, member)
    kind = ctx.member_info.get("kind")
    if kind not in deriver.spec.kinds:
        raise DeriveNotFound(f"deriver {name!r} does not support member kind {kind!r}")
    typed, canonical, params_key_ = validate_params(deriver, ctx, params)
    digest = ctx.digest
    cached = _find_cached(store, digest, member, deriver, params_key_)
    if cached is not None:
        return cached
    return _derive_and_store(store, digest, member, deriver, typed, canonical, params_key_)


def _derive_and_store(
    store: Store,
    digest: str,
    member: str,
    deriver: Deriver,
    typed: Mapping[str, Any],
    canonical: Mapping[str, str],
    params_key_: str,
) -> DerivedResult:
    """Run a deriver, write its outputs atomically and index the result."""
    spec = deriver.spec
    name = spec.name
    version = spec.version
    ctx = open_context(store, digest, member)
    try:
        outputs = deriver.derive(ctx, typed)
    except BadParams:
        raise
    except Exception as exc:
        _record_row(
            store,
            digest,
            member,
            name,
            version,
            params_key_,
            links_key([token for _, token in ctx.links]),
            STATUS_FAILED,
            f"{type(exc).__name__}: {exc}",
        )
        raise DeriveError(f"deriver {name!r} failed: {type(exc).__name__}: {exc}") from exc
    try:
        encoded = _encode_outputs(outputs, name)
    except DeriveError as exc:
        _record_row(
            store,
            digest,
            member,
            name,
            version,
            params_key_,
            links_key([token for _, token in ctx.links]),
            STATUS_FAILED,
            f"{type(exc).__name__}: {exc}",
        )
        raise DeriveError(f"deriver {name!r} failed: {type(exc).__name__}: {exc}") from exc
    links = ctx.links
    links_key_ = links_key([token for _, token in links])
    staging = store.new_staging()
    try:
        for file_name, data in encoded:
            (staging / file_name).write_bytes(data)
        files = _file_records(staging)
        meta = {
            "deriver": name,
            "version": version,
            "member": member,
            "params": dict(canonical),
            "params_key": params_key_,
            "links": [
                {
                    "target": {
                        "member": target.member,
                        "sha256": target.sha256,
                        "kind": target.kind,
                    },
                    "token": token,
                }
                for target, token in links
            ],
            "links_key": links_key_,
            "files": [
                {"name": record.name, "size": record.size, "sha256": record.sha256}
                for record in files
            ],
        }
        (staging / META_FILE).write_bytes(_dumps(meta))
        final = _cache_dir(store, digest, member, name, version, params_key_, links_key_)
        os.makedirs(final.parent, exist_ok=True)
        if os.path.lexists(final):
            moving = store.staging_dir / ("deleting-" + uuid.uuid4().hex)
            os.rename(final, moving)
            _make_writable_and_remove(moving)
        os.rename(staging, final)
    except BaseException:
        store.discard_staging(staging)
        raise
    _record_row(
        store,
        digest,
        member,
        name,
        version,
        params_key_,
        links_key_,
        STATUS_READY,
        None,
    )
    return _result(
        store,
        digest,
        member,
        name,
        version,
        canonical,
        params_key_,
        links_key_,
        files,
        cached=False,
    )


def _safe_params_key(params: Mapping[str, str]) -> str:
    """Return params_key for a best-effort eager outcome."""
    try:
        return params_key(params)
    except BadParams:
        return ""


def derive_eager(store: Store, digest: str) -> list[EagerOutcome]:
    """Run every eager deriver over every supported member, collecting outcomes."""
    try:
        record = store.get(digest)
    except ValueError as exc:
        raise DeriveNotFound(f"unknown bundle {digest!r}") from exc
    if record is None:
        raise DeriveNotFound(f"unknown bundle {digest!r}")
    digest = record.digest
    outcomes: list[EagerOutcome] = []
    for deriver in registered_derivers():
        if not deriver.spec.eager:
            continue
        for info in record.members:
            kind = info.get("kind")
            if kind not in deriver.spec.kinds:
                continue
            member_id = str(info["id"])
            ctx = open_context(store, digest, member_id)
            try:
                space = deriver.param_space(ctx)
            except Exception as exc:
                outcomes.append(
                    EagerOutcome(
                        member=member_id,
                        deriver=deriver.spec.name,
                        params_key="",
                        result=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            for params in space:
                try:
                    result = get_or_derive(store, digest, member_id, deriver.spec.name, params)
                except (DeriveError, BadParams) as exc:
                    outcomes.append(
                        EagerOutcome(
                            member=member_id,
                            deriver=deriver.spec.name,
                            params_key=_safe_params_key(params),
                            result=None,
                            error=str(exc),
                        )
                    )
                else:
                    outcomes.append(
                        EagerOutcome(
                            member=member_id,
                            deriver=deriver.spec.name,
                            params_key=result.params_key,
                            result=result,
                            error=None,
                        )
                    )
    return outcomes


@dataclass(frozen=True)
class DerivedFile:
    """One output file of a derivation."""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class DerivedResult:
    """A derivation result, either freshly computed or served from cache."""

    deriver: str
    version: int
    member: str
    params: Mapping[str, str]
    params_key: str
    links_key: str
    directory: Path
    files: tuple[DerivedFile, ...]
    cached: bool


@dataclass(frozen=True)
class EagerOutcome:
    """One (deriver, member, params) outcome of an eager pass."""

    member: str
    deriver: str
    params_key: str
    result: DerivedResult | None
    error: str | None


# One line per deriver module; each module calls register() at import time.
DERIVER_MODULES: tuple[str, ...] = ("plateau_rt.viewer.derive.overview",)
for _module in DERIVER_MODULES:
    importlib.import_module(_module)
