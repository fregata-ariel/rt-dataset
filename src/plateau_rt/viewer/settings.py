"""Viewer settings loaded from the environment (Sionna-free)."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from plateau_rt.viewer.extract import ExtractLimits

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30

DEFAULT_DATA_DIR = "viewer_data"
DEFAULT_MAX_UPLOAD_BYTES = 4 * GiB
DEFAULT_MAX_EXTRACTED_BYTES = 16 * GiB
DEFAULT_MAX_FILES = 100_000
DEFAULT_MAX_ARRAY_BYTES = 1 * GiB
DEFAULT_DERIVE_TIMEOUT_S = 120.0
DEFAULT_DERIVE_MEM_BYTES = 4 * GiB
DEFAULT_MAX_CONCURRENT_DERIVES = 2
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost")

ENV_VARS: dict[str, str] = {
    "data_dir": "VIEWER_DATA",
    "max_upload_bytes": "VIEWER_MAX_UPLOAD_BYTES",
    "max_extracted_bytes": "VIEWER_MAX_EXTRACTED_BYTES",
    "max_files": "VIEWER_MAX_FILES",
    "max_array_bytes": "VIEWER_MAX_ARRAY_BYTES",
    "import_roots": "VIEWER_IMPORT_ROOTS",
    "derive_timeout_s": "VIEWER_DERIVE_TIMEOUT_S",
    "derive_mem_bytes": "VIEWER_DERIVE_MEM_BYTES",
    "max_concurrent_derives": "VIEWER_MAX_CONCURRENT_DERIVES",
    "allowed_hosts": "VIEWER_ALLOWED_HOSTS",
    "allowed_origins": "VIEWER_ALLOWED_ORIGINS",
    "read_only": "VIEWER_READ_ONLY",
}

_BYTE_RE = re.compile(r"^([0-9]+)(?: ?(KiB|MiB|GiB|TiB))?$")
_UNITS = {"KiB": KiB, "MiB": MiB, "GiB": GiB, "TiB": 1 << 40}
_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")
_FORBIDDEN_HOST_CHARS = (" ", "\t", "\n", "\r", "/", ",")
_ORIGIN_RE = re.compile(r"^https?://[a-z0-9.\-\[\]:]+$")


class SettingsError(ValueError):
    """Invalid viewer configuration (raised at startup)."""


@dataclass(frozen=True)
class ViewerSettings:
    """Validated viewer configuration with environment loading."""

    data_dir: Path = Path(DEFAULT_DATA_DIR)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES
    max_files: int = DEFAULT_MAX_FILES
    max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES
    import_roots: tuple[Path, ...] = ()
    derive_timeout_s: float = DEFAULT_DERIVE_TIMEOUT_S
    derive_mem_bytes: int = DEFAULT_DERIVE_MEM_BYTES
    max_concurrent_derives: int = DEFAULT_MAX_CONCURRENT_DERIVES
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    allowed_origins: tuple[str, ...] = ()
    read_only: bool = False

    def __post_init__(self) -> None:
        for field in (
            "max_upload_bytes",
            "max_extracted_bytes",
            "max_files",
            "max_array_bytes",
            "derive_mem_bytes",
            "max_concurrent_derives",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SettingsError(
                    f"{field} ({ENV_VARS[field]}) must be a positive integer, got {value!r}"
                )
        if (
            isinstance(self.derive_timeout_s, bool)
            or not isinstance(self.derive_timeout_s, (int, float))
            or not math.isfinite(self.derive_timeout_s)
            or self.derive_timeout_s <= 0
        ):
            raise SettingsError(
                f"derive_timeout_s ({ENV_VARS['derive_timeout_s']}) must be a finite "
                f"positive number, got {self.derive_timeout_s!r}"
            )
        if not isinstance(self.data_dir, Path):
            raise SettingsError(f"data_dir ({ENV_VARS['data_dir']}) must be a Path")
        if self.data_dir.exists() and not self.data_dir.is_dir():
            raise SettingsError(
                f"data_dir ({ENV_VARS['data_dir']}) must be a directory, got {self.data_dir}"
            )
        if not isinstance(self.import_roots, tuple):
            raise SettingsError(f"import_roots ({ENV_VARS['import_roots']}) must be a tuple")
        seen: set[str] = set()
        for root in self.import_roots:
            if not isinstance(root, Path):
                raise SettingsError(
                    f"import_roots ({ENV_VARS['import_roots']}) must contain Path objects"
                )
            if not root.is_absolute() or not root.is_dir():
                raise SettingsError(
                    f"import_roots ({ENV_VARS['import_roots']}) must be absolute existing "
                    f"directories, got {root}"
                )
            key = os.path.normpath(str(root))
            if key in seen:
                raise SettingsError(
                    f"import_roots ({ENV_VARS['import_roots']}) contains a duplicate: {root}"
                )
            seen.add(key)
        if not isinstance(self.allowed_hosts, tuple) or not self.allowed_hosts:
            raise SettingsError(f"allowed_hosts ({ENV_VARS['allowed_hosts']}) must be non-empty")
        for host in self.allowed_hosts:
            if (
                not isinstance(host, str)
                or not host
                or any(char in host for char in _FORBIDDEN_HOST_CHARS)
            ):
                raise SettingsError(
                    f"allowed_hosts ({ENV_VARS['allowed_hosts']}) has an invalid entry {host!r}"
                )
        if not isinstance(self.allowed_origins, tuple):
            raise SettingsError(f"allowed_origins ({ENV_VARS['allowed_origins']}) must be a tuple")
        for origin in self.allowed_origins:
            if not isinstance(origin, str) or _ORIGIN_RE.fullmatch(origin) is None:
                raise SettingsError(
                    f"allowed_origins ({ENV_VARS['allowed_origins']}) has an invalid entry "
                    f"{origin!r}"
                )
        if not isinstance(self.read_only, bool):
            raise SettingsError(
                f"read_only ({ENV_VARS['read_only']}) must be a bool, got {self.read_only!r}"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ViewerSettings:
        """Build settings from ``environ`` (defaults to ``os.environ``)."""
        env = os.environ if environ is None else environ
        kwargs: dict[str, object] = {}
        data = _value(env, "data_dir")
        kwargs["data_dir"] = Path(data or DEFAULT_DATA_DIR).expanduser().resolve()
        byte_fields = (
            "max_upload_bytes",
            "max_extracted_bytes",
            "max_array_bytes",
            "derive_mem_bytes",
        )
        defaults = {
            "max_upload_bytes": DEFAULT_MAX_UPLOAD_BYTES,
            "max_extracted_bytes": DEFAULT_MAX_EXTRACTED_BYTES,
            "max_array_bytes": DEFAULT_MAX_ARRAY_BYTES,
            "derive_mem_bytes": DEFAULT_DERIVE_MEM_BYTES,
        }
        for field in byte_fields:
            raw = _value(env, field)
            kwargs[field] = defaults[field] if raw is None else _parse_bytes(raw, field)
        files = _value(env, "max_files")
        kwargs["max_files"] = DEFAULT_MAX_FILES if files is None else _parse_int(files, "max_files")
        concurrency = _value(env, "max_concurrent_derives")
        kwargs["max_concurrent_derives"] = (
            DEFAULT_MAX_CONCURRENT_DERIVES
            if concurrency is None
            else _parse_int(concurrency, "max_concurrent_derives")
        )
        timeout = _value(env, "derive_timeout_s")
        kwargs["derive_timeout_s"] = (
            DEFAULT_DERIVE_TIMEOUT_S
            if timeout is None
            else _parse_float(timeout, "derive_timeout_s")
        )
        roots = _value(env, "import_roots")
        kwargs["import_roots"] = () if roots is None else _parse_roots(roots)
        hosts = _value(env, "allowed_hosts")
        kwargs["allowed_hosts"] = DEFAULT_ALLOWED_HOSTS if hosts is None else _parse_hosts(hosts)
        origins = _value(env, "allowed_origins")
        kwargs["allowed_origins"] = () if origins is None else _parse_origins(origins)
        read_only = _value(env, "read_only")
        kwargs["read_only"] = False if read_only is None else _parse_bool(read_only)
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def extract_limits(self) -> ExtractLimits:
        """Return the extraction caps implied by these settings."""
        return ExtractLimits(max_files=self.max_files, max_extracted_bytes=self.max_extracted_bytes)


def _value(env: Mapping[str, str], field: str) -> str | None:
    """Return the stripped value of ``field``'s env var, or None when unset/blank."""
    raw = env.get(ENV_VARS[field])
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _parse_bytes(raw: str, field: str) -> int:
    """Parse a decimal byte count with an optional binary suffix."""
    match = _BYTE_RE.match(raw)
    if match is None:
        raise SettingsError(f"{field} ({ENV_VARS[field]}) must be a byte size, got {raw!r}")
    number = int(match.group(1))
    unit = match.group(2)
    return number if unit is None else number * _UNITS[unit]


def _parse_int(raw: str, field: str) -> int:
    """Parse a decimal integer env var."""
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(
            f"{field} ({ENV_VARS[field]}) must be a decimal integer, got {raw!r}"
        ) from exc


def _parse_float(raw: str, field: str) -> float:
    """Parse a float env var."""
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"{field} ({ENV_VARS[field]}) must be a number, got {raw!r}") from exc


def _parse_roots(raw: str) -> tuple[Path, ...]:
    """Parse an ``os.pathsep``-separated list of absolute import roots."""
    roots: list[Path] = []
    for item in raw.split(os.pathsep):
        if not item:
            continue
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            raise SettingsError(
                f"import_roots ({ENV_VARS['import_roots']}) entries must be absolute, got {item!r}"
            )
        roots.append(candidate.resolve())
    return tuple(roots)


def _parse_hosts(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated host allow-list."""
    hosts: list[str] = []
    for item in raw.split(","):
        host = item.strip()
        if not host:
            raise SettingsError(f"allowed_hosts ({ENV_VARS['allowed_hosts']}) has an empty entry")
        hosts.append(host)
    return tuple(hosts)


def _parse_origins(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated origin allow-list, lower-cased with one trailing slash removed."""
    origins: list[str] = []
    for item in raw.split(","):
        origin = item.strip()
        if not origin:
            raise SettingsError(
                f"allowed_origins ({ENV_VARS['allowed_origins']}) has an empty entry"
            )
        origin = origin.lower()
        if origin.endswith("/"):
            origin = origin[:-1]
        origins.append(origin)
    return tuple(origins)


def _parse_bool(raw: str) -> bool:
    """Parse a case-insensitive boolean env var."""
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise SettingsError(f"read_only ({ENV_VARS['read_only']}) must be a boolean, got {raw!r}")
