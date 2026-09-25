"""Build and dataset provenance (stdlib only, Sionna-free).

Records the Python version, installed package versions, git commits and the
invoking command so that generated manifests stay reproducible without
importing any heavy dependency.
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PROVENANCE_PACKAGES: tuple[tuple[str, str], ...] = (
    ("sionna_rt", "sionna-rt"),
    ("mitsuba", "mitsuba"),
    ("drjit", "drjit"),
    ("numpy", "numpy"),
)
SIONNA_COMMIT_ENV = "PLATEAU_RT_SIONNA_COMMIT"
SIONNA_SUBMODULE_PATH = Path("third_party") / "sionna-rt"
DEFAULT_REPO_ROOT: Path = Path(__file__).resolve().parents[3]

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SIONNA_ENV_RE = re.compile(r"^[0-9a-f]{7,40}$")


def package_version(distribution: str) -> str | None:
    """Return the installed version of ``distribution`` without importing it."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _resolve_gitdir(checkout: Path) -> Path | None:
    """Resolve the git directory of ``checkout`` (worktrees included)."""
    dotgit = checkout / ".git"
    try:
        if dotgit.is_dir():
            return dotgit
        if dotgit.is_file():
            content = dotgit.read_text(encoding="utf-8").strip()
            if not content.startswith("gitdir:"):
                return None
            gitdir = content[len("gitdir:") :].strip()
            path = Path(gitdir)
            if not path.is_absolute():
                path = checkout / path
            return path
        return None
    except (OSError, UnicodeDecodeError):
        return None


def _read_sha_file(path: Path) -> str | None:
    """Return the stripped 40-hex content of ``path``, else None."""
    try:
        content = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if _SHA40_RE.match(content):
        return content
    return None


def _lookup_packed_ref(packed_refs: Path, ref: str) -> str | None:
    """Return the sha for ``ref`` in a packed-refs file, else None."""
    try:
        text = packed_refs.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("^"):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[1] == ref and _SHA40_RE.match(parts[0]):
            return parts[0]
    return None


def git_head_commit(checkout: Path) -> str | None:
    """Return the HEAD commit sha of a git checkout without a git subprocess."""
    try:
        checkout = Path(checkout)
        gitdir = _resolve_gitdir(checkout)
        if gitdir is None or not gitdir.is_dir():
            return None
        try:
            head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        if _SHA40_RE.match(head):
            return head
        if not head.startswith("ref:"):
            return None
        ref = head[len("ref:") :].strip()
        candidates: list[Path] = [gitdir]
        commondir = gitdir / "commondir"
        try:
            if commondir.is_file():
                name = commondir.read_text(encoding="utf-8").strip()
                common = Path(name)
                if not common.is_absolute():
                    common = gitdir / common
                candidates.append(common)
        except (OSError, UnicodeDecodeError):
            return None
        for candidate in candidates:
            loose = candidate / ref
            try:
                if loose.is_file():
                    sha = _read_sha_file(loose)
                    if sha is not None:
                        return sha
            except OSError:
                continue
            packed = candidate / "packed-refs"
            try:
                if packed.is_file():
                    sha = _lookup_packed_ref(packed, ref)
                    if sha is not None:
                        return sha
            except OSError:
                continue
        return None
    except (OSError, UnicodeDecodeError):
        return None


def sionna_rt_commit(
    *,
    repo_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Return the Sionna fork commit and where it was found."""
    root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT
    env = environ if environ is not None else os.environ
    raw = env.get(SIONNA_COMMIT_ENV)
    if raw is not None and raw != "":
        value = raw.strip().lower()
        if not _SIONNA_ENV_RE.match(value):
            raise ValueError(f"{SIONNA_COMMIT_ENV} must be 7-40 hex chars, got {raw!r}")
        return (value, f"env:{SIONNA_COMMIT_ENV}")
    commit = git_head_commit(root / SIONNA_SUBMODULE_PATH)
    if commit is not None:
        return (commit, "submodule_checkout")
    return (None, None)


def collect_provenance(
    *,
    argv: Sequence[str] | None = None,
    now: datetime | None = None,
    repo_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect the provenance payload recorded in build and dataset manifests."""
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    generated = moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    args = list(sys.argv) if argv is None else [str(item) for item in argv]
    root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT
    env = environ if environ is not None else os.environ
    sionna_commit, sionna_source = sionna_rt_commit(repo_root=root, environ=env)
    packages = {key: package_version(dist) for key, dist in PROVENANCE_PACKAGES}
    return {
        "generated_at_utc": generated,
        "python_version": platform.python_version(),
        "packages": packages,
        "sionna_rt_commit": sionna_commit,
        "sionna_rt_commit_source": sionna_source,
        "plateau_rt_commit": git_head_commit(root),
        "command": shlex.join(args),
        "argv": list(args),
    }
