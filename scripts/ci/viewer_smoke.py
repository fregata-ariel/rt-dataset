"""Smoke checks for the light (CUDA-free, Sionna-free) viewer image.

Two commands, both usable with the repository checked out or with this file
fed on stdin (``python - < viewer_smoke.py ...``), so nothing here relies on
``__file__``:

- ``image-check``: assert the image contains no heavy GPU stack (no importable
  ``sionna``/``mitsuba``/``drjit``/``matplotlib``, none of their distributions
  nor any ``nvidia-*``/``cuda-*`` distributions, no CUDA environment variables
  or libraries) and that the viewer modules import. Prints a JSON summary.
- ``http``: exercise a running viewer (upload a fixture archive, derive the
  ``overview`` of its ``rf_dataset`` member, fetch ``overview.json``, wait for
  the bundle status to become complete). Prints a JSON summary.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FORBIDDEN_MODULES: tuple[str, ...] = ("sionna", "mitsuba", "drjit", "matplotlib")
FORBIDDEN_DISTRIBUTIONS: tuple[str, ...] = (
    "sionna",
    "sionna-rt",
    "mitsuba",
    "drjit",
    "matplotlib",
)
FORBIDDEN_DISTRIBUTION_PREFIXES: tuple[str, ...] = ("nvidia-", "cuda-")
CUDA_ENV_VARS: tuple[str, ...] = (
    "CUDA_VERSION",
    "NV_CUDA_LIB_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
)
CUDA_LIBRARY_PATTERNS: tuple[str, ...] = (
    "libcuda.*",
    "libcuda-*",
    "libcudart*",
    "libcublas*",
    "libcudnn*",
    "libcufft*",
    "libcurand*",
    "libcusolver*",
    "libcusparse*",
    "libnvrtc*",
    "libnvJitLink*",
    "libnvidia-*",
    "libnvoptix*",
)
LIBRARY_ROOTS: tuple[str, ...] = ("/usr", "/lib", "/lib64", "/opt")
VIEWER_MODULES: tuple[str, ...] = ("plateau_rt.viewer.api.app", "plateau_rt.viewer.__main__")

_REQUEST_TIMEOUT_S = 30.0
_BODY_PREVIEW_CHARS = 500


class SmokeError(Exception):
    """A failed smoke expectation (step, status and body preview)."""


def find_forbidden_modules(names: Iterable[str] = FORBIDDEN_MODULES) -> list[str]:
    """Return the names in ``names`` that are currently importable."""
    found: list[str] = []
    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is not None:
            found.append(name)
    return found


def find_forbidden_distributions(dist_names: Iterable[str] | None = None) -> list[str]:
    """Return sorted forbidden distribution names among ``dist_names``."""
    if dist_names is None:
        raw_names: list[str] = []
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            if name:
                raw_names.append(str(name))
        dist_names = raw_names
    bad: set[str] = set()
    for raw in dist_names:
        normalised = str(raw).lower().replace("_", "-").replace(".", "-")
        if normalised in FORBIDDEN_DISTRIBUTIONS or normalised.startswith(
            FORBIDDEN_DISTRIBUTION_PREFIXES
        ):
            bad.add(normalised)
    return sorted(bad)


def find_cuda_env(environ: Mapping[str, str] | None = None) -> list[str]:
    """Return the sorted CUDA environment variables present in ``environ``."""
    env = os.environ if environ is None else environ
    return sorted(name for name in CUDA_ENV_VARS if name in env)


def find_cuda_libraries(roots: Iterable[str | Path] = LIBRARY_ROOTS) -> list[str]:
    """Return sorted paths of CUDA-looking libraries under ``roots``."""
    found: set[str] = set()
    for root in roots:
        root_str = os.fspath(root)
        if not os.path.isdir(root_str):
            continue
        for dirpath, dirnames, filenames in os.walk(
            root_str, followlinks=False, onerror=lambda _exc: None
        ):
            for filename in filenames:
                if any(fnmatch.fnmatchcase(filename, pattern) for pattern in CUDA_LIBRARY_PATTERNS):
                    found.add(os.path.join(dirpath, filename))
            if os.path.basename(dirpath) == "local":
                for dirname in dirnames:
                    if dirname == "cuda" or fnmatch.fnmatchcase(dirname, "cuda-*"):
                        found.add(os.path.join(dirpath, dirname))
    return sorted(found)


def find_import_failures(modules: Iterable[str] = VIEWER_MODULES) -> list[str]:
    """Import each module, returning ``"<module>: <ExcType>: <msg>"`` failures."""
    failures: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - any import failure is reported
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def image_problems() -> list[str]:
    """Combine every image check into human-readable problem lines."""
    problems: list[str] = []
    for name in find_forbidden_modules():
        problems.append(f"module importable: {name}")
    for name in find_forbidden_distributions():
        problems.append(f"distribution installed: {name}")
    for name in find_cuda_env():
        problems.append(f"CUDA environment variable: {name}")
    for path in find_cuda_libraries():
        problems.append(f"CUDA library: {path}")
    for failure in find_import_failures():
        problems.append(f"viewer import failed: {failure}")
    return problems


def run_image_check() -> int:
    """Print the JSON image-check summary; return 0 when clean, else 1."""
    problems = image_problems()
    print(json.dumps({"ok": not problems, "problems": problems}, indent=2, sort_keys=True))
    return 0 if not problems else 1


def _short(body: bytes) -> str:
    """Decode ``body`` and truncate it to a short preview."""
    return body.decode("utf-8", "replace")[:_BODY_PREVIEW_CHARS]


def _request(
    method: str, url: str, *, body: bytes | None = None, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    """Send one HTTP request, returning ``(status, headers, body)``."""
    request = urllib.request.Request(url, data=body, headers=dict(headers or {}), method=method)
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return response.status, response_headers, response.read()
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        except Exception:  # noqa: BLE001 - a best-effort error body is enough
            payload = b""
        error_headers = {key.lower(): value for key, value in (exc.headers or {}).items()}
        return exc.code, error_headers, payload
    except Exception as exc:  # noqa: BLE001 - connection errors become SmokeError
        raise SmokeError(f"request {method} {url} failed: {exc}") from exc


def _join(base_url: str, url: str) -> str:
    """Join a possibly relative server URL with ``base_url``."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return base_url.rstrip("/") + "/" + url.lstrip("/")


def _load_json(step: str, status: int, body: bytes) -> Any:
    """Parse ``body`` as JSON, raising a step-labelled ``SmokeError`` on failure."""
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SmokeError(f"{step} failed: status {status}: invalid JSON: {exc}") from exc


def run_http_smoke(
    base_url: str,
    archive: Path,
    *,
    name: str = "viewer-smoke",
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Upload ``archive`` and drive the viewer API to a complete bundle."""
    base = base_url.rstrip("/")
    try:
        archive_bytes = Path(archive).read_bytes()
    except OSError as exc:
        raise SmokeError(f"read archive failed: {exc}") from exc

    status, _, body = _request("GET", base + "/api/health")
    if status != 200:
        raise SmokeError(f"GET /api/health failed: status {status}: {_short(body)}")
    health = _load_json("GET /api/health", status, body)
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise SmokeError(f"GET /api/health failed: status {status}: {_short(body)}")

    status, headers, body = _request("GET", base + "/")
    if status != 200:
        raise SmokeError(f"GET / failed: status {status}: {_short(body)}")
    if not headers.get("content-type", "").startswith("text/html"):
        raise SmokeError(f"GET / failed: status {status}: expected text/html: {_short(body)}")

    upload_url = base + "/api/bundles/upload?name=" + urllib.parse.quote(name, safe="")
    status, _, body = _request(
        "PUT",
        upload_url,
        body=archive_bytes,
        headers={"Content-Type": "application/octet-stream", "X-Viewer-Request": "1"},
    )
    if status != 201:
        raise SmokeError(f"PUT /api/bundles/upload failed: status {status}: {_short(body)}")
    upload = _load_json("PUT /api/bundles/upload", status, body)
    if not isinstance(upload, dict) or upload.get("created") is not True:
        raise SmokeError(
            f"PUT /api/bundles/upload failed: status {status}: expected created true: "
            f"{_short(body)}"
        )
    members = upload.get("members")
    if not isinstance(members, list):
        raise SmokeError(
            f"PUT /api/bundles/upload failed: status {status}: missing members: {_short(body)}"
        )
    rf_member = next(
        (m for m in members if isinstance(m, dict) and m.get("kind") == "rf_dataset"), None
    )
    if rf_member is None:
        raise SmokeError(
            f"PUT /api/bundles/upload failed: status {status}: no rf_dataset member: {_short(body)}"
        )
    digest = upload.get("digest")
    member = rf_member.get("id")
    if not isinstance(digest, str) or not digest or not isinstance(member, str) or not member:
        raise SmokeError(
            f"PUT /api/bundles/upload failed: status {status}: missing digest/member: "
            f"{_short(body)}"
        )

    status, _, body = _request("GET", base + "/api/bundles")
    if status != 200:
        raise SmokeError(f"GET /api/bundles failed: status {status}: {_short(body)}")
    listing = _load_json("GET /api/bundles", status, body)
    bundles = listing.get("bundles") if isinstance(listing, dict) else None
    digests = [b.get("digest") for b in bundles] if isinstance(bundles, list) else []
    if digest not in digests:
        raise SmokeError(
            f"GET /api/bundles failed: status {status}: digest {digest} not listed: {_short(body)}"
        )

    derive_url = f"{base}/api/bundles/{digest}/members/{member}/derived/overview"
    status, _, body = _request("GET", derive_url)
    if status == 200:
        payload = _load_json("GET derived/overview", status, body)
    elif status == 202:
        accepted = _load_json("GET derived/overview", status, body)
        job_id = accepted.get("job_id") if isinstance(accepted, dict) else None
        status_url = accepted.get("status_url") if isinstance(accepted, dict) else None
        if not job_id or not status_url:
            raise SmokeError(
                "GET derived/overview failed: status 202: missing job_id/status_url: "
                f"{_short(body)}"
            )
        payload = _poll_job(base, derive_url, str(status_url), timeout_s)
    else:
        raise SmokeError(f"GET derived/overview failed: status {status}: {_short(body)}")
    if not isinstance(payload, dict):
        raise SmokeError(f"GET derived/overview failed: expected a JSON object: {_short(body)}")

    files = payload.get("files")
    overview_entry = (
        next(
            (f for f in files if isinstance(f, dict) and f.get("name") == "overview.json"),
            None,
        )
        if isinstance(files, list)
        else None
    )
    if overview_entry is None or not overview_entry.get("url"):
        raise SmokeError(f"GET derived/overview failed: no overview.json file: {_short(body)}")
    status, file_headers, body = _request("GET", _join(base, str(overview_entry["url"])))
    if status != 200:
        raise SmokeError(f"GET overview.json failed: status {status}: {_short(body)}")
    if "etag" not in file_headers:
        raise SmokeError(f"GET overview.json failed: status {status}: missing ETag: {_short(body)}")
    overview = _load_json("GET overview.json", status, body)
    if not isinstance(overview, dict) or overview.get("member") != member:
        raise SmokeError(f"GET overview.json failed: wrong member: {_short(body)}")
    num_views = overview.get("num_views")
    num_bs = overview.get("num_bs")
    if (
        not isinstance(num_views, int)
        or isinstance(num_views, bool)
        or num_views <= 0
        or not isinstance(num_bs, int)
        or isinstance(num_bs, bool)
        or num_bs <= 0
    ):
        raise SmokeError(f"GET overview.json failed: bad num_views/num_bs: {_short(body)}")

    _poll_status(base, digest, timeout_s)

    return {
        "base_url": base_url,
        "digest": digest,
        "member": member,
        "overview": {"num_views": num_views, "num_bs": num_bs},
        "health": health,
    }


def _poll_job(base: str, derive_url: str, status_url: str, timeout_s: float) -> dict[str, Any]:
    """Poll a derivation job until done, returning its result payload."""
    job_url = _join(base, status_url)
    deadline = time.monotonic() + timeout_s
    while True:
        status, _, body = _request("GET", job_url)
        if status != 200:
            raise SmokeError(f"GET {status_url} failed: status {status}: {_short(body)}")
        job = _load_json(f"GET {status_url}", status, body)
        job_status = job.get("status") if isinstance(job, dict) else None
        if job_status == "done":
            result = job.get("result") if isinstance(job, dict) else None
            if result is None:
                status, _, body = _request("GET", derive_url)
                if status != 200:
                    raise SmokeError(
                        f"GET derived/overview failed: status {status}: {_short(body)}"
                    )
                return _load_json("GET derived/overview", status, body)
            if not isinstance(result, dict):
                raise SmokeError(f"GET {status_url} failed: bad job result: {_short(body)}")
            return result
        if job_status == "failed":
            reason = job.get("reason") if isinstance(job, dict) else None
            error = job.get("error") if isinstance(job, dict) else None
            raise SmokeError(f"GET {status_url} failed: job failed: {reason}: {error}")
        if time.monotonic() > deadline:
            raise SmokeError(f"GET {status_url} failed: job did not finish within {timeout_s} s")
        time.sleep(0.5)


def _poll_status(base: str, digest: str, timeout_s: float) -> Any:
    """Poll the bundle status route until the bundle is complete."""
    status_route = f"/api/bundles/{digest}/status"
    deadline = time.monotonic() + timeout_s
    while True:
        status, _, body = _request("GET", base + status_route)
        if status != 200:
            raise SmokeError(f"GET {status_route} failed: status {status}: {_short(body)}")
        payload = _load_json(f"GET {status_route}", status, body)
        if isinstance(payload, dict) and payload.get("complete") is True:
            return payload
        eager = payload.get("eager") if isinstance(payload, dict) else None
        failed = eager.get("failed", 0) if isinstance(eager, dict) else 0
        if isinstance(failed, int) and failed > 0:
            raise SmokeError(f"GET {status_route} failed: eager derivations failed: {_short(body)}")
        if time.monotonic() > deadline:
            raise SmokeError(
                f"GET {status_route} failed: bundle not complete within {timeout_s} s: "
                f"{_short(body)}"
            )
        time.sleep(0.5)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Smoke checks for the light viewer image.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("image-check", help="check the image has no heavy GPU stack")
    http_parser = sub.add_parser("http", help="exercise a running viewer over HTTP")
    http_parser.add_argument("--base-url", required=True, help="viewer base URL")
    http_parser.add_argument(
        "--archive", required=True, type=Path, help="fixture archive to upload"
    )
    http_parser.add_argument("--name", default="viewer-smoke", help="bundle name for the upload")
    http_parser.add_argument("--timeout", default=120.0, type=float, help="poll timeout in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke CLI; return the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "image-check":
        return run_image_check()
    if args.command == "http":
        try:
            result = run_http_smoke(
                args.base_url, args.archive, name=args.name, timeout_s=args.timeout
            )
        except SmokeError as exc:
            print(f"viewer smoke failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    parser.print_help(file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
