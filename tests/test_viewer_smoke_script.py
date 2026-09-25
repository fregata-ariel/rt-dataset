"""Tests for the light viewer-image smoke script (``scripts/ci/viewer_smoke.py``)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from viewer_bundle_fixtures import (
    BROKEN_CASES,
    make_archive,
    write_broken_dataset,
    write_bundle_dir,
    write_fixture_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT = REPO_ROOT / "scripts" / "ci" / "viewer_smoke.py"


def _load_script() -> ModuleType:
    """Load ``viewer_smoke.py`` by path without relying on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("viewer_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


def _env() -> dict[str, str]:
    """Return the environment for a viewer subprocess with the repo ``src`` first."""
    env = dict(os.environ)
    prefix = str(SRC_DIR)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = prefix if existing is None else prefix + os.pathsep + existing
    return env


def _free_port() -> int:
    """Return a currently free localhost TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_find_forbidden_modules() -> None:
    """An importable stdlib module is reported, a missing name is not."""
    assert smoke.find_forbidden_modules(["json", "definitely_not_a_module_xyz"]) == ["json"]


def test_find_forbidden_distributions() -> None:
    """Distribution names are normalised before matching the forbidden lists."""
    assert smoke.find_forbidden_distributions(
        [
            "numpy",
            "Sionna_RT",
            "nvidia-cublas-cu12",
            "matplotlib",
            "cuda-python",
            "fastapi",
            "drjit",
        ]
    ) == ["cuda-python", "drjit", "matplotlib", "nvidia-cublas-cu12", "sionna-rt"]
    assert smoke.find_forbidden_distributions(["numpy"]) == []


def test_find_cuda_env() -> None:
    """Only the known CUDA variables are reported."""
    assert smoke.find_cuda_env({"CUDA_VERSION": "12.8", "PATH": "/bin"}) == ["CUDA_VERSION"]
    assert smoke.find_cuda_env({}) == []


def test_find_cuda_libraries(tmp_path: Path) -> None:
    """CUDA-looking files, a broken symlink and local/cuda are reported exactly."""
    lib = tmp_path / "lib"
    (lib / "x86_64-linux-gnu").mkdir(parents=True)
    (lib / "x86_64-linux-gnu" / "libcudart.so.12").write_bytes(b"x")
    (lib / "libcuda.so.1").write_bytes(b"x")
    (lib / "libnvidia-ml.so").symlink_to("missing")
    (lib / "libcurl.so.4").write_bytes(b"x")
    (lib / "libatomic.so.1").write_bytes(b"x")
    (tmp_path / "local" / "cuda").mkdir(parents=True)
    (tmp_path / "share" / "cuda").mkdir(parents=True)
    assert smoke.find_cuda_libraries([tmp_path]) == sorted(
        [
            str(lib / "x86_64-linux-gnu" / "libcudart.so.12"),
            str(lib / "libcuda.so.1"),
            str(lib / "libnvidia-ml.so"),
            str(tmp_path / "local" / "cuda"),
        ]
    )
    assert smoke.find_cuda_libraries([tmp_path / "nope"]) == []


def test_find_import_failures() -> None:
    """Exactly the unimportable module is reported with its name."""
    failures = smoke.find_import_failures(["json", "definitely_not_a_module_xyz"])
    assert len(failures) == 1
    assert "definitely_not_a_module_xyz" in failures[0]


def test_image_check_fails_where_heavy_stack_is_installed() -> None:
    """In the prod CI image the heavy stack is present, so image-check fails."""
    if importlib.util.find_spec("sionna") is None:
        pytest.skip("heavy stack is not installed here")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "image-check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("sionna" in problem for problem in payload["problems"])


@pytest.fixture(scope="module")
def server_url(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Start a real viewer server once per module and yield its base URL."""
    port = _free_port()
    data = tmp_path_factory.mktemp("smoke-server-data")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "plateau_rt.viewer",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--data",
            str(data),
        ],
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise AssertionError(f"server exited early: {stderr}")
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1)
            except httpx.HTTPError:
                time.sleep(0.2)
                continue
            if response.status_code == 200:
                break
            time.sleep(0.2)
        else:
            raise AssertionError("server never became healthy")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(scope="module")
def good_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a valid fixture bundle zip once per module."""
    root = tmp_path_factory.mktemp("smoke-good") / "bundle"
    write_fixture_bundle(root)
    out = tmp_path_factory.mktemp("smoke-good-zip") / "bundle.zip"
    make_archive(root, out, "zip")
    return out


@pytest.fixture(scope="module")
def broken_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a broken fixture bundle zip once per module."""
    root = tmp_path_factory.mktemp("smoke-broken") / "bundle"
    write_broken_dataset(root / "dataset", BROKEN_CASES[0])
    write_bundle_dir(
        root,
        members=[{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}],
    )
    out = tmp_path_factory.mktemp("smoke-broken-zip") / "broken.zip"
    make_archive(root, out, "zip")
    return out


def _run_http(base_url: str, archive: Path) -> subprocess.CompletedProcess[str]:
    """Run the smoke ``http`` command against ``base_url`` with ``archive``."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "http", "--base-url", base_url, "--archive", str(archive)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_http_smoke_flow(server_url: str, good_archive: Path, broken_archive: Path) -> None:
    """Upload, duplicate, broken and unreachable flows answer as documented, in order."""
    first = _run_http(server_url, good_archive)
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert re.fullmatch(r"[0-9a-f]{64}", payload["digest"]) is not None
    assert payload["member"] == "dataset"
    assert payload["overview"]["num_views"] > 0

    duplicate = _run_http(server_url, good_archive)
    assert duplicate.returncode == 1
    assert "viewer smoke failed" in duplicate.stderr

    broken = _run_http(server_url, broken_archive)
    assert broken.returncode == 1
    assert "upload" in broken.stderr
    assert "400" in broken.stderr

    closed = _run_http(f"http://127.0.0.1:{_free_port()}", good_archive)
    assert closed.returncode == 1
    assert "Traceback" not in closed.stderr


def test_main_without_command() -> None:
    """Calling ``main`` with no command prints help and returns 2."""
    assert smoke.main([]) == 2
