"""Tests for the headless viewer CLI (``python -m plateau_rt.viewer``)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from viewer_bundle_fixtures import (
    BundleFixture,
    make_archive,
    write_broken_dataset,
    write_fixture_bundle,
)
from viewer_job_derivers import Boom, EnumDeriver, SlowDeriver

from plateau_rt.viewer.__main__ import main
from plateau_rt.viewer.derive import registered

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> BundleFixture:
    """Build the canonical v3 fixture bundle once per module."""
    root = tmp_path_factory.mktemp("cli-bundle") / "bundle"
    return write_fixture_bundle(root, schema_version=3)


@pytest.fixture(scope="module")
def zip_path(bundle: BundleFixture, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a zip of the fixture bundle once per module."""
    out = tmp_path_factory.mktemp("cli-zip") / "bundle.zip"
    make_archive(bundle.root, out, "zip")
    return out


def _env() -> dict[str, str]:
    """Return the environment for a CLI subprocess with the repo ``src`` on ``PYTHONPATH``."""
    env = dict(os.environ)
    prefix = str(SRC_DIR)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = prefix if existing is None else prefix + os.pathsep + existing
    return env


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the viewer CLI in a subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, "-m", "plateau_rt.viewer", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=_env(),
    )


def _free_port() -> int:
    """Return a currently free localhost TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ingest_via_main(data: Path, source: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Run the ingest command in-process and return the digest."""
    code = main(["ingest", str(source), "--data", str(data)])
    assert code == 0
    return json.loads(capsys.readouterr().out)["digest"]


def test_ingest_directory_and_archive(
    tmp_path: Path, bundle: BundleFixture, zip_path: Path
) -> None:
    """Ingesting a directory twice and a zip of it share one digest."""
    data = tmp_path / "data"
    first = _run("ingest", str(bundle.root), "--data", str(data))
    assert first.returncode == 0
    body = json.loads(first.stdout)
    assert len(body["digest"]) == 64
    assert body["created"] is True
    again = _run("ingest", str(bundle.root), "--data", str(data))
    assert again.returncode == 0
    assert json.loads(again.stdout)["created"] is False
    archive = _run("ingest", str(zip_path), "--data", str(data))
    assert archive.returncode == 0
    assert json.loads(archive.stdout)["digest"] == body["digest"]


def test_ingest_errors(tmp_path: Path) -> None:
    """Broken and missing inputs exit 1 with the documented error types."""
    data = tmp_path / "data"
    broken = tmp_path / "broken"
    write_broken_dataset(broken, "duplicate_view_id")
    response = _run("ingest", str(broken), "--data", str(data))
    assert response.returncode == 1
    assert json.loads(response.stdout)["error"]["type"] == "validation"
    missing = _run("ingest", str(tmp_path / "nope"), "--data", str(data))
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["error"]["type"] == "not_found"


def test_derive_all_lazy_cached(
    tmp_path: Path, bundle: BundleFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """``derive --all-lazy`` runs the eager deriver and reports it cached on a second run."""
    data = tmp_path / "data"
    digest = _ingest_via_main(data, bundle.root, capsys)
    first = _run("derive", digest, "--all-lazy", "--data", str(data))
    assert first.returncode == 0, first.stdout + first.stderr
    payload = json.loads(first.stdout)
    assert payload["failed"] == 0
    overview = next(item for item in payload["results"] if item["deriver"] == "overview")
    assert overview["status"] == "done"
    assert overview["cached"] is False
    code = main(["derive", digest, "--all-lazy", "--data", str(data)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    overview = next(item for item in payload["results"] if item["deriver"] == "overview")
    assert overview["cached"] is True


def test_derive_broken_bundle(
    tmp_path: Path, bundle: BundleFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupted stored manifest fails the overview derivation and exits 1."""
    data = tmp_path / "data"
    digest = _ingest_via_main(data, bundle.root, capsys)
    manifest = data / "bundles" / digest / "raw" / "dataset" / "dataset_manifest.json"
    os.chmod(manifest, 0o644)
    manifest.write_bytes(b"{")
    result = _run("derive", digest, "--all-lazy", "--data", str(data))
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["failed"] == 1
    overview = next(item for item in payload["results"] if item["deriver"] == "overview")
    assert overview["status"] == "failed"
    assert overview["reason"] == "error"
    zeros = _run("derive", "0" * 64, "--data", str(tmp_path / "data2"))
    assert zeros.returncode == 1
    assert json.loads(zeros.stdout)["error"]["type"] == "not_found"


def test_serve_help() -> None:
    """``serve --help`` documents the single-worker constraint and has no --workers option."""
    result = _run("serve", "--help")
    assert result.returncode == 0
    assert "one worker" in result.stdout
    assert "--workers" not in result.stdout


def test_serve_health(tmp_path: Path) -> None:
    """The serve command starts a healthy HTTP backend on the requested port."""
    port = _free_port()
    data = tmp_path / "data"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "plateau_rt.viewer",
            "serve",
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
        deadline = time.monotonic() + 30
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
        assert response.json()["status"] == "ok"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_derive_in_process_lazy_combinations(
    tmp_path: Path, bundle: BundleFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--all-lazy`` runs every enum combination of a lazy deriver."""
    data = tmp_path / "data"
    digest = _ingest_via_main(data, bundle.root, capsys)
    with registered(EnumDeriver()):
        code = main(["derive", digest, "--all-lazy", "--data", str(data)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["failed"] == 0
    modes = {
        item["params"].get("mode"): item["status"]
        for item in payload["results"]
        if item["deriver"] == "tenum"
    }
    assert modes == {"a": "done", "b": "done"}


def test_derive_in_process_failure_and_skip(
    tmp_path: Path, bundle: BundleFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing lazy deriver exits 1; a range-parameter deriver is skipped."""
    data = tmp_path / "data"
    digest = _ingest_via_main(data, bundle.root, capsys)
    with registered(Boom()):
        code = main(["derive", digest, "--all-lazy", "--data", str(data)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    boom = next(item for item in payload["results"] if item["deriver"] == "tboom")
    assert boom["status"] == "failed"
    assert boom["reason"] == "error"

    other = tmp_path / "data2"
    digest2 = _ingest_via_main(other, bundle.root, capsys)
    with registered(SlowDeriver()):
        code = main(["derive", digest2, "--all-lazy", "--data", str(other)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    skipped = next(item for item in payload["results"] if item["deriver"] == "tslow")
    assert skipped["status"] == "skipped"


def test_derive_without_all_lazy(
    tmp_path: Path, bundle: BundleFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--all-lazy`` the lazy derivers are not enumerated."""
    data = tmp_path / "data"
    digest = _ingest_via_main(data, bundle.root, capsys)
    with registered(EnumDeriver()):
        code = main(["derive", digest, "--data", str(data)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert all(item["deriver"] != "tenum" for item in payload["results"])
    assert any(item["deriver"] == "overview" for item in payload["results"])
