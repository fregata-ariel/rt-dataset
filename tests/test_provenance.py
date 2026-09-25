"""Tests for build/dataset provenance payloads and git discovery."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy
import pytest

from plateau_rt.application.provenance import (
    collect_provenance,
    git_head_commit,
    package_version,
    sionna_rt_commit,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def test_collect_provenance_fixed_inputs(tmp_path: Path) -> None:
    """Fixed argv/now give the exact payload shape and values."""
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=9)))
    payload = collect_provenance(
        argv=["prog", "rf-camera-multiview", "a b"],
        now=now,
        repo_root=tmp_path,
        environ={},
    )
    assert payload["generated_at_utc"] == "2026-01-01T18:04:05Z"
    assert payload["argv"] == ["prog", "rf-camera-multiview", "a b"]
    assert payload["command"] == "prog rf-camera-multiview 'a b'"
    assert payload["python_version"] == platform.python_version()
    assert payload["packages"]["numpy"] == numpy.__version__
    assert list(payload.keys()) == [
        "generated_at_utc",
        "python_version",
        "packages",
        "sionna_rt_commit",
        "sionna_rt_commit_source",
        "plateau_rt_commit",
        "command",
        "argv",
    ]
    assert json.dumps(payload)
    assert payload["sionna_rt_commit"] is None
    assert payload["plateau_rt_commit"] is None


def test_collect_provenance_rejects_naive_now(tmp_path: Path) -> None:
    """A naive datetime raises ValueError; unknown distributions give None."""
    with pytest.raises(ValueError):
        collect_provenance(now=datetime(2026, 1, 2, 3, 4, 5), repo_root=tmp_path)
    assert package_version("definitely-not-a-real-dist-xyz") is None


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_git_head_commit_detached(tmp_path: Path) -> None:
    """A detached HEAD file resolves directly."""
    _write(tmp_path / ".git" / "HEAD", SHA_A + "\n")
    assert git_head_commit(tmp_path) == SHA_A


def test_git_head_commit_loose_ref(tmp_path: Path) -> None:
    """A symref HEAD resolves through the loose ref file."""
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(tmp_path / ".git" / "refs" / "heads" / "main", SHA_A + "\n")
    assert git_head_commit(tmp_path) == SHA_A


def test_git_head_commit_packed_refs(tmp_path: Path) -> None:
    """A symref HEAD resolves through packed-refs, skipping comments."""
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(
        tmp_path / ".git" / "packed-refs",
        f"# pack-refs with: peeled fully-sorted \n{SHA_B} refs/heads/main\n^{SHA_A}\n",
    )
    assert git_head_commit(tmp_path) == SHA_B


def test_git_head_commit_gitfile(tmp_path: Path) -> None:
    """A .git file with a relative gitdir resolves the detached HEAD."""
    gitdir = tmp_path / "gitdirs" / "sub"
    _write(gitdir / "HEAD", SHA_A + "\n")
    _write(tmp_path / "checkout" / ".git", "gitdir: ../gitdirs/sub\n")
    assert git_head_commit(tmp_path / "checkout") == SHA_A


def test_git_head_commit_worktree_commondir(tmp_path: Path) -> None:
    """A worktree gitdir resolves a ref stored under its commondir."""
    gitdir = tmp_path / "worktree" / ".git"
    common = tmp_path / "common"
    _write(gitdir / "HEAD", "ref: refs/heads/feature\n")
    _write(gitdir / "commondir", "../../common\n")
    _write(common / "refs" / "heads" / "feature", SHA_B + "\n")
    assert git_head_commit(tmp_path / "worktree") == SHA_B


def test_git_head_commit_missing_and_dangling(tmp_path: Path) -> None:
    """No .git gives None; a ref that resolves nowhere gives None."""
    assert git_head_commit(tmp_path / "absent") is None
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/ghost\n")
    assert git_head_commit(tmp_path) is None


def test_sionna_rt_commit_env_wins(tmp_path: Path) -> None:
    """The env var wins over any checkout and is lowercased."""
    submodule = tmp_path / "third_party" / "sionna-rt"
    _write(submodule / ".git" / "HEAD", SHA_A + "\n")
    commit, source = sionna_rt_commit(
        repo_root=tmp_path, environ={"PLATEAU_RT_SIONNA_COMMIT": "ABCDEF1"}
    )
    assert (commit, source) == ("abcdef1", "env:PLATEAU_RT_SIONNA_COMMIT")
    with pytest.raises(ValueError):
        sionna_rt_commit(repo_root=tmp_path, environ={"PLATEAU_RT_SIONNA_COMMIT": "not-a-sha"})


def test_sionna_rt_commit_submodule_and_empty(tmp_path: Path) -> None:
    """A fake submodule checkout reports its HEAD; nothing gives (None, None)."""
    submodule = tmp_path / "third_party" / "sionna-rt"
    _write(submodule / ".git" / "HEAD", SHA_A + "\n")
    assert sionna_rt_commit(repo_root=tmp_path, environ={}) == (SHA_A, "submodule_checkout")
    assert sionna_rt_commit(repo_root=tmp_path / "empty", environ={}) == (None, None)


def test_collect_provenance_reports_fake_repo_commits(tmp_path: Path) -> None:
    """Both commit fields come from the fake repo checkout."""
    submodule = tmp_path / "third_party" / "sionna-rt"
    _write(submodule / ".git" / "HEAD", SHA_A + "\n")
    _write(tmp_path / ".git" / "HEAD", SHA_B + "\n")
    payload = collect_provenance(repo_root=tmp_path, environ={})
    assert payload["sionna_rt_commit"] == SHA_A
    assert payload["sionna_rt_commit_source"] == "submodule_checkout"
    assert payload["plateau_rt_commit"] == SHA_B
