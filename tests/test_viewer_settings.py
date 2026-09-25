"""Tests for viewer settings loading and validation."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from plateau_rt.viewer.extract import ExtractLimits
from plateau_rt.viewer.settings import (
    DEFAULT_ALLOWED_HOSTS,
    DEFAULT_DATA_DIR,
    DEFAULT_DERIVE_MEM_BYTES,
    DEFAULT_DERIVE_TIMEOUT_S,
    DEFAULT_MAX_ARRAY_BYTES,
    DEFAULT_MAX_CONCURRENT_DERIVES,
    DEFAULT_MAX_EXTRACTED_BYTES,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_UPLOAD_BYTES,
    GiB,
    MiB,
    SettingsError,
    ViewerSettings,
)


def test_defaults() -> None:
    settings = ViewerSettings.from_env({})
    assert settings.max_upload_bytes == DEFAULT_MAX_UPLOAD_BYTES == 4 * GiB
    assert settings.max_extracted_bytes == DEFAULT_MAX_EXTRACTED_BYTES == 16 * GiB
    assert settings.max_files == DEFAULT_MAX_FILES == 100_000
    assert settings.max_array_bytes == DEFAULT_MAX_ARRAY_BYTES == 1 * GiB
    assert settings.derive_timeout_s == DEFAULT_DERIVE_TIMEOUT_S == 120.0
    assert settings.derive_mem_bytes == DEFAULT_DERIVE_MEM_BYTES == 4 * GiB
    assert settings.max_concurrent_derives == DEFAULT_MAX_CONCURRENT_DERIVES == 2
    assert settings.data_dir.is_absolute()
    assert settings.data_dir.name == DEFAULT_DATA_DIR
    assert settings.import_roots == ()
    assert settings.allowed_hosts == DEFAULT_ALLOWED_HOSTS == ("127.0.0.1", "localhost")
    assert settings.allowed_origins == ()
    assert settings.read_only is False
    assert settings.extract_limits == ExtractLimits(100_000, 16 * GiB)


def test_parsing_valid_values(tmp_path: Path) -> None:
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    data_dir = tmp_path / "data"
    environ = {
        "VIEWER_DATA": str(data_dir),
        "VIEWER_MAX_UPLOAD_BYTES": "2 GiB",
        "VIEWER_MAX_EXTRACTED_BYTES": "512MiB",
        "VIEWER_MAX_FILES": "1024",
        "VIEWER_MAX_ARRAY_BYTES": "1024",
        "VIEWER_IMPORT_ROOTS": os.pathsep.join([str(root_a), str(root_b)]),
        "VIEWER_DERIVE_TIMEOUT_S": "30.5",
        "VIEWER_DERIVE_MEM_BYTES": "3",
        "VIEWER_MAX_CONCURRENT_DERIVES": "3",
        "VIEWER_ALLOWED_HOSTS": " a.example , b ",
        "VIEWER_READ_ONLY": "TRUE",
    }
    settings = ViewerSettings.from_env(environ)
    assert settings.data_dir == data_dir.resolve()
    assert settings.max_upload_bytes == 2 * GiB
    assert settings.max_extracted_bytes == 512 * MiB
    assert settings.max_files == 1024
    assert settings.max_array_bytes == 1024
    assert settings.import_roots == (root_a.resolve(), root_b.resolve())
    assert settings.derive_timeout_s == 30.5
    assert settings.derive_mem_bytes == 3
    assert settings.max_concurrent_derives == 3
    assert settings.allowed_hosts == ("a.example", "b")
    assert settings.read_only is True


def test_parsing_allowed_origins() -> None:
    """VIEWER_ALLOWED_ORIGINS is split, stripped, lower-cased and loses a trailing slash."""
    settings = ViewerSettings.from_env(
        {"VIEWER_ALLOWED_ORIGINS": " https://Viewer.Example.org/ , http://127.0.0.1:8765 "}
    )
    assert settings.allowed_origins == (
        "https://viewer.example.org",
        "http://127.0.0.1:8765",
    )


@pytest.mark.parametrize("value", ["a,,b", "ftp://x", "https://x/path", "x.example", "https://u@x"])
def test_invalid_allowed_origins(value: str) -> None:
    """Malformed origin entries are rejected at startup."""
    with pytest.raises(SettingsError) as info:
        ViewerSettings.from_env({"VIEWER_ALLOWED_ORIGINS": value})
    assert "VIEWER_ALLOWED_ORIGINS" in str(info.value)


def test_direct_allowed_origins_validates() -> None:
    """Direct construction validates allowed_origins entries too."""
    with pytest.raises(SettingsError):
        ViewerSettings(allowed_origins=("HTTPS://X",))
    with pytest.raises(SettingsError):
        ViewerSettings(allowed_origins=("https://x/path",))
    assert ViewerSettings(allowed_origins=("https://x",)).allowed_origins == ("https://x",)


def test_blank_values_use_defaults() -> None:
    settings = ViewerSettings.from_env(
        {
            "VIEWER_DATA": "   ",
            "VIEWER_MAX_FILES": "  ",
            "VIEWER_ALLOWED_HOSTS": "",
            "VIEWER_ALLOWED_ORIGINS": "   ",
            "VIEWER_READ_ONLY": "\t",
        }
    )
    assert settings.data_dir.name == DEFAULT_DATA_DIR
    assert settings.max_files == DEFAULT_MAX_FILES
    assert settings.allowed_hosts == DEFAULT_ALLOWED_HOSTS
    assert settings.allowed_origins == ()
    assert settings.read_only is False


INVALID_SCALARS = [
    ("VIEWER_MAX_UPLOAD_BYTES", "-1"),
    ("VIEWER_MAX_UPLOAD_BYTES", "0"),
    ("VIEWER_MAX_UPLOAD_BYTES", "1.5GiB"),
    ("VIEWER_MAX_UPLOAD_BYTES", "1GB"),
    ("VIEWER_MAX_EXTRACTED_BYTES", "abc"),
    ("VIEWER_MAX_FILES", "-5"),
    ("VIEWER_MAX_FILES", "0"),
    ("VIEWER_MAX_FILES", "1e3"),
    ("VIEWER_MAX_ARRAY_BYTES", "0"),
    ("VIEWER_DERIVE_TIMEOUT_S", "0"),
    ("VIEWER_DERIVE_TIMEOUT_S", "-1"),
    ("VIEWER_DERIVE_TIMEOUT_S", "nan"),
    ("VIEWER_DERIVE_TIMEOUT_S", "inf"),
    ("VIEWER_DERIVE_MEM_BYTES", "-1"),
    ("VIEWER_MAX_CONCURRENT_DERIVES", "0"),
    ("VIEWER_ALLOWED_HOSTS", ","),
    ("VIEWER_ALLOWED_HOSTS", "a,,b"),
    ("VIEWER_READ_ONLY", "maybe"),
]


@pytest.mark.parametrize(("key", "value"), INVALID_SCALARS)
def test_invalid_scalar_values(key: str, value: str) -> None:
    with pytest.raises(SettingsError) as info:
        ViewerSettings.from_env({key: value})
    assert key in str(info.value)


def test_invalid_import_roots(tmp_path: Path) -> None:
    existing_file = tmp_path / "afile"
    existing_file.write_bytes(b"x")
    for value in (str(tmp_path / "missing"), str(existing_file), "relative/dir"):
        with pytest.raises(SettingsError) as info:
            ViewerSettings.from_env({"VIEWER_IMPORT_ROOTS": value})
        assert "VIEWER_IMPORT_ROOTS" in str(info.value)


def test_invalid_data_dir_is_a_file(tmp_path: Path) -> None:
    existing_file = tmp_path / "afile"
    existing_file.write_bytes(b"x")
    with pytest.raises(SettingsError) as info:
        ViewerSettings.from_env({"VIEWER_DATA": str(existing_file)})
    assert "VIEWER_DATA" in str(info.value)


def test_direct_construction_validates() -> None:
    with pytest.raises(SettingsError):
        ViewerSettings(max_files=0)
    with pytest.raises(SettingsError):
        ViewerSettings(read_only="yes")  # type: ignore[arg-type]
    assert issubclass(SettingsError, ValueError)


def test_settings_are_frozen() -> None:
    settings = ViewerSettings()
    with pytest.raises(FrozenInstanceError):
        settings.max_files = 1  # type: ignore[misc]
