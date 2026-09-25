"""HTTP-free bundle ingest: validate a staged root or archive and commit it (Sionna-free)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plateau_rt.viewer import VIEWER_VERSION
from plateau_rt.viewer.extract import safe_extract
from plateau_rt.viewer.kinds import BundleValidationError, MemberInfo, validate_bundle
from plateau_rt.viewer.store import Store


@dataclass(frozen=True)
class IngestResult:
    """Outcome of committing one bundle."""

    digest: str
    created: bool
    members: tuple[dict[str, Any], ...]


def member_records(infos: Sequence[MemberInfo]) -> list[dict[str, Any]]:
    """Return the store member records (id, kind, path, links, validation) for ``infos``."""
    return [
        {
            **info.member.to_dict(),
            "validation": {
                "schema_version": info.schema_version,
                "mode": info.mode,
                "summary": dict(info.summary),
            },
        }
        for info in infos
    ]


def ingest_staged(store: Store, root: Path, *, name: str) -> IngestResult:
    """Validate the staged bundle root ``root`` and commit it, discarding staging on failure."""
    try:
        infos = validate_bundle(root, max_array_bytes=store.settings.max_array_bytes)
        members = member_records(infos)
        try:
            digest, created = store.commit(
                root, name=name, members=members, validated_with=VIEWER_VERSION
            )
        except BundleValidationError:
            raise
        except ValueError as exc:
            raise BundleValidationError(None, str(exc)) from exc
    except BaseException:
        try:
            store.discard_staging(root)
        except Exception:
            pass
        raise
    if not created:
        record = store.get(digest)
        if record is not None:
            members = list(record.members)
    return IngestResult(digest=digest, created=created, members=tuple(members))


def ingest_archive(store: Store, archive_path: Path, *, name: str) -> IngestResult:
    """Extract ``archive_path`` into staging, then validate and commit it."""
    dest = store.new_staging()
    try:
        report = safe_extract(archive_path, dest, store.settings.extract_limits)
        return ingest_staged(store, report.root, name=name)
    except BaseException:
        if os.path.lexists(dest):
            try:
                store.discard_staging(dest)
            except Exception:
                pass
        raise
