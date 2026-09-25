"""Headless command line interface for the plateau_rt viewer (``python -m plateau_rt.viewer``)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plateau_rt.viewer.api.app import create_app
from plateau_rt.viewer.extract import UnsafeArchiveError
from plateau_rt.viewer.ingest import ingest_archive, ingest_staged
from plateau_rt.viewer.jobs import JobManager, eager_items, lazy_items, lookup_cached
from plateau_rt.viewer.kinds import BundleValidationError, UnknownKindError
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store

SERVE_DESCRIPTION = (
    "Run the viewer backend. Job state and request deduplication live inside the server "
    "process, so the server runs exactly one worker (multiple workers would run duplicate "
    "jobs and lose job status); there is deliberately no multi-worker option."
)


def _build_settings(data: str | None) -> ViewerSettings:
    """Return the viewer settings, overriding the data directory when ``data`` is given."""
    settings = ViewerSettings.from_env()
    if data is not None:
        settings = dataclasses.replace(settings, data_dir=Path(data).expanduser().resolve())
    return settings


def _print_json(payload: Any) -> None:
    """Print one JSON object to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _error(kind: str, member: str | None, message: str) -> dict[str, Any]:
    """Return the documented error envelope."""
    return {"error": {"type": kind, "member": member, "message": message}}


def _serve(args: argparse.Namespace) -> int:
    """Run the uvicorn server in the current process."""
    import uvicorn

    settings = _build_settings(args.data)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    """Ingest a directory or archive and print the result as JSON."""
    settings = _build_settings(args.data)
    store = Store(settings)
    path = Path(args.path)
    name = args.name if args.name is not None else path.name
    try:
        if path.is_dir():
            root = store.stage_from_directory(path, settings.extract_limits)
            result = ingest_staged(store, root, name=name)
        elif path.is_file():
            result = ingest_archive(store, path, name=name)
        else:
            _print_json(_error("not_found", None, f"no such path: {args.path}"))
            return 1
    except UnsafeArchiveError as exc:
        _print_json(_error("unsafe_archive", None, str(exc)))
        return 1
    except UnknownKindError as exc:
        _print_json(_error("unknown_kind", exc.member_id, exc.message))
        return 1
    except BundleValidationError as exc:
        _print_json(_error("validation", exc.member_id, exc.message))
        return 1
    except (ValueError, OSError) as exc:
        _print_json(_error("error", None, str(exc)))
        return 1
    _print_json(
        {"digest": result.digest, "created": result.created, "members": list(result.members)}
    )
    return 0


def _result_item(
    item: Any,
    status: str,
    cached: bool,
    job_id: str | None,
    reason: str | None,
    error: str | None,
) -> dict[str, Any]:
    """Build one JSON result row of the ``derive`` command."""
    if item.prepared is not None:
        params = dict(item.prepared.params)
        params_key = item.prepared.params_key
    else:
        params = dict(item.params)
        params_key = ""
    return {
        "member": item.member,
        "deriver": item.deriver,
        "params": params,
        "params_key": params_key,
        "status": status,
        "cached": cached,
        "job_id": job_id,
        "reason": reason,
        "error": error,
    }


def _derive(args: argparse.Namespace) -> int:
    """Derive the eager (and optionally lazy) combinations of a bundle."""
    settings = _build_settings(args.data)
    store = Store(settings)
    try:
        record = store.get(args.digest)
    except ValueError as exc:
        _print_json(_error("not_found", None, str(exc)))
        return 1
    if record is None:
        _print_json(_error("not_found", None, f"unknown bundle {args.digest!r}"))
        return 1
    digest = record.digest
    items = eager_items(store, digest)
    if args.all_lazy:
        items.extend(lazy_items(store, digest))
    manager = JobManager(store)
    results: list[dict[str, Any]] = []
    submitted: list[tuple[int, str]] = []
    try:
        for item in items:
            if item.prepared is None:
                if (item.error or "").startswith("skipped:"):
                    results.append(_result_item(item, "skipped", False, None, None, item.error))
                else:
                    results.append(_result_item(item, "failed", False, None, "error", item.error))
                continue
            if lookup_cached(store, item.prepared) is not None:
                results.append(_result_item(item, "done", True, None, None, None))
                continue
            kind = "eager" if item.prepared.deriver.spec.eager else "lazy"
            job = manager.submit(item.prepared, kind=kind, force=True)
            submitted.append((len(results), job.job_id))
            results.append(_result_item(item, job.status, False, job.job_id, None, None))
        infos = manager.wait([job_id for _, job_id in submitted])
        by_id = {info.job_id: info for info in infos}
        for index, job_id in submitted:
            info = by_id.get(job_id)
            if info is None:
                continue
            results[index]["status"] = info.status
            results[index]["reason"] = info.reason
            results[index]["error"] = info.error
    finally:
        manager.shutdown()
    failed = sum(1 for result in results if result["status"] == "failed")
    _print_json({"digest": digest, "results": results, "failed": failed})
    return 1 if failed else 0


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser with its sub-commands."""
    parser = argparse.ArgumentParser(prog="python -m plateau_rt.viewer")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", description=SERVE_DESCRIPTION, help="run the HTTP backend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--data", default=None)
    serve.set_defaults(handler=_serve)

    ingest = sub.add_parser("ingest", help="ingest a bundle directory or archive")
    ingest.add_argument("path")
    ingest.add_argument("--name", default=None)
    ingest.add_argument("--data", default=None)
    ingest.set_defaults(handler=_ingest)

    derive = sub.add_parser("derive", help="run the derivations of one bundle")
    derive.add_argument("digest")
    derive.add_argument("--all-lazy", action="store_true")
    derive.add_argument("--data", default=None)
    derive.set_defaults(handler=_derive)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the viewer CLI and return its exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is None:
        parser.print_help(sys.stderr)
        return 2
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
