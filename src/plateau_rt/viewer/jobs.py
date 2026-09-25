"""In-process job manager for viewer derivations.

Each derivation runs in a ``spawn`` child process with time and memory limits; see
``docs/viewer.md`` ("Jobs").
"""

from __future__ import annotations

import datetime
import heapq
import logging
import multiprocessing
import re
import resource
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

from plateau_rt.viewer.derive import (
    RANGE_KINDS,
    DerivedResult,
    DeriveNotFound,
    Deriver,
    _find_cached,
    get_deriver,
    get_or_derive,
    open_context,
    register,
    registered_derivers,
    validate_params,
)
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store

JOB_STATUSES = ("queued", "running", "done", "failed")
TERMINAL_STATUSES = ("done", "failed")
JOB_KINDS = ("eager", "lazy")
FAIL_REASONS = ("timeout", "memory_limit", "restart", "error", "crashed")
STAGES = ("queued", "starting", "deriving", "done")
LAZY_PRIORITY, EAGER_PRIORITY = 0, 1
KILL_JOIN_TIMEOUT_S = 5.0
MAX_ERROR_CHARS = 2000

_JOB_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prepared:
    """A validated derivation request: the deriver object, canonical params and cache key."""

    deriver: Deriver
    digest: str
    member: str
    params: dict[str, str]
    params_key: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Return the dedup key (digest, member, deriver name, params_key)."""
        return (self.digest, self.member, self.deriver.spec.name, self.params_key)


@dataclass(frozen=True)
class JobInfo:
    """One row of the jobs table."""

    job_id: str
    kind: str
    digest: str
    member: str
    deriver: str
    params_key: str
    status: str
    stage: str | None
    reason: str | None
    error: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON fields (all of the above)."""
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "digest": self.digest,
            "member": self.member,
            "deriver": self.deriver,
            "params_key": self.params_key,
            "status": self.status,
            "stage": self.stage,
            "reason": self.reason,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class EagerItem:
    """One eager (deriver, member, params) combination of a bundle."""

    member: str
    deriver: str
    params: dict[str, str]
    prepared: Prepared | None
    error: str | None


@dataclass
class _ActiveJob:
    """In-memory state of a queued or running job."""

    prepared: Prepared
    kind: str
    status: str
    process: Any = None
    cancelled: bool = False


def _now() -> str:
    """Return the store timestamp for the current UTC time."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_error(status: str, text: str | None) -> tuple[str | None, str | None]:
    """Return the parsed ``(reason, error)`` of a failed row."""
    if status != "failed" or text is None:
        return None, None
    reason, separator, message = text.partition(": ")
    if separator and reason in FAIL_REASONS:
        return reason, message
    return "error", text


def _job_info(row: Any) -> JobInfo:
    """Build a :class:`JobInfo` from one SQLite row."""
    reason, error = _parse_error(str(row["status"]), row["error"])
    return JobInfo(
        job_id=str(row["job_id"]),
        kind=str(row["kind"]),
        digest=str(row["digest"] or ""),
        member=str(row["member"] or ""),
        deriver=str(row["deriver"] or ""),
        params_key=str(row["params_key"] or ""),
        status=str(row["status"]),
        stage=None if row["stage"] is None else str(row["stage"]),
        reason=reason,
        error=error,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _job_columns() -> str:
    """Return the SELECT list for a jobs row."""
    return (
        "job_id, kind, digest, member, deriver, params_key, status, stage, "
        "error, created_at, updated_at"
    )


def prepare(
    store: Store, digest: str, member: str, deriver: str, params: Mapping[str, str]
) -> Prepared:
    """Validate a derivation request and return the prepared form."""
    deriver_obj = get_deriver(deriver)
    ctx = open_context(store, digest, member)
    kind = ctx.member_info.get("kind")
    if kind not in deriver_obj.spec.kinds:
        raise DeriveNotFound(f"deriver {deriver!r} does not support member kind {kind!r}")
    _typed, canonical, params_key_ = validate_params(deriver_obj, ctx, params)
    return Prepared(deriver_obj, ctx.digest, member, canonical, params_key_)


def lookup_cached(store: Store, prepared: Prepared) -> DerivedResult | None:
    """Return a valid cached result for ``prepared``, or None."""
    return _find_cached(
        store, prepared.digest, prepared.member, prepared.deriver, prepared.params_key
    )


def _resolve_record(store: Store, digest: str) -> Any:
    """Return the bundle record for ``digest`` or raise :class:`DeriveNotFound`."""
    try:
        record = store.get(digest)
    except ValueError as exc:
        raise DeriveNotFound(f"unknown bundle {digest!r}") from exc
    if record is None:
        raise DeriveNotFound(f"unknown bundle {digest!r}")
    return record


def _items(store: Store, digest: str, *, eager: bool) -> list[EagerItem]:
    """Enumerate the combinations of the eager or non-eager derivers of a bundle."""
    record = _resolve_record(store, digest)
    digest = record.digest
    items: list[EagerItem] = []
    for deriver in registered_derivers():
        if deriver.spec.eager is not eager:
            continue
        for info in record.members:
            kind = info.get("kind")
            if kind not in deriver.spec.kinds:
                continue
            member = str(info["id"])
            if not eager and any(param.kind in RANGE_KINDS for param in deriver.spec.params):
                items.append(
                    EagerItem(
                        member=member,
                        deriver=deriver.spec.name,
                        params={},
                        prepared=None,
                        error="skipped: range parameters cannot be enumerated",
                    )
                )
                continue
            try:
                ctx = open_context(store, digest, member)
                space = deriver.param_space(ctx)
            except Exception as exc:
                items.append(
                    EagerItem(
                        member=member,
                        deriver=deriver.spec.name,
                        params={},
                        prepared=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            for params in space:
                try:
                    prepared = prepare(store, digest, member, deriver.spec.name, params)
                except Exception as exc:
                    items.append(
                        EagerItem(
                            member=member,
                            deriver=deriver.spec.name,
                            params=dict(params),
                            prepared=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                else:
                    items.append(
                        EagerItem(
                            member=member,
                            deriver=deriver.spec.name,
                            params=dict(params),
                            prepared=prepared,
                            error=None,
                        )
                    )
    return items


def eager_items(store: Store, digest: str) -> list[EagerItem]:
    """Return every eager (deriver, member, params) combination of a bundle."""
    return _items(store, digest, eager=True)


def lazy_items(store: Store, digest: str) -> list[EagerItem]:
    """Return every non-eager (deriver, member, params) combination of a bundle."""
    return _items(store, digest, eager=False)


def _has_memory_error(exc: BaseException) -> bool:
    """Return True when a :class:`MemoryError` appears in the exception chain."""
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, MemoryError):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _child_main(
    conn: Connection,
    settings: ViewerSettings,
    digest: str,
    member: str,
    deriver: Deriver,
    params: dict[str, str],
    mem_bytes: int,
) -> None:
    """Child entry point: apply the memory limit, derive and report back over ``conn``."""
    try:
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        new_soft = mem_bytes if hard == resource.RLIM_INFINITY else min(mem_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        register(deriver, replace=True)
        store = Store(settings)
        conn.send({"type": "stage", "stage": "deriving"})
        get_or_derive(store, digest, member, deriver.spec.name, params)
        conn.send({"type": "done"})
    except BaseException as exc:
        reason = "memory_limit" if _has_memory_error(exc) else "error"
        message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        try:
            conn.send({"type": "failed", "reason": reason, "message": message})
        except Exception:
            pass
    finally:
        conn.close()


def _kill_process(process: Any) -> None:
    """Kill ``process`` if alive and join it within the kill timeout."""
    if process.is_alive():
        process.kill()
    process.join(KILL_JOIN_TIMEOUT_S)
    if process.is_alive():
        process.kill()
        process.join(KILL_JOIN_TIMEOUT_S)


def _reap_process(process: Any) -> None:
    """Join ``process`` within the kill timeout, killing it when it overruns."""
    process.join(KILL_JOIN_TIMEOUT_S)
    if process.is_alive():
        process.kill()
        process.join(KILL_JOIN_TIMEOUT_S)


class JobManager:
    """Queue, deduplicate and run derivations in spawn children with limits."""

    def __init__(self, store: Store) -> None:
        """Create a manager for ``store`` with lazily started workers."""
        self.store = store
        self.settings = store.settings
        self._cond = threading.Condition()
        self._heap: list[tuple[int, int, str]] = []
        self._active: dict[str, _ActiveJob] = {}
        self._by_key: dict[tuple[str, str, str, str], str] = {}
        self._workers: list[threading.Thread] = []
        self._seq = 0
        self._shutdown = False

    def submit(self, prepared: Prepared, *, kind: str = "lazy", force: bool = False) -> JobInfo:
        """Queue a derivation, returning an existing or sticky-failed job when applicable."""
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        key = prepared.key
        with self._cond:
            if self._shutdown:
                raise RuntimeError("job manager is shut down")
            active_id = self._by_key.get(key)
            if active_id is not None:
                active = self._active.get(active_id)
                if active is not None:
                    if kind == "lazy" and active.status == "queued":
                        self._push(active_id, LAZY_PRIORITY)
                        self._cond.notify_all()
                    info = self.get(active_id)
                    if info is not None:
                        return info
            if not force:
                latest = self.latest(prepared.digest, prepared.member, key[2], prepared.params_key)
                if latest is not None and latest.status == "failed" and latest.reason != "restart":
                    return latest
            job_id = uuid.uuid4().hex
            now = _now()
            self._insert(job_id, kind, prepared, now)
            self._active[job_id] = _ActiveJob(prepared=prepared, kind=kind, status="queued")
            self._by_key[key] = job_id
            self._ensure_workers()
            self._push(job_id, EAGER_PRIORITY if kind == "eager" else LAZY_PRIORITY)
            self._cond.notify_all()
            info = self.get(job_id)
            if info is None:  # pragma: no cover - the row was just inserted
                raise RuntimeError(f"job {job_id} disappeared")
            return info

    def submit_eager(self, digest: str, *, force: bool = False) -> list[JobInfo]:
        """Queue every missing eager derivation of a bundle."""
        jobs: list[JobInfo] = []
        for item in eager_items(self.store, digest):
            if item.prepared is None:
                continue
            if lookup_cached(self.store, item.prepared) is not None:
                continue
            jobs.append(self.submit(item.prepared, kind="eager", force=force))
        return jobs

    def recover(self) -> list[JobInfo]:
        """Mark stale jobs failed and re-queue the missing eager derivations."""
        now = _now()
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', "
                "error = 'restart: the server restarted before the job finished', "
                "updated_at = ? WHERE status IN ('queued', 'running')",
                (now,),
            )
        submitted: list[JobInfo] = []
        for record in self.store.list():
            try:
                submitted.extend(self.submit_eager(record.digest, force=True))
            except Exception:
                _LOGGER.exception("eager recovery failed for bundle %s", record.digest)
        return submitted

    def get(self, job_id: str) -> JobInfo | None:
        """Return the job row for ``job_id``, or None when unknown or malformed."""
        if not isinstance(job_id, str) or _JOB_ID_RE.fullmatch(job_id) is None:
            return None
        with self.store.connect() as conn:
            row = conn.execute(
                f"SELECT {_job_columns()} FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return None if row is None else _job_info(row)

    def latest(self, digest: str, member: str, deriver: str, params_key: str) -> JobInfo | None:
        """Return the most recent job row for one dedup key, or None."""
        with self.store.connect() as conn:
            row = conn.execute(
                f"SELECT {_job_columns()} FROM jobs WHERE digest = ? AND member = ? "
                "AND deriver = ? AND params_key = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (digest, member, deriver, params_key),
            ).fetchone()
        return None if row is None else _job_info(row)

    def latest_by_key(self, digest: str) -> dict[tuple[str, str, str], JobInfo]:
        """Return the most recent job per ``(member, deriver, params_key)`` of a bundle."""
        with self.store.connect() as conn:
            rows = conn.execute(
                f"SELECT {_job_columns()} FROM jobs WHERE digest = ? ORDER BY created_at, rowid",
                (digest,),
            ).fetchall()
        latest: dict[tuple[str, str, str], JobInfo] = {}
        for row in rows:
            info = _job_info(row)
            latest[(info.member, info.deriver, info.params_key)] = info
        return latest

    def wait(self, job_ids: Sequence[str], timeout: float | None = None) -> list[JobInfo]:
        """Block until every job is terminal, returning its row (missing jobs are skipped)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            infos = [self.get(job_id) for job_id in job_ids]
            if all(info is None or info.status in TERMINAL_STATUSES for info in infos):
                return [info for info in infos if info is not None]
            if deadline is None:
                wait_for = 0.2
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"jobs did not finish within {timeout} s")
                wait_for = min(remaining, 0.2)
            with self._cond:
                self._cond.wait(wait_for)

    def cancel_bundle(self, digest: str) -> None:
        """Cancel every active job of ``digest`` and wait for the running ones to stop."""
        with self._cond:
            for job_id, active in list(self._active.items()):
                if active.prepared.digest != digest or active.status not in ("queued", "running"):
                    continue
                active.cancelled = True
                if active.status == "queued":
                    self._drop(job_id)
                elif active.process is not None and active.process.is_alive():
                    active.process.kill()
            self._cond.notify_all()
            self._wait_bundle_cancelled(digest)

    def _wait_bundle_cancelled(self, digest: str) -> None:
        """Wait (bounded by the kill timeout) until no job of ``digest`` is running."""
        deadline = time.monotonic() + KILL_JOIN_TIMEOUT_S
        while time.monotonic() < deadline:
            running = any(
                active.prepared.digest == digest and active.status == "running"
                for active in self._active.values()
            )
            if not running:
                return
            remaining = deadline - time.monotonic()
            self._cond.wait(min(remaining, 0.05))

    def shutdown(self) -> None:
        """Stop accepting jobs, fail active ones as restarted and join the workers."""
        with self._cond:
            if self._shutdown:
                return
            self._shutdown = True
            for job_id, active in list(self._active.items()):
                if active.status in ("queued", "running"):
                    active.cancelled = True
                    active.status = "failed"
                    self._fail(job_id, "restart", "the server stopped before the job finished")
                if active.process is not None and active.process.is_alive():
                    active.process.kill()
            self._cond.notify_all()
        for worker in self._workers:
            worker.join(KILL_JOIN_TIMEOUT_S)

    def _insert(self, job_id: str, kind: str, prepared: Prepared, now: str) -> None:
        """Insert one queued job row."""
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(job_id, kind, digest, member, deriver, params_key, status, "
                "stage, done_bytes, total_bytes, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', NULL, NULL, NULL, ?, ?)",
                (
                    job_id,
                    kind,
                    prepared.digest,
                    prepared.member,
                    prepared.deriver.spec.name,
                    prepared.params_key,
                    now,
                    now,
                ),
            )

    def _push(self, job_id: str, priority: int) -> None:
        """Push one heap entry for ``job_id`` (caller holds the lock)."""
        self._seq += 1
        heapq.heappush(self._heap, (priority, self._seq, job_id))

    def _ensure_workers(self) -> None:
        """Start the worker threads once (caller holds the lock)."""
        if self._workers:
            return
        for index in range(self.settings.max_concurrent_derives):
            worker = threading.Thread(target=self._worker, name=f"viewer-jobs-{index}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def _pop_queued(self) -> str | None:
        """Pop the next queued job id, dropping cancelled or stale entries."""
        while self._heap:
            _priority, _seq, job_id = heapq.heappop(self._heap)
            active = self._active.get(job_id)
            if active is None or active.status != "queued":
                continue
            if active.cancelled:
                self._drop(job_id)
                continue
            return job_id
        return None

    def _drop(self, job_id: str) -> None:
        """Remove a job from the active maps (caller holds the lock)."""
        active = self._active.pop(job_id, None)
        if active is not None:
            self._by_key.pop(active.prepared.key, None)

    def _worker(self) -> None:
        """Run queued jobs one at a time until shutdown."""
        while True:
            with self._cond:
                while True:
                    if self._shutdown:
                        return
                    job_id = self._pop_queued()
                    if job_id is not None:
                        break
                    self._cond.wait(0.2)
                active = self._active.get(job_id)
                if active is None:
                    continue
                active.status = "running"
            try:
                self._execute(job_id)
            except BaseException:
                _LOGGER.exception("job %s runner crashed", job_id)
                try:
                    self._fail(job_id, "error", "job runner crashed")
                except Exception:
                    _LOGGER.exception("could not record job %s failure", job_id)
                with self._cond:
                    self._drop(job_id)
                    self._cond.notify_all()

    def _execute(self, job_id: str) -> None:
        """Start the child for ``job_id`` and follow its messages until it ends."""
        with self._cond:
            active = self._active.get(job_id)
            if active is None:
                return
            prepared = active.prepared
        settings = self.settings
        self._update(job_id, status="running", stage="starting")
        process: Any = None
        recv_conn: Connection | None = None
        send_conn: Connection | None = None
        try:
            ctx = multiprocessing.get_context("spawn")
            recv_conn, send_conn = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_child_main,
                args=(
                    send_conn,
                    settings,
                    prepared.digest,
                    prepared.member,
                    prepared.deriver,
                    prepared.params,
                    settings.derive_mem_bytes,
                ),
                daemon=True,
                name=f"viewer-derive-{job_id[:8]}",
            )
            process.start()
            send_conn.close()
            send_conn = None
            with self._cond:
                current = self._active.get(job_id)
                if current is not None:
                    current.process = process
            deadline = time.monotonic() + settings.derive_timeout_s
            while True:
                if not self._running(job_id):
                    self._kill(process)
                    if self._shutdown:
                        self._fail(job_id, "restart", "the server stopped before the job finished")
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill(process)
                    self._fail(
                        job_id, "timeout", f"derivation exceeded {settings.derive_timeout_s:g} s"
                    )
                    break
                if not recv_conn.poll(min(remaining, 0.2)):
                    continue
                try:
                    message = recv_conn.recv()
                except EOFError:
                    self._reap(process)
                    self._fail(
                        job_id,
                        "crashed",
                        f"derive process exited with code {process.exitcode}",
                    )
                    break
                if not isinstance(message, dict):
                    continue
                mtype = message.get("type")
                if mtype == "stage":
                    self._update(job_id, stage=str(message.get("stage", "deriving")))
                elif mtype == "done":
                    self._reap(process)
                    self._done(job_id)
                    break
                elif mtype == "failed":
                    self._reap(process)
                    self._fail(
                        job_id,
                        str(message.get("reason", "error")),
                        str(message.get("message", "")),
                    )
                    break
        except BaseException:
            _LOGGER.exception("job %s child failed", job_id)
            if process is not None:
                self._kill(process)
            try:
                self._fail(job_id, "error", "job execution failed")
            except Exception:
                _LOGGER.exception("could not record job %s failure", job_id)
        finally:
            if send_conn is not None:
                send_conn.close()
            if recv_conn is not None:
                recv_conn.close()
            with self._cond:
                self._drop(job_id)
                self._cond.notify_all()

    def _running(self, job_id: str) -> bool:
        """Return True when the job is still active and not cancelled."""
        with self._cond:
            active = self._active.get(job_id)
            return active is not None and not active.cancelled

    def _kill(self, process: Any) -> None:
        """Kill and join a child process."""
        _kill_process(process)

    def _reap(self, process: Any) -> None:
        """Join a child process, killing it when it overruns."""
        _reap_process(process)

    def _done(self, job_id: str) -> None:
        """Record a successful job."""
        with self._cond:
            active = self._active.get(job_id)
            if active is not None:
                active.status = "done"
        self._update(job_id, status="done", stage="done")

    def _fail(self, job_id: str, reason: str, message: str) -> None:
        """Record a failed job with ``reason: message`` truncation."""
        with self._cond:
            active = self._active.get(job_id)
            if active is not None:
                active.status = "failed"
        text = f"{reason}: {message[:MAX_ERROR_CHARS]}"
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE job_id = ?",
                (text, _now(), job_id),
            )

    def _update(self, job_id: str, *, status: str | None = None, stage: str | None = None) -> None:
        """Update the status and/or stage of a job row."""
        assignments = ["updated_at = ?"]
        args: list[Any] = [_now()]
        if status is not None:
            assignments.append("status = ?")
            args.append(status)
        if stage is not None:
            assignments.append("stage = ?")
            args.append(stage)
        args.append(job_id)
        with self.store.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ?", args)
