"""N-02 (I-11): the run lock of `medrag-nightly`.

Two nightly runs cannot run at the same time -- the second one triggered is
REJECTED (no wait queue, kept simple -- the task text's explicit preference).
The lock is file-based (single machine, no Redis needed): it carries the
PID + start time. Even if the process holding the lock dies via `kill -9`,
the next call does NOT assume the lock is permanent -- it verifies the PID is
still alive with `psutil.pid_exists()`; if the PID is gone the lock is STALE
and is taken over (NO TTL, the only source is PID liveness -- this was
preferred because it answers "is the owner really dead?" directly instead of
"let it go when the duration expires").

`os.kill(pid, 0)` is NOT used: on Windows CPython's `os.kill()` also maps
signal 0 to `TerminateProcess` (see CPython `nt_kill`) -- so asking "is it
alive" could accidentally kill the process ITSELF. Instead `psutil.pid_exists()`
is used: it sends no signal, only reads the PID table, and is safe on every
platform.

2026-08-27 fix (`_is_stale`): when the lock file moved to a persistent path
(`NIGHTLY_LOCK_PATH=/corpus/...`) in containerized deployments, the "single
machine" assumption is no longer fully right -- a redeploy creates a NEW
container (a NEW PID namespace) on the SAME host, and the `pid` number in the
old lock can match a COMPLETELY UNRELATED (but genuinely alive) process in the
new container, so `psutil.pid_exists()` returns a false positive. Fix: if the
`hostname` that wrote the lock/with the lock differs from ours
(`socket.gethostname()`, the container id in Docker), the PID is declared STALE
WITHOUT even looking at it -- in this deployment `pipeline` is a SINGLE REPLICA
and the whole stack is redeployed together, so this is a CERTAIN signal (NOT a
TTL guess). Within the same container (tests, a single-host scenario) the
hostname always stays the same and the behavior is IDENTICAL to before.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Self

import psutil

DEFAULT_LOCK_PATH = Path(tempfile.gettempdir()) / "medrag-nightly.lock"


def _lock_path() -> Path:
    """Overridable via the `NIGHTLY_LOCK_PATH` env var (for tests and the
    deployment environment) -- else a fixed file in the system temp dir."""
    raw = os.environ.get("NIGHTLY_LOCK_PATH")
    return Path(raw) if raw else DEFAULT_LOCK_PATH


class NightlyLockHeld(RuntimeError):
    """Another `medrag-nightly` run is already running -- REJECT (I-11's
    decision; no wait queue)."""


def _read_lock(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _is_alive(pid: object) -> bool:
    return isinstance(pid, int) and psutil.pid_exists(pid)


def _is_stale(existing: dict) -> bool:
    """2026-08-27 fix: when the lock file moved to a persistent path (`/corpus`)
    in containerized deployments, the `pid` may not belong to our own container's
    PID namespace -- after a redeploy a NEW container counts PIDs from scratch, so
    the number in the old lock (e.g. 7) can correspond to a COMPLETELY UNRELATED
    process in the new container and cause a "still running" false positive
    (risked with a live test: not validated, spotted by code reading).

    Fix: if the `hostname` that wrote the lock (already recorded) differs from our
    current `socket.gethostname()`, it belongs to a different container
    generation -- in this deployment `pipeline` is a SINGLE REPLICA and the whole
    stack is redeployed together, so this is a CERTAIN signal (not a TTL guess):
    the old container is gone, no need to look at the PID. Within the same
    container (tests, single-host scenario) the hostname always stays the same --
    behavior is IDENTICAL to before (`_is_alive(pid)`), backward-compatible."""
    if existing.get("hostname") != socket.gethostname():
        return True
    return not _is_alive(existing.get("pid"))


class NightlyLock:
    """Context manager: `with NightlyLock():` -- raises `NightlyLockHeld` if the
    lock cannot be taken. If `path` is not given, `_lock_path()` (env var or the
    system temp dir) is used."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else _lock_path()
        self._acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_lock(self.path)
        if existing is not None:
            pid = existing.get("pid")
            if not _is_stale(existing):
                raise NightlyLockHeld(
                    f"baska bir medrag-nightly kosusu calisiyor (pid={pid}, "
                    f"host={existing.get('hostname')}, "
                    f"baslangic={existing.get('started_at')}) -- "
                    "REDDEDILDI (bekleme kuyrugu yok, I-11)")
            print(f"medrag-nightly: eski kilit STALE (pid={pid}, "
                  f"host={existing.get('hostname')}) -- devraliniyor")
        # write-then-replace: in a race condition (two processes calling acquire
        # at the same time) the last writer wins -- acceptable simplicity for a
        # single machine (the task text's "keep it simple" preference); real
        # multi-machine coordination would need a Redis lock, out of scope.
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=".tmp_lock_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }, f)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._acquired = True

    def release(self) -> None:
        if not self._acquired:
            return
        current = _read_lock(self.path)
        # We only delete OUR OWN lock -- by the time release() is called another
        # process may have declared it STALE and taken it over, and deleting
        # that process's lock would be wrong.
        if current is not None and current.get("pid") == os.getpid():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._acquired = False

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
