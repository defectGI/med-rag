"""Single-file advisory lock: the registry lock via `fcntl.flock`.

The registry (document_nodes.json) is updated by TWO writers: api (upload/delete)
and worker (parse/chunk blocks). Atomic replace alone is not enough --
if two writers read and write different copies at the same time, one's record is
LOST. `flock` provides inter-process mutual exclusion on Linux; the lock file is
the registry's sibling `<name>.lock`.

The api side takes the SAME lock at the SAME path (same contract); the lock
file name is `<registry name>.lock`.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def lock_path_for(target: Path) -> Path:
    return target.with_name(target.name + ".lock")


@contextmanager
def file_lock(target: Path) -> Iterator[Path]:
    """Inter-process exclusive lock for `target` (the file being protected)."""
    lp = lock_path_for(target)
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield lp
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


__all__ = ["file_lock", "lock_path_for"]
