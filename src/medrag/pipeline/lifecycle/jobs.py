"""File-based job queue (A6): no redis, single-worker assumption.

The queue = the `{job_id}.json` files in a directory. The claim is atomic:
it is taken by renaming the file `.json` -> `.claim` (which makes taking the
same job twice impossible at the OS level). When the job finishes, `.claim`
is deleted.

Interrupt recovery: if the worker dies the `.claim` file stays on disk; on
restart `claim_next` first returns the `.claim` files -- jobs are not lost
(actions are idempotent, so running twice is safe: process re-parses, delete
re-deletes and silently no-ops).

Order: job_id is timestamped (`YYYYmmdd-HHMMSS-<uuid8>`), so alphabetic sort
of the file names = order of enqueue (uuids may interleave within the same
second; sufficient for a single user).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("medrag.pipeline.lifecycle.jobs")

#: Actions that can be marked in the queue. api writes, worker interprets.
ACTIONS = ("process", "delete")


@dataclass(frozen=True)
class Job:
    job_id: str
    action: str
    doc_id: str
    rel_path: str
    enqueued_at: str

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "action": self.action,
            "doc_id": self.doc_id,
            "rel_path": self.rel_path,
            "enqueued_at": self.enqueued_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        return cls(
            job_id=str(data["job_id"]),
            action=str(data["action"]),
            doc_id=str(data["doc_id"]),
            rel_path=str(data.get("rel_path") or ""),
            enqueued_at=str(data.get("enqueued_at") or ""),
        )


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def enqueue(jobs_dir: Path, *, action: str, doc_id: str, rel_path: str) -> Job:
    """Writes a new job file; `isler/{job_id}.json`."""
    if action not in ACTIONS:
        raise ValueError(f"bilinmeyen eylem: {action!r} (geçerli: {ACTIONS})")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    job = Job(
        job_id=now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8],
        action=action,
        doc_id=doc_id,
        rel_path=rel_path,
        enqueued_at=now.isoformat(timespec="seconds"),
    )
    _atomic_write_json(jobs_dir / f"{job.job_id}.json", job.to_dict())
    logger.info("iş kuyruğa girildi: %s %s (%s)", action, doc_id, job.job_id)
    return job


def pending_count(jobs_dir: Path) -> int:
    if not jobs_dir.is_dir():
        return 0
    return sum(1 for p in jobs_dir.iterdir() if p.suffix in (".json", ".claim"))


def _parse_job_file(path: Path) -> Job | None:
    try:
        return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.warning("okunamayan iş dosyası atlandı: %s", path, exc_info=True)
        return None


def recover(jobs_dir: Path) -> int:
    """Interrupt recovery: turns leftover `.claim` files back into `.json`.

    Called once at worker START -- jobs interrupted in a previous run return to
    the queue (actions are idempotent). It is NOT EMBEDDED in `claim_next`:
    if it were there, returning a `.claim` would allow the SAME job to be
    claimed TWICE (breaking the atomic single-winner guarantee). Return value:
    the number of jobs recovered."""
    if not jobs_dir.is_dir():
        return 0
    n = 0
    for claim in jobs_dir.glob("*.claim"):
        try:
            claim.rename(claim.with_suffix(".json"))
            n += 1
        except OSError:
            continue
    return n


def claim_next(jobs_dir: Path) -> Job | None:
    """Atomically claims the next job (`.json` -> `.claim`).

    Only `.json` files are tried (oldest first); the rename is atomic at the OS
    level -- two processes cannot take the same job. Unreadable files are
    deleted (one corrupt file must not block the queue forever). Left
    `.claim` files are NOT CLAIMED -- they are `recover()`'s job."""
    if not jobs_dir.is_dir():
        return None

    for path in sorted(jobs_dir.glob("*.json")):
        claim = path.with_suffix(".claim")
        try:
            os.rename(path, claim)  # atomik iddia
        except OSError:  # başka bir süreç aldı
            continue
        job = _parse_job_file(claim)
        if job is None:
            claim.unlink(missing_ok=True)
            continue
        return job
    return None


def finish_job(jobs_dir: Path, job: Job) -> None:
    """Deletes the job's `.claim` file (idempotent)."""
    (jobs_dir / f"{job.job_id}.claim").unlink(missing_ok=True)


__all__ = ["ACTIONS", "Job", "claim_next", "enqueue", "finish_job", "pending_count", "recover"]
