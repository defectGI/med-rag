"""Job queue worker (A6): `python -m medrag.pipeline.lifecycle.worker`.

It takes the job files in `ISLER_DIR` in order (jobs.claim_next) and runs
`runner.process_document`/`runner.delete_document`. State transitions are
written to `/durum/<doc_id>.json`; api reads those files.

Concurrency: a SINGLE thread, sequential execution. VLM/embedding calls are
already heavy; at the 500+ document scale a single queue gives a predictable
cost (PARSED+EMBED concurrency produces messy failure modes). If needed, the
number of workers per queue can be added later.

Interrupt recovery: `.claim` files left half-done are re-processed at restart
(actions are idempotent). If a job fails it is dropped and the state turns to
`error` -- retrying is a user action (re-uploading the file already means
"reprocess" under A4 semantics).
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from medrag.pipeline.lifecycle import jobs as job_queue
from medrag.pipeline.lifecycle import runner
from medrag.pipeline.lifecycle.durum import write_status
from medrag.pipeline.lifecycle.paths import durum_dir, isler_dir

logger = logging.getLogger("medrag.pipeline.lifecycle.worker")

#: Sleep duration when the queue is empty (sec). No wait when jobs exist.
POLL_INTERVAL_SECONDS = 1.0


class _Stopper:
    """SIGTERM/SIGINT -> graceful stop: exit once the running job finishes."""

    def __init__(self) -> None:
        self.stop = False

    def request(self, *_args) -> None:
        self.stop = True


def process_job(job: job_queue.Job, *, runner_mod=None) -> None:
    """Runs a single job; catches the exception and turns it into `error` state
    (the worker itself NEVER crashes -- the queue stays alive). `runner_mod` is
    injectable for tests; if not given the module-level `runner` is used
    (resolved at call time -- no default-arg that would break monkeypatching).
    This is the worker's own resilience contract: no job let it die."""
    runner_mod = runner_mod or runner
    ddir = durum_dir()
    try:
        if job.action == "process":
            write_status(ddir, job.doc_id, "queued", detail="işleniyor",
                         rel_path=job.rel_path)
            runner_mod.process_document(
                job.doc_id,
                status=lambda state, detail=None, **kw:
                    write_status(ddir, job.doc_id, state, detail=detail,
                                 rel_path=job.rel_path)
                    if not kw.get("silent") else None,
            )
        elif job.action == "delete":
            runner_mod.delete_document(
                job.doc_id,
                status=lambda state, detail=None, **kw:
                    write_status(ddir, job.doc_id, state, detail=detail,
                                 rel_path=job.rel_path)
                    if not kw.get("silent") else None,
            )
        else:
            logger.warning("bilinmeyen eylem atlandı: %r (%s)", job.action, job.job_id)
    except Exception as exc:
        logger.exception("iş başarısız: %s %s", job.action, job.doc_id)
        write_status(ddir, job.doc_id, "error",
                     detail=f"{type(exc).__name__}: {exc}",
                     rel_path=job.rel_path)
    finally:
        job_queue.finish_job(isler_dir(), job)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("=== lifecycle worker başlıyor ===")

    # .env files (cli + parser) are loaded early: the parse stage runs with the
    # same env; a missing required variable fails loudly here.
    runner._bootstrap_env()

    stopper = _Stopper()
    signal.signal(signal.SIGTERM, stopper.request)
    signal.signal(signal.SIGINT, stopper.request)

    jdir = isler_dir()
    # Interrupt recovery: `.claim` jobs left over from a previous run go back to
    # the queue (actions are idempotent). A separate step -- it does not break
    # claim_next's single-winner guarantee.
    kalan = job_queue.recover(jdir)
    if kalan:
        logger.info("kesinti toparlama: %d iş kuyruğa geri alındı", kalan)
    logger.info("kuyruk: %s | durum: %s", jdir, durum_dir())

    while not stopper.stop:
        job = job_queue.claim_next(jdir)
        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        logger.info("iş alındı: %s %s (%s)", job.action, job.doc_id, job.job_id)
        process_job(job)

    logger.info("worker durdu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
