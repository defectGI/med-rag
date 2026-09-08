"""Gate that serializes the GPU-heavy (linking->generation) SQL chain across
the whole process.

WHY: the SQL half of `sql_topn` runs two large models (linking=gemma4:31b ->
generation=qwen2.5-coder:32b) in the SAME Ollama instance. If both models
don't fit in a single 32GB VRAM, two SQL chains racing for the GPU at the
same time put Ollama into a constant offload/reload (thrashing) loop -- which
can be SLOWER than running serially. The requirement: handle it cleanly if
they don't fit -- i.e. the two large-model calls never race concurrently, and
they queue in orderly fashion whether they fit or not.

CONCURRENCY comes from two sources (see webapp.py): (1) within a single
request, `ComparisonFlow`'s `side_a || side_b` gather -- SAME event loop;
(2) two different users asking at once -- Flask `threaded=True` + each
request has its own `asyncio.run`, i.e. SEPARATE thread + SEPARATE loop.
Hence the gate is NOT `asyncio.Lock` (bound to a loop, not thread-safe, would
collapse under (2)) but `threading.Semaphore`-based -- one counter covers
both sources.

The lock is taken WITHOUT blocking the event loop: the blocking `acquire()`
is moved to a worker thread via `asyncio.to_thread`. So `side_a`/`side_b` on
the same loop don't deadlock (the loop keeps answering -- one can progress
while the other waits for the lock), and users on separate threads queue
nicely on the same semaphore.

`top_n` (embedding search, cheap, <1GB VRAM) is OUTSIDE this gate -- only the
SQL retriever is wrapped; top_n keeps its real parallelism.

`timeout_seconds` -- the semaphore is ALWAYS released within a finite time.
The original root cause (chatbot/text2sql/config.toml `[linking] max_tokens`
mistakenly set to 262144, forcing `num_predict` to that value and driving
Ollama into thrashing) was fixed separately, but that was only THIS
incident's source -- a single slow/hung SQL call (bad config, network issue,
GPU jamming unexpectedly, whatever) under `concurrency=1` would hold this
semaphore INDEFINITELY and lock out ALL other SQL-based users too. On timeout
`asyncio.wait_for` still runs `finally` (semaphore released) and
`TimeoutError` propagates as a normal exception -- the existing Orchestrator/
webapp error path (log + short error message to the user) kicks in unchanged;
the service ITSELF never crashes or wedges. `None` (default) = unbounded,
legacy behavior -- backward compatible.
"""

from __future__ import annotations

import asyncio
import threading

from medrag.api.retrieval.core import RetrievalResult, Retriever


class SqlGateTimeout(TimeoutError):
    """A single SQL retrieval call exceeded `timeout_seconds` -- the semaphore
    was already released (see the `finally` in
    `SerializedRetriever.retrieve`); only THIS request fails and the gate
    opens for other users immediately."""


class SerializedRetriever:
    """Puts a `Retriever` behind a process-wide `threading.Semaphore`.

    Implements the `Retriever` protocol; passes `retrieve` through VERBATIM
    with `**kwargs` -- `DbQueryRetriever`'s optional `on_sql`/`on_stage` hooks
    and `SqlTopNFlow._run_sql`'s TypeError-probe fallback are preserved
    intact.

    `concurrency` is the semaphore counter: 1 = full serialization (the two
    large models never race); if both models fit on one GPU it can be 2 to
    try real batch parallelism (config `[session] sql_concurrency`).

    `timeout_seconds`: `None` (default) = unbounded wait (legacy behavior).
    If given and `self._inner.retrieve(...)` doesn't finish in that time,
    `SqlGateTimeout` is raised and the semaphore is released immediately (see
    the docstring above).
    """

    def __init__(
        self, inner: Retriever, *, concurrency: int = 1, timeout_seconds: float | None = None
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"sql_concurrency >= 1 olmalı, {concurrency} verildi")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds > 0 olmalı, {timeout_seconds} verildi")
        self._inner = inner
        self._sem = threading.Semaphore(concurrency)
        self._timeout = timeout_seconds

    async def retrieve(self, query: str, *, k: int = 10, **kwargs) -> list[RetrievalResult]:
        # Move the blocking acquire to a worker thread -- doesn't block the
        # event loop, so side_a/side_b on the same loop serialize without
        # deadlocking.
        await asyncio.to_thread(self._sem.acquire)
        try:
            coro = self._inner.retrieve(query, k=k, **kwargs)
            if self._timeout is None:
                return await coro
            try:
                return await asyncio.wait_for(coro, timeout=self._timeout)
            except TimeoutError as exc:
                raise SqlGateTimeout(
                    f"SQL retrieval {self._timeout}s içinde bitmedi (gate serbest bırakıldı)."
                ) from exc
        finally:
            self._sem.release()


__all__ = ["SerializedRetriever"]
