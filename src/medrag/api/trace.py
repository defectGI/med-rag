"""Trace event emission: an optional, observation-only callback threaded
through the Orchestrator and every Flow so a UI can show "what's happening in
the background" live (SQL queries + returned chunks + intermediate stages in
real time).

It NEVER affects the answer itself -- with `on_trace=None` (or a no-op that is
never called) the behavior is exactly the same. This is SEPARATE from the
decision that flow internal steps never leak into the answering model's
message history (orchestrator.py): traces go to the user (browser), not to
the model.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from medrag.api.retrieval.core import RetrievalResult

# (step_name, data) -- data may be plain text or a JSON-serializable
# structure such as the output of results_to_trace.
TraceFn = Callable[[str, Any], None]

_T = TypeVar("_T")


def emit_timing(on_trace: TraceFn | None, stage: str, t0: float) -> None:
    """Publishes the elapsed time between `t0` (a `time.perf_counter()`
    moment) and now as a `timing` trace event -- `{"stage": ..., "seconds": ...}`.

    The single point shared by the Orchestrator (L0-L4 outer stages) and the
    Flows (in-flow sub-stages: rewrite/top_n/sql/...), so every intermediate
    stage shows live with its duration recorded. Does nothing when `on_trace`
    is absent -- timing the call is the caller's job; this helper only handles
    emission, so the format (`round(..., 2)`, `stage`/`seconds` keys) stays
    defined in one place and matches the webapp's `timing` cards exactly."""
    if on_trace:
        on_trace("timing", {"stage": stage, "seconds": round(time.perf_counter() - t0, 2)})


async def timed(on_trace: TraceFn | None, stage: str, coro: Awaitable[_T]) -> _T:
    """Awaits `coro`, returns its value unchanged, and emits a `timing` event
    for `stage` on completion. For measuring the individual wall-clock time of
    branches running in parallel inside `asyncio.gather` (e.g. the first
    top_n alongside SQL in `sql_topn`): wrap each branch in its own
    `timed(...)`.

    Observation only: with `on_trace=None` the behavior/return value is
    exactly `await coro` (only the emission is skipped)."""
    t0 = time.perf_counter()
    try:
        return await coro
    finally:
        emit_timing(on_trace, stage, t0)


def results_to_trace(results: list[RetrievalResult], *, snippet_len: int = 1200) -> list[dict]:
    """Condenses a list of `RetrievalResult` into a small trace-panel-friendly
    list -- not the full raw metadata, just id/score/truncated text."""
    trace = []
    for r in results:
        text = r.text
        if len(text) > snippet_len:
            text = text[:snippet_len] + "…"
        trace.append({"id": r.id, "score": round(r.score, 4), "text": text})
    return trace


__all__ = ["TraceFn", "emit_timing", "results_to_trace", "timed"]
