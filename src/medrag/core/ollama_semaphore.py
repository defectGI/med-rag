"""Shared semaphore limiting the number of concurrent calls going to Ollama.

Why: the nightly run (parse + facts stages) uses Ollama SHARED with the live
service. While the nightly run is going, the service keeps sending requests to
the same Ollama (chatbot/api) -- if parse/facts fired their own Ollama calls
with unbounded parallelism, a live request's queue wait could exceed the 120 s
threshold the nightly run is held to.

What this module is: ONE, SHARED concurrency limit (`threading.Semaphore`) that
the parser/facts side wraps its own Ollama call in. Every caller inside the
process shares the SAME semaphore (`get_ollama_semaphore()` returns a single
singleton) -- each module setting up its own semaphore would make the limit
ineffective (N separate limits = unlimited).

What it is NOT: this module sends NO HTTP request to Ollama and makes no real
model call. The limit value is read from `[pipeline] ollama_max_concurrent`
(core/config/default.toml).

Wiring note (the same pattern as the surrounding config constants -- the
constant lives here, wiring it into the real call sites
(`urun/pipeline/parser/llm/openai_compat.py`, the facts-side LLM calls) is a
separate integration step and was left out of this scope; the facts side is
already blocked at import time for its own unrelated reasons). Example usage:

    from medrag.core.ollama_semaphore import ollama_call_slot

    with ollama_call_slot():
        response = client.chat(...)  # the real Ollama request

Verification limit (HARD RULE -- NO real LLM/Ollama call on this machine):
this module's tests prove, with a fake workload, that the semaphore forces N
concurrent calls down to N and makes the N+1st wait. Whether the 120 s
threshold holds under a real Ollama load can only be measured on the
GPU/service machine.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_lock = threading.Lock()
_semaphore: threading.Semaphore | None = None
_semaphore_max: int | None = None


def _max_concurrent_from_config() -> int:
    # Local import: core/config already lives in the core layer, but it is read
    # only when needed so the module's import chain isn't grown for nothing.
    from medrag.core.config.schema import load_core_config

    return load_core_config().pipeline.ollama_max_concurrent


def get_ollama_semaphore(max_concurrent: int | None = None) -> threading.Semaphore:
    """Returns the SHARED, process-wide semaphore.

    On the first call, if `max_concurrent` is not given, it is read from config
    and the semaphore is created with that value; later calls return the SAME
    semaphore (as long as no parameter is passed) -- rebuilding it would lose
    slots that are already taken. If `max_concurrent` IS passed explicitly and
    differs from the previous value (for tests), the semaphore is REBUILT with
    that new value.
    """
    global _semaphore, _semaphore_max
    with _lock:
        if max_concurrent is not None:
            resolved = max_concurrent
        elif _semaphore is not None:
            return _semaphore
        else:
            resolved = _max_concurrent_from_config()
        if _semaphore is None or _semaphore_max != resolved:
            _semaphore = threading.Semaphore(resolved)
            _semaphore_max = resolved
        return _semaphore


@contextmanager
def ollama_call_slot(max_concurrent: int | None = None) -> Iterator[None]:
    """Context manager to wrap ONE call going to Ollama.

    Blocks until a slot frees up (queues), never swallows an exception --
    `finally` releases the slot under every condition.
    """
    sem = get_ollama_semaphore(max_concurrent)
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def reset_ollama_semaphore_for_tests() -> None:
    """For tests only: resets the process-wide singleton.

    Used between tests to try a different `max_concurrent` scenario or to avoid
    being affected by the singleton another test left behind.
    """
    global _semaphore, _semaphore_max
    with _lock:
        _semaphore = None
        _semaphore_max = None


__all__ = [
    "get_ollama_semaphore",
    "ollama_call_slot",
    "reset_ollama_semaphore_for_tests",
]
