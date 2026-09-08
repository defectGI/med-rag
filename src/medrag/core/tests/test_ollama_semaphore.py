"""Proves that `core/ollama_semaphore.py` enforces its concurrency limit
without ever sending a request to the real Ollama (fake workload -- HARD RULE).

Scope: the semaphore alone -- that N concurrent calls are held to N and the
N+1st is made to wait. Whether the 120 s threshold holds under a real Ollama
load is NOT in scope for these tests (it is measured on the GPU/service
machine).
"""

from __future__ import annotations

import threading
import time

import pytest

from medrag.core.ollama_semaphore import (
    get_ollama_semaphore,
    ollama_call_slot,
    reset_ollama_semaphore_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Resets the process-wide singleton before/after every test -- so tests
    don't pollute each other's `max_concurrent` choice."""
    reset_ollama_semaphore_for_tests()
    yield
    reset_ollama_semaphore_for_tests()


def test_n_concurrent_calls_never_exceed_the_limit():
    """8 fake "Ollama calls" are run in parallel with a limit of N=3; at no
    moment does the number running simultaneously EXCEED 3 (fake workload: just
    a short sleep)."""
    max_concurrent = 3
    total_calls = 8
    current = 0
    peak = 0
    lock = threading.Lock()

    def fake_ollama_call() -> None:
        nonlocal current, peak
        with ollama_call_slot(max_concurrent=max_concurrent):
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.05)  # NO real request, only there to observe concurrency
            with lock:
                current -= 1

    threads = [threading.Thread(target=fake_ollama_call) for _ in range(total_calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert peak == max_concurrent


def test_n_plus_one_th_call_blocks_until_a_slot_frees():
    """After the 2 slots of N=2 are full, the 3rd call WAITS; it does not
    proceed until a slot is released, and proceeds immediately once one is."""
    max_concurrent = 2
    sem = get_ollama_semaphore(max_concurrent=max_concurrent)
    assert sem is get_ollama_semaphore()  # same singleton, not rebuilt on the second call

    holder_a_ready = threading.Event()
    holder_b_ready = threading.Event()
    release_holders = threading.Event()
    third_acquired = threading.Event()

    def holder(ready_event: threading.Event) -> None:
        with ollama_call_slot(max_concurrent=max_concurrent):
            ready_event.set()
            release_holders.wait(timeout=5)

    def third_caller() -> None:
        with ollama_call_slot(max_concurrent=max_concurrent):
            third_acquired.set()

    t_a = threading.Thread(target=holder, args=(holder_a_ready,))
    t_b = threading.Thread(target=holder, args=(holder_b_ready,))
    t_a.start()
    t_b.start()
    assert holder_a_ready.wait(timeout=5)
    assert holder_b_ready.wait(timeout=5)

    t_c = threading.Thread(target=third_caller)
    t_c.start()

    # Both slots are full -- the third call must NOT PROCEED within a short window.
    assert not third_acquired.wait(timeout=0.3)

    # Once a slot is released, the third call must proceed immediately.
    release_holders.set()
    assert third_acquired.wait(timeout=5)

    for t in (t_a, t_b, t_c):
        t.join(timeout=5)
        assert not t.is_alive()


def test_default_max_concurrent_is_read_from_config():
    """If `max_concurrent` is not given, `[pipeline] ollama_max_concurrent` from
    config is read (the real value in default.toml is 2, documented with the
    feature)."""
    from medrag.core.config.schema import load_core_config

    expected = load_core_config().pipeline.ollama_max_concurrent
    sem = get_ollama_semaphore()
    # threading.Semaphore doesn't expose its internal counter; the indirect
    # proof: filling `expected` slots makes `acquire(blocking=False)` succeed
    # for all of them, ONE MORE must FAIL.
    acquired = [sem.acquire(blocking=False) for _ in range(expected)]
    assert all(acquired)
    assert sem.acquire(blocking=False) is False
    for _ in range(expected):
        sem.release()


def test_second_call_with_explicit_max_concurrent_rebuilds_singleton():
    """For tests: if `max_concurrent` is passed explicitly with a DIFFERENT
    value, the singleton is rebuilt with that new ceiling (this never happens
    in production -- test isolation only)."""
    sem_a = get_ollama_semaphore(max_concurrent=1)
    sem_b = get_ollama_semaphore(max_concurrent=4)
    assert sem_a is not sem_b
    acquired = [sem_b.acquire(blocking=False) for _ in range(4)]
    assert all(acquired)
    assert sem_b.acquire(blocking=False) is False
    for _ in range(4):
        sem_b.release()


def test_slot_is_released_even_if_the_wrapped_call_raises():
    """The slot is released even if the work inside `ollama_call_slot` raises --
    otherwise a single error would have blocked the whole nightly run."""
    max_concurrent = 1

    with pytest.raises(RuntimeError, match="sahte hata"), ollama_call_slot(
        max_concurrent=max_concurrent
    ):
        raise RuntimeError("sahte hata")

    sem = get_ollama_semaphore(max_concurrent=max_concurrent)
    assert sem.acquire(blocking=False) is True
    sem.release()
