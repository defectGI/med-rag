"""`SerializedRetriever.timeout_seconds` -- the service must not be brought down
by SQL errors. The structural guarantee locked in here: a single slow/hanging
SQL call cannot hold the gate shut indefinitely for other users."""

import asyncio

import pytest

from medrag.api.gpu_gate import SerializedRetriever, SqlGateTimeout


class _SlowRetriever:
    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def retrieve(self, query, k=10, **kwargs):
        await asyncio.sleep(self._delay)
        return ["ok"]


def test_no_timeout_by_default_preserves_old_behavior():
    # If timeout_seconds is not given (legacy caller code), behavior is identical --
    # backward compatibility (see test_factory.py::test_serialized_retriever_passes_through_kwargs).
    wrapped = SerializedRetriever(_SlowRetriever(0.01), concurrency=1)
    out = asyncio.run(wrapped.retrieve("q"))
    assert out == ["ok"]


def test_fast_call_succeeds_within_timeout():
    wrapped = SerializedRetriever(_SlowRetriever(0.01), concurrency=1, timeout_seconds=1.0)
    out = asyncio.run(wrapped.retrieve("q"))
    assert out == ["ok"]


def test_slow_call_raises_sql_gate_timeout_not_hang():
    wrapped = SerializedRetriever(_SlowRetriever(1.0), concurrency=1, timeout_seconds=0.05)
    with pytest.raises(SqlGateTimeout):
        asyncio.run(wrapped.retrieve("q"))


def test_timeout_releases_semaphore_for_next_caller():
    # The real guarantee: even when one request times out, the gate opens
    # IMMEDIATELY -- the next caller is not locked out indefinitely (the exact
    # opposite of the root cause).
    wrapped = SerializedRetriever(_SlowRetriever(1.0), concurrency=1, timeout_seconds=0.05)
    with pytest.raises(SqlGateTimeout):
        asyncio.run(wrapped.retrieve("first"))

    fast = SerializedRetriever(_SlowRetriever(0.01), concurrency=1, timeout_seconds=1.0)
    fast._sem = wrapped._sem  # share the same semaphore, just like the process-wide gate
    out = asyncio.run(fast.retrieve("second"))
    assert out == ["ok"]


def test_invalid_timeout_rejected():
    with pytest.raises(ValueError):
        SerializedRetriever(_SlowRetriever(0.01), concurrency=1, timeout_seconds=0)
    with pytest.raises(ValueError):
        SerializedRetriever(_SlowRetriever(0.01), concurrency=1, timeout_seconds=-5)
