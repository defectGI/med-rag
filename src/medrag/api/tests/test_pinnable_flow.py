"""`PinnableFlow` is only a `runtime_checkable` Protocol -- its real behavior
is already tested via `Orchestrator._check_pin` in `test_orchestrator.py`.
This file only locks in that the protocol itself does structural typing as
expected (correctly distinguishing implementing/non-implementing objects)."""

from medrag.api.flows.base import FlowContext, PinnableFlow
from medrag.api.session_state import PinnedFlow


class _PlainFlow:
    """Implements only `Flow`; NO `check_pin` -- at this STAGE no flow in the
    real repo implements `PinnableFlow`; this class represents that case."""

    async def run(self, query: str, on_trace=None) -> FlowContext:
        return FlowContext(results=[])


class _PinnableFlowImpl:
    async def run(self, query: str, on_trace=None) -> FlowContext:
        return FlowContext(results=[])

    async def check_pin(self, query: str, pinned: PinnedFlow) -> bool:
        return True


def test_plain_flow_is_not_pinnable():
    assert not isinstance(_PlainFlow(), PinnableFlow)


def test_flow_with_check_pin_is_pinnable():
    assert isinstance(_PinnableFlowImpl(), PinnableFlow)
