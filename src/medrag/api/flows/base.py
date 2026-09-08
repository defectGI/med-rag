"""The interface every intent-routed retrieval path (flow) implements.

A Flow owns everything between "the intent is already determined" and "a
final merged context, ready for the answering model, is produced": which
retrieval calls run, in what order/parallelism, and how their outputs are
merged. It does NOT call the answering model itself -- the Orchestrator does;
the split is deliberate: a flow's intermediate outputs (a rewrite, a SQL
string, extracted keywords, the first top_n call) never leak into the
answering model's own message history.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from medrag.api.retrieval.core import RetrievalResult
from medrag.api.session_state import PinnedFlow
from medrag.api.trace import TraceFn

ResultShape = Literal["zero", "one", "few", "many"]
"""Closed-set values of `FlowContext.result_shape` -- see
`compute_result_shape`."""


def compute_result_shape(count: int, *, few_max: int) -> ResultShape:
    """`count` (length of a flow's produced `results`) -> a closed-set bucket.
    The threshold (`few_max`) comes from `config/default.toml [result_shape]
    few_max` (root CONFIG.md: tuning isn't hardcoded in code) -- the calling
    flow carries it as a value injected into its own constructor (see
    flows/comparison.py, flows/doc_download.py, flows/recommendation.py,
    flows/sql_topn.py)."""
    if count <= 0:
        return "zero"
    if count == 1:
        return "one"
    if count <= few_max:
        return "few"
    return "many"


class FlowContext(BaseModel):
    """What a Flow hands back to the Orchestrator.

    ``results`` is the final, ranked evidence list the answering model will
    see.

    Strategy selection does NOT happen here -- since one flow (e.g.
    `sql_topn`) can serve multiple intents (`product_fact`, `aggregation`,
    ...), the choice of which `strategies/<key>.md` to read is made by the
    Orchestrator directly, from the classified `IntentLabel`, not by the flow
    (see `orchestrator.py::Orchestrator.handle`). This keeps the "one intent,
    one file" principle (`strategies/README.md`) independent of flows being
    shared.
    """

    results: list[RetrievalResult] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)
    """A flow-SPECIFIC, opaque workspace -- the Orchestrator/AnsweringModel
    NEVER read/interpret it (it does not REACH the answering model;
    `_render_context` only uses `results`); it exists only to carry data from
    the SAME flow's `run()`/`run_pinned()` to its own `pin_request()` within
    the same turn (see `SelfPinningFlow`) -- think of it as the
    temporary/raw form, on this turn, of `PinnedFlow.expected` (e.g.
    `RecommendationFlow`'s answers collected so far + which question was
    asked, see flows/recommendation.py). Like `PinnedFlow.expected`, the
    generic mechanism doesn't know/validate its content; it only carries it
    verbatim."""
    suggested_next_flow: str | None = None
    """A flow can mark here, while producing its own results, the offer "next
    turn, if the user wants, go directly to this flow with these candidates"
    (e.g. `RecommendationFlow`, when suggesting 2+ similar products -- see
    flows/recommendation.py). The `Orchestrator` copies it into
    `SessionState.suggested_next_flow` (see orchestrator.py); unlike
    `PinnedFlow` there is NO lifecycle (check_pin/pin_request) -- one-shot,
    reset next turn no matter what the user says. `None` = no offer (default,
    zero behavior change for ALL other flows)."""
    suggested_next_candidates: list[str] = Field(default_factory=list)
    """Candidates (e.g. product codes) to pre-fill the target flow with when
    `suggested_next_flow` is set -- see
    `SessionState.suggested_next_candidates`."""

    result_shape: ResultShape | None = None
    """A structured signal derived from the flow RESULT, summarizing
    `results`' length into a closed bucket (zero/one/few/many) -- unlike
    `scratch`, this DOES REACH the answering model (via
    `answering_model.py::_build_messages`) as a short note (see
    answering_model.py::_result_signal_note). `None` = this flow doesn't fill
    the signal yet (default, zero behavior change)."""
    residual: str | None = None
    """Leftover text a flow's deterministic parser could NOT CONSUME from the
    query (e.g. what remains after extracting product code/doc_type/family) --
    `None` if empty. SAME principle as `result_shape`: it summarizes the flow
    RESULT, it isn't an internal computation step."""
    parse_note: str | None = None
    """A closed-set parsing note (e.g. `"near_miss_product_code"` --
    distinguishing "no code found while a token that LOOKS like a product code
    (e.g. 'DE12') was present" from landing in the same bucket with nothing at
    all (`result_shape="zero"`)). The value set is flow-specific (see the
    relevant flow module's docstring); `None` = no note."""


@runtime_checkable
class Flow(Protocol):
    """The single method every intent-routed retrieval path implements.

    ``on_trace`` is optional: when given, the flow publishes live progress
    events while running (SQL generation, top_n calls, ...; see
    `trace.py`). When passed ``None`` (the default), behavior and return value
    are exactly the same -- trace is a purely observational layer."""

    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext: ...


@runtime_checkable
class PinnableFlow(Protocol):
    """Optional capability: a flow takes over, IN PERSON, the "is this turn
    still about what the pin waits for, or is the pin broken" check for a
    session pinned to it. The `Orchestrator` calls this INSTEAD of the full
    reconcile+classify chain, as a cheap check.

    `Orchestrator._check_pin` treats the pin as GONE on the SAFE side
    ("broken") when a flow does NOT implement this protocol or when
    `check_pin` raises, and drops to the full pipeline -- the user never gets
    stuck (escape-hatch default, see orchestrator.py).

    The protocol's ORIGINAL design deliberately did NOT require an LLM call:
    although the signature is `async`, the expected implementation was pure
    code (e.g. checking whether the product codes in `pinned.expected` still
    occur in this turn's query) -- repeating the reconciler's full LLM
    round-trip would have defeated the mechanism's purpose (cheapness) from
    the start.

    **REVISED:** "is it still relevant" is really a natural-language
    interpretation job -- it can't be imitated with a fixed word list (no
    expression outside the list can ever be classified correctly; the pin's
    trigger and end behavior was too dominant, and the cancel-word logic was
    hardcoded/inelegant). The `check_pin` of
    `RecommendationFlow`/`ComparisonFlow`/`DocDownloadFlow` now delegates to
    the shared `continuation.ContinuationChecker` (see that module) --
    cheapness was traded for correctness. The protocol's SIGNATURE is
    UNCHANGED (`async def check_pin(...) -> bool`); only the "be pure code"
    ADVICE is no longer mandatory: when writing a new flow, starting with pure
    code is still a sensible default, but an LLM call isn't blocked where the
    three flows above needed one. `Orchestrator._check_pin`'s
    "exception -> pin broken" escape hatch (see orchestrator.py) is UNCHANGED
    -- a network/parse error still drops to the full pipeline without ever
    sticking the user."""

    async def check_pin(self, query: str, pinned: PinnedFlow) -> bool: ...


@runtime_checkable
class SelfPinningFlow(Protocol):
    """Optional capability: a flow decides FOR ITSELF, AFTER its own `run()`
    completes, whether the session gets pinned to it for the next turn (or
    whether the current pin is cleared). `PinnableFlow.check_pin` answers "is
    the pin still valid while it exists"; this protocol answers "should the pin
    be ESTABLISHED/UPDATED/REMOVED now" -- together they cover a flow's full
    pin LIFECYCLE (establish -> maintain -> release).

    The `Orchestrator` calls this only for flows that implement it
    (`isinstance` check) -- a flow that doesn't never touches
    `session_state.pinned` through this mechanism, and flows that only use the
    `PinnableFlow` path keep their behavior EXACTLY (see orchestrator.py,
    `_handle_full_pipeline`/`_answer_pinned`).

    Return `None` = no pin / remove the pin; `PinnedFlow(...)` =
    ESTABLISH/UPDATE this pin. `async` -- so the flow can reuse its own
    (default deterministic/cheap) parsing logic (e.g. `ComparisonFlow`'s
    `splitter`) -- but it does NOT require an LLM round-trip (see the
    `ComparisonFlow.pin_request` docstring)."""

    async def pin_request(self, query: str, context: FlowContext) -> PinnedFlow | None: ...


@runtime_checkable
class PinAwareRunFlow(Protocol):
    """Optional capability: a flow has `run_pinned()` called INSTEAD of
    `run()` while pinned, because this turn's raw query (`query`) is
    MEANINGLESS on its own -- it gains meaning only merged with the state
    collected in PREVIOUS turns (`PinnedFlow.expected`). Flows like
    `ComparisonFlow` do NOT need this (they recompute a full, independent
    split from each turn's raw query) -- therefore the `Flow.run(query,
    on_trace)` protocol was NOT REPLACED; instead a SEPARATE method was added
    that only the flows needing it opt into (detected via isinstance).

    `Orchestrator._answer_pinned` calls this INSTEAD of `flow.run(...)` when
    the flow implements it (see orchestrator.py) -- flows that don't see zero
    behavior change. The schema of `pinned.expected` is entirely flow-SPECIFIC
    for the implementing flow (e.g. `RecommendationFlow`'s
    `{"collected": ..., "asked": ..., "asking": ..., "raw_query": ...}`
    schema, see flows/recommendation.py) -- the generic mechanism (this
    protocol + `PinnedFlow`) never interprets it."""

    async def run_pinned(
        self, query: str, pinned: PinnedFlow, on_trace: TraceFn | None = None
    ) -> FlowContext: ...


__all__ = [
    "Flow",
    "FlowContext",
    "PinAwareRunFlow",
    "PinnableFlow",
    "ResultShape",
    "SelfPinningFlow",
    "compute_result_shape",
]
