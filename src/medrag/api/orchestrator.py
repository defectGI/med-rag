"""The single place joining intent classification, routing, flow execution,
conversation memory and the answering-model call.

Nothing a Flow does internally (a rewrite, a SQL string, extracted keywords,
an intermediate top_n call) reaches the answering model on its own; what
reaches it is only `FlowContext.results` + the matching `strategies/<key>.md`
guidance text + the REAL user/assistant messages of PREVIOUS turns
(`ConversationMemory`). The two are different things: flow internal steps
never enter memory, but previous turns' real question/answers do -- so the
conversation "flows".

Classifier/flow/answering model can each be replaced without touching this
boundary.

`on_trace` does NOT LEAK these intermediate steps to the answering model --
it shows them to the user (a live UI panel). The two mechanisms are fully
independent: traces go to the browser, `FlowContext`/`history` go to the
model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from medrag.api.continuation import ContinuationChecker
from medrag.api.conversation_log import CHANNEL_WEB
from medrag.api.flows.base import (
    Flow,
    FlowContext,
    PinAwareRunFlow,
    PinnableFlow,
    SelfPinningFlow,
)
from medrag.api.flows.doc_download import DocumentLookup
from medrag.api.memory import ConversationMemory, Message
from medrag.api.preamble import PreambleSettings, PreambleTicker
from medrag.api.reconciler import Reconciler
from medrag.api.retrieval.core import IntentLabel, IntentResult
from medrag.api.retrieval.modules.intent_classification import IntentClassifier
from medrag.api.router import Router
from medrag.api.session_state import PinnedFlow, SessionState, SessionStateStore
from medrag.api.trace import TraceFn, emit_timing

# Deliberately NOT retrieval/strategies/: that directory isn't shipped in the
# PyPI wheel (absent from retrieval/MANIFEST.in) and the label->strategy
# mapping is the consuming project's job anyway (retrieval ARCHITECTURE.md
# #6/#7). This component carries its own strategy files (see
# strategies/README.md). `strategies/` stayed as DATA in the old `chatbot/`
# directory at the repo root during the package restructure.
# parents[3] = repo root.
STRATEGIES_DIR = Path(__file__).resolve().parents[3] / "chatbot" / "strategies"

# A word set used to understand whether the user accepted the previous turn's
# handoff offer ("want to compare these?") on the next turn. Now engaged ONLY
# as a FALLBACK when `continuation_checker` is never injected (dev/test) -- the
# primary decision is delegated by `Orchestrator._is_affirmative` to the
# shared `ContinuationChecker` (same root rationale as the "REVISED" note in
# `flows/base.py::PinnableFlow`: a fixed word list was structurally inadequate
# for this soft-skill decision).
# NOTE: the set below is user-facing matching data -- keep as-is.
_AFFIRMATIVE_WORDS = {
    "evet", "tamam", "olur", "tabii", "tabi", "isterim", "lütfen", "evet lütfen",
    "karşılaştır", "kıyasla",
}


@runtime_checkable
class AnsweringModel(Protocol):
    """The final model call that returns to the user. Kept behind a protocol so
    tests can inject a fake implementation -- the real one
    (`answering_model.OpenAICompatAnsweringModel`) is a remote,
    OpenAI-compatible endpoint (GPU rule: never run here), configured through
    this component's own `.env`. ``history`` is the REAL messages of previous
    turns -- not mixed with flow internal steps."""

    async def answer(
        self,
        query: str,
        context: FlowContext,
        strategy_prompt: str,
        history: list[Message],
        *,
        session_goal: str = "",
        hedge: bool = False,
        followups: str = "",
        channel: str = CHANNEL_WEB,
    ) -> str: ...


# Strategy files can carry both the guidance text that goes to the model and
# developer notes (decision references, `flows/X.py` pointers, speculative
# '## Notes' bullets) -- EVERYTHING after this marker is a developer note and
# never reaches the model. The marker line is an HTML comment (so the file also
# looks clean in a markdown render).
_DEV_NOTES_MARKER = "<!-- CHATBOT:DEV-NOTES (modele gitmez) -->"


def load_strategy(strategy_key: str) -> str:
    """Reads `strategies/<strategy_key>.md`; returns an empty string if it
    isn't written yet (strategy drafts are filled in one by one, see
    strategies/README.md).

    If the file contains the `_DEV_NOTES_MARKER` line, only the section BEFORE
    it (the part sent to the model) is returned. If the marker is ABSENT
    (backward compatibility) the WHOLE file is returned as before -- this
    behavior must NOT break."""
    path = STRATEGIES_DIR / f"{strategy_key}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    marker_idx = text.find(_DEV_NOTES_MARKER)
    if marker_idx == -1:
        return text
    return text[:marker_idx].rstrip("\n")


class Orchestrator:
    """Wires the L0-L4 flow.

    `reconciler`/`session_state` are OPTIONAL: without them L0/L1 are skipped
    (the raw query is classified directly) -- old behavior, backward
    compatible. With them, every turn is first reconciled (state +
    de-ellipsis), then the RESOLVED query is classified and routed; the
    answering model frames the reply with the ongoing goal. State writing (L0)
    is logically parallel to answer production: this turn's answer is framed
    by the pre-turn goal, the state update is for the next turn.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        router: Router,
        answering_model: AnsweringModel,
        memory: ConversationMemory,
        *,
        reconciler: Reconciler | None = None,
        session_state: SessionStateStore | None = None,
        hedge_threshold: float = 0.5,
        continuation_checker: ContinuationChecker | None = None,
        channel: str = CHANNEL_WEB,
        preamble: PreambleSettings | None = None,
        document_lookup: DocumentLookup | None = None,
    ) -> None:
        self._classifier = classifier
        self._router = router
        self._model = answering_model
        self._memory = memory
        self._reconciler = reconciler
        self._session_state = session_state or SessionStateStore()
        self._hedge_threshold = hedge_threshold
        self._continuation_checker = continuation_checker
        # While the user talks about products (the sql_topn core shared by
        # product_fact/aggregation/visual_request) the bot does NOT SUGGEST
        # files -- `doc_download` is deliberately a SEPARATE flow and
        # `available_document_followups` (session_state.py) fills only if
        # `doc_download` ran in the PREVIOUS turn. `_document_hint` (below)
        # fills this gap in a RESTRICTED way: on this turn's single-product
        # results verified against the `document` table, if a real file exists
        # it appends a ONE-line suggestion. `None` = off, zero behavior change
        # (when not injected, as in dev/test).
        self._document_lookup = document_lookup
        # Which channel this Orchestrator INSTANCE serves -- webapp.py and
        # wa_bot.py each build a SEPARATE Orchestrator from
        # `factory.build_orchestrator_from_env` (OrchestratorBundle is the
        # shared WIRING function, not an instance), so the caller pins the
        # channel HERE; it doesn't need to be re-passed on every `handle()`
        # call. It flows into `self._model.answer(..., channel=self._channel)`
        # (see the three call sites below) -- the FORMATTING instruction
        # changes accordingly
        # (answering_model.py::_FORMATTING_WHATSAPP/_FORMATTING_WEB).
        self._channel = channel
        # Pre-info bubbles (see preamble.py). `None` = off, behavior exactly as
        # before -- same optionality pattern as `reconciler`/`session_state`.
        self._preamble = preamble

    async def handle(
        self, session_id: str, query: str, on_trace: TraceFn | None = None
    ) -> str:
        """The duration of every L0-L4 stage is published as a ``timing``
        trace event -- `{"stage": ..., "seconds": ...}`, so how long each stage
        took is visible. This concretely shows that the reconciler adds a full
        LLM round-trip even on non-SQL turns (e.g. even a one-word message like
        "selam" makes 3 sequential model calls: reconcile+intent+answering).

        Pinned session: if `prev_state.pinned` is filled, the full L0-L4 chain
        (reconcile + classify) is SKIPPED and the flow drops directly to the
        pinned flow's cheap `check_pin` (see `_check_pin`/`_answer_pinned`). If
        the check says "pin broken" (or the flow doesn't implement
        `PinnableFlow` at all), the escape hatch kicks in: the pin is reset and
        the whole turn goes through the normal full pipeline (below,
        `_handle_full_pipeline`) -- the user NEVER gets stuck, worst case the
        session returns to the cost of an unpinned one."""
        # Read once regardless of reconciler/pin -- both L0 (when a reconciler
        # exists) and `followups` need the PREVIOUS turn's `last_results` at
        # turn START; without a reconciler the state used to never be read
        # beforehand (only a fresh `.get()` after L4), which meant followups
        # never reached the answering model.
        prev_state = self._session_state.get(session_id)

        # The preamble is CONSTRUCTED here but NOT STARTED. `start()` is
        # deliberately called separately in each of the three branches, only
        # after which flow will be run is HONESTLY known -- saying "comparing"
        # before the pin check (avg 0.5s) or the handoff approval passes would
        # be a promise that doesn't materialize if the pin breaks.
        # `ticker.trace` is the `on_trace` passed down (to the flows): the raw
        # callback itself when passive, a wrapper that listens to routing
        # events and forwards them verbatim when active (so the watchdog knows
        # which flow we're in).
        ticker = PreambleTicker(query, on_trace, self._preamble)

        if prev_state.suggested_next_flow is not None:
            # One-shot handoff offer -- no matter what the user says (accept or
            # reject), the offer is reset AFTER this turn is processed (see the
            # SessionState.suggested_next_flow docstring). Unlike `PinnedFlow`,
            # there is NO persistent check_pin/pin_request lifecycle.
            target_flow_name = prev_state.suggested_next_flow
            candidates = list(prev_state.suggested_next_candidates)
            prev_state.suggested_next_flow = None
            prev_state.suggested_next_candidates = []
            if await self._is_affirmative(query, candidates):
                ticker.start(known_flow=target_flow_name)
                try:
                    return await self._answer_handoff(
                        session_id, query, prev_state, target_flow_name, candidates, ticker.trace
                    )
                finally:
                    ticker.stop()
            # Declined/not understood -- the offer was already reset; this turn
            # continues through the normal path (pin if any, else full
            # pipeline).

        if prev_state.pinned is not None:
            pin = prev_state.pinned
            flow = self._router.flow_by_name(pin.flow)
            t0 = time.perf_counter()
            still_pinned = await self._check_pin(flow, query, pin)
            emit_timing(on_trace, "pin_check", t0)
            if on_trace:
                on_trace("pin_check", "held" if still_pinned else "broken")
            if still_pinned:
                ticker.start(known_flow=pin.flow)
                try:
                    return await self._answer_pinned(
                        session_id, query, prev_state, flow, ticker.trace
                    )
                finally:
                    ticker.stop()
            # Escape hatch: pin broken -- NEVER stick, reset the pin and drop
            # to the full pipeline (this state is already a copy from
            # `store.get()`, mutating here is safe -- the stored version is
            # overwritten at the end of this turn by
            # `_handle_full_pipeline` anyway).
            if on_trace:
                on_trace("pin_fallback", pin.flow)
            prev_state.pinned = None

        # `known_flow=None`: on the full pipeline the flow is only known AFTER
        # the intent (~5th second) -- the early bubble is therefore chosen from
        # a generic/product-code pool, while the watchdog learns the real flow
        # from the router event `ticker.trace` listens to.
        ticker.start(known_flow=None)
        try:
            return await self._handle_full_pipeline(
                session_id, query, prev_state, ticker.trace
            )
        finally:
            ticker.stop()

    async def _check_pin(self, flow: Flow, query: str, pin: PinnedFlow) -> bool:
        """"Continue or broken" check -- the core of the pinned path. If the flow
        doesn't implement `PinnableFlow` OR the check itself blows up (network,
        unexpected data, ...) we stay on the SAFE side: the pin counts as "broken"
        and the caller drops to the full pipeline. Same pattern as the Reconciler's
        "any error NEVER drops the turn" principle, applied to a different failure
        mode (here: the user never gets stuck)."""
        if not isinstance(flow, PinnableFlow):
            return False
        try:
            return bool(await flow.check_pin(query, pin))
        except Exception:  # noqa: BLE001 -- if the pin check blows up, safe side: pin counts as "broken", caller drops to the full pipeline (described in the docstring)
            return False

    async def _is_affirmative(self, query: str, candidates: list[str]) -> bool:
        """Did the user accept the previous turn's handoff offer ("want to
        compare these?")? If `continuation_checker` is injected, the decision
        delegates to it; network/parse errors are NOT swallowed here -- the
        caller (`handle`) already has the "if unsure, treat as declined" safe
        default (the try/except below) -- a separate error-swallowing mechanism
        is NOT REPEATED here."""
        if self._continuation_checker is None:
            normalized = query.strip().lower()
            return normalized in _AFFIRMATIVE_WORDS
        context = (
            "The user was asked last turn whether they'd like to compare "
            f"these products: {', '.join(candidates)}. The user is expected "
            "to accept or decline this offer."
        )
        try:
            return await self._continuation_checker.check(query, context)
        except Exception:  # noqa: BLE001 -- unclear/error -> treat as declined, fall back to the normal pipeline
            return False

    async def _document_hint(self, context: FlowContext) -> str:
        """Sibling of `followups_prompt` that looks at THIS turn's
        `context.results` instead of the PREVIOUS turn's -- triggers only while
        products are being discussed (the `__init__` note above), needs no
        second intent/flow distinction: `product_code` metadata exists only on
        db_query/SQL rows (`retrieval.modules.db_query.mapping`), not on
        top_n/chunk rows, and not on `doc_download`'s own rows either (it uses
        the `model` key, see flows/doc_download.py::_to_result) -- so this
        check naturally produces a candidate only on single-product SQL turns
        and silently skips `doc_download` turns (which already have their own
        document follow-ups).

        Returns empty WITHOUT GUESSING if multiple/no product codes are found
        (ambiguous -- e.g. aggregation's multi-product result). Also returns
        empty if the single found product has no real active file in the
        `document` table -- never invents, only suggests what actually
        exists."""
        if self._document_lookup is None:
            return ""
        codes = {code for r in context.results if (code := r.metadata.get("product_code"))}
        if len(codes) != 1:
            return ""
        try:
            rows = await self._document_lookup.find(list(codes))
        except Exception:  # noqa: BLE001 -- the hint is optional; it never drops the answer
            return ""
        if not rows:
            return ""
        row = rows[0]
        return (
            "A file is also available for this product but wasn't part of "
            "the evidence above -- suggest it briefly only if it fits the "
            "flow of the conversation and the user seems interested, DON'T "
            f"FORCE it: {row.file_name} ({row.doc_type})"
        )

    async def _answer_handoff(
        self,
        session_id: str,
        query: str,
        prev_state: SessionState,
        target_flow_name: str,
        candidates: list[str],
        on_trace: TraceFn | None,
    ) -> str:
        """The user accepted the previous turn's handoff offer -- runs the
        target flow (today: `ComparisonFlow`) with the candidates PRE-FILLED
        in the query, SKIPPING reconcile/classify (the same "cheap path"
        philosophy as `_answer_pinned` on the pinned path, but a one-shot
        trigger INSTEAD of a persistent pin). `strategy_key` is
        `target_flow_name` ITSELF -- the only handoff target today,
        `ComparisonFlow`, keeps `flow_name`/`intent_key` equal to `"comparison"`
        (see flows/comparison.py); this ASSUMPTION is the same one
        `PinnedFlow.flow`/`PinnedFlow.intent` already make (see the
        session_state.py::PinnedFlow docstring, known limitation)."""
        total_t0 = time.perf_counter()
        history = self._memory.get(session_id)
        flow = self._router.flow_by_name(target_flow_name)
        synthesized_query = " ".join([*candidates, query]).strip() or query

        t0 = time.perf_counter()
        context = await flow.run(synthesized_query, on_trace)
        emit_timing(on_trace, "retrieval", t0)
        strategy_key = target_flow_name
        strategy_prompt = load_strategy(strategy_key)
        # On this path the reconciler never runs (no rec.transition), but a
        # handoff turn is structurally always a continuation of the earlier
        # offer -- a fixed, honest note (see session_state._TRANSITION_NOTES).
        session_goal = prev_state.as_goal_prompt(transition="pinned_continuation")
        if on_trace:
            on_trace("handoff", target_flow_name)
            on_trace("strategy", strategy_key)
            if session_goal:
                on_trace("goal", session_goal)
            on_trace("answering", "cevap üretiliyor...")

        followups = prev_state.followups_prompt()
        t0 = time.perf_counter()
        reply = await self._model.answer(
            query, context, strategy_prompt, history,
            session_goal=session_goal, hedge=False, followups=followups,
            channel=self._channel,
        )
        emit_timing(on_trace, "answering", t0)
        emit_timing(on_trace, "total", total_t0)

        new_state = prev_state.model_copy(deep=True)
        new_state.turn += 1
        new_state.last_results = context.results
        new_state.suggested_next_flow = context.suggested_next_flow
        new_state.suggested_next_candidates = list(context.suggested_next_candidates)
        if isinstance(flow, SelfPinningFlow):
            new_state.pinned = await flow.pin_request(synthesized_query, context)
        self._session_state.set(session_id, new_state)
        self._memory.append(session_id, "user", query)
        self._memory.append(session_id, "assistant", reply)
        return reply

    async def _answer_pinned(
        self,
        session_id: str,
        query: str,
        prev_state: SessionState,
        flow: Flow,
        on_trace: TraceFn | None,
    ) -> str:
        """The pinned cheap path: NO reconcile/classify, straight to the pinned
        flow's `run` + strategy + answering model. No `resolved_query` is
        produced (the reconciler isn't called) -- a deliberate limit of this
        design stage; the raw query is used directly (a flow entering a pin
        interprets the query itself inside its own `check_pin` if needed).
        `session_goal` is NOT hedged (`hedge=False`) -- the low-confidence
        signal comes only from the reconciler's own confidence, and that step
        never ran on the pinned path."""
        total_t0 = time.perf_counter()
        history = self._memory.get(session_id)
        pin = prev_state.pinned
        assert pin is not None  # only `handle` calls this while it's filled

        t0 = time.perf_counter()
        # If the flow flagged that it must MERGE this raw query with the state
        # collected in PREVIOUS turns (`PinAwareRunFlow`, e.g.
        # `RecommendationFlow`'s clarification dialog), `run_pinned` is called;
        # flows that don't implement it (e.g. `ComparisonFlow`) keep using
        # `run()` as before.
        if isinstance(flow, PinAwareRunFlow):
            context = await flow.run_pinned(query, pin, on_trace)
        else:
            context = await flow.run(query, on_trace)
        emit_timing(on_trace, "retrieval", t0)
        strategy_key = pin.intent
        strategy_prompt = load_strategy(strategy_key)
        # On the pinned path the reconciler never runs (no rec.transition), but
        # a pinned turn is structurally always a continuation of the earlier
        # clarification -- a fixed, honest note (see
        # session_state._TRANSITION_NOTES).
        session_goal = prev_state.as_goal_prompt(transition="pinned_continuation")
        if on_trace:
            on_trace("strategy", strategy_key)
            if session_goal:
                on_trace("goal", session_goal)
            on_trace("answering", "cevap üretiliyor...")

        followups = prev_state.followups_prompt()
        t0 = time.perf_counter()
        reply = await self._model.answer(
            query, context, strategy_prompt, history,
            session_goal=session_goal, hedge=False, followups=followups,
            channel=self._channel,
        )
        emit_timing(on_trace, "answering", t0)
        emit_timing(on_trace, "total", total_t0)

        # The pin is PRESERVED by default (this flow is still pinned to
        # itself) -- only turn/last_results advance; the last_results rule is
        # the same here: independent of the reconciler path, EVERY flow's
        # result carries into the next turn.
        new_state = prev_state.model_copy(deep=True)
        new_state.turn += 1
        new_state.last_results = context.results
        # The handoff offer (if any) is also carried into the next turn -- same
        # pattern as `last_results`, independent of the pinned path.
        new_state.suggested_next_flow = context.suggested_next_flow
        new_state.suggested_next_candidates = list(context.suggested_next_candidates)
        # If the flow manages its own pin lifecycle (today only
        # `ComparisonFlow`, e.g. it clears the pin ITSELF once clarification
        # resolves) `pin_request` overwrites this slot -- for flows that don't
        # implement it, the pin stays copied verbatim, zero behavior change
        # (see flows/base.py::SelfPinningFlow docstring).
        if isinstance(flow, SelfPinningFlow):
            new_state.pinned = await flow.pin_request(query, context)
        self._session_state.set(session_id, new_state)
        self._memory.append(session_id, "user", query)
        self._memory.append(session_id, "assistant", reply)
        return reply

    async def _handle_full_pipeline(
        self,
        session_id: str,
        query: str,
        prev_state: SessionState,
        on_trace: TraceFn | None,
    ) -> str:
        """The old `handle` body -- the full L0-L4 chain (reconcile + classify
        + route + answer). `prev_state.pinned` is always `None` here (`handle`
        calls this directly when there was no pin, or resets and calls it when
        the pin broke) -- this method does NOT KNOW the pin concept; existing
        behavior is preserved exactly (zero change for unpinned sessions)."""
        total_t0 = time.perf_counter()
        history = self._memory.get(session_id)

        # L0+L1: update state + resolve the elliptical query (skip if no
        # reconciler).
        resolved = query
        session_goal = ""
        hedge = False
        rec = None
        if self._reconciler is not None:
            state = prev_state
            t0 = time.perf_counter()
            rec = await self._reconciler.reconcile(state, history, query)
            emit_timing(on_trace, "reconcile", t0)
            resolved = rec.resolved_query
            # The transition (continue/evolve/switch/digress) now also reaches
            # the answering model -- a framing note on switch/digress.
            session_goal = rec.new_state.as_goal_prompt(transition=rec.transition)
            hedge = rec.confidence is not None and rec.confidence < self._hedge_threshold
            if on_trace:
                on_trace("reconcile", rec.transition)
                if resolved != query:
                    on_trace("resolved_query", resolved)
                if rec.raw_text is not None:
                    # The reconciler's model returned something but it couldn't
                    # be parsed -- instead of silently falling back to
                    # "continue", show what it returned (makes the
                    # intermediate model's output visible).
                    on_trace("reconcile_raw", rec.raw_text)

        # L2: intent. If fusion is on (the reconciler also produced the intent)
        # NO separate classify call is made -- one full LLM round-trip is cut.
        # If rec.intent is missing/invalid (fusion off, no reconciler, or the
        # model returned an invalid label that `_apply` dropped) it falls back
        # to the old path, the separate classifier (graceful fallback).
        intent_label: IntentLabel | None = None
        intent_confidence = rec.confidence if rec is not None else None
        if rec is not None and rec.intent:
            try:
                intent_label = IntentLabel(rec.intent)
            except ValueError:
                intent_label = None  # unexpected label -> classifier fallback
        if intent_label is not None:
            if on_trace:
                on_trace(
                    "intent",
                    f"{intent_label.value} (füzyon)"
                    + (f" (confidence={intent_confidence:.2f})" if intent_confidence is not None else ""),
                )
        else:
            t0 = time.perf_counter()
            intent: IntentResult = await self._classifier.classify(resolved)
            emit_timing(on_trace, "intent", t0)
            intent_label = intent.label
            if on_trace:
                on_trace(
                    "intent",
                    f"{intent.label.value}"
                    + (f" (confidence={intent.confidence:.2f})" if intent.confidence is not None else ""),
                )
                if intent.raw_text is not None:
                    # The model returned something outside the label set and we
                    # silently fell to out_of_scope -- show the raw text.
                    on_trace("intent_raw", intent.raw_text)

        # L3: deterministic routing + retrieval (with the resolved query).
        # This step's own internals (SQL: linking_done/generation_done) already
        # carry their latency_ms (retrieval 0.5.0) -- only the flow's TOTAL
        # duration (the whole retrieval) is measured here.
        flow = self._router.flow_for(intent_label, on_trace)
        t0 = time.perf_counter()
        context = await flow.run(resolved, on_trace)
        emit_timing(on_trace, "retrieval", t0)
        strategy_key = intent_label.value
        strategy_prompt = load_strategy(strategy_key)
        if on_trace:
            on_trace("strategy", strategy_key)
            if session_goal:
                on_trace("goal", session_goal)
            on_trace("answering", "cevap üretiliyor...")

        # L4: the answer framed with the ongoing goal + a follow-up suggestion.
        # `followups` is PRIMARYLY derived NOT from this turn's
        # `context.results` but from `prev_state.last_results` (the PREVIOUS
        # turn) -- this turn's raw results already go to the model in full text
        # via `_render_context`; `followups` is a separate signal: "features
        # that came last turn but were never asked about" (see
        # `SessionState.followups_prompt`). Only when that's EMPTY,
        # `_document_hint` is tried from THIS turn's results below (see there).
        followups = prev_state.followups_prompt()
        # If the previous turn already produced a suggestion (spec/document
        # follow-up), nothing is ADDED ON TOP -- same principle as
        # `followups_prompt`'s "one natural suggestion, not a fixed menu feel
        # for the model" (see its docstring in session_state.py). Only if it's
        # empty, a candidate from THIS turn's own results is tried
        # (`_document_hint`).
        if not followups:
            followups = await self._document_hint(context)
        t0 = time.perf_counter()
        reply = await self._model.answer(
            resolved, context, strategy_prompt, history,
            session_goal=session_goal, hedge=hedge, followups=followups,
            channel=self._channel,
        )
        emit_timing(on_trace, "answering", t0)
        emit_timing(on_trace, "total", total_t0)

        # L0 write (for the next turn) + keeping the real (raw) messages in
        # memory. `last_results` is a SECOND write path INDEPENDENT of the
        # Reconciler -- even without a Reconciler, every flow's result carries
        # into the next turn (not flow-specific either).
        state_for_next_turn = rec.new_state if rec is not None else prev_state
        state_for_next_turn.last_results = context.results
        # The handoff offer (if any) is carried on the full-pipeline path too.
        state_for_next_turn.suggested_next_flow = context.suggested_next_flow
        state_for_next_turn.suggested_next_candidates = list(context.suggested_next_candidates)
        # The same optional lifecycle hook applies on the full-pipeline path --
        # `state_for_next_turn.pinned` is already `None` up to this point (the
        # reconciler always produces a fresh pin-less `SessionState`, see
        # reconciler.py; and if there was no pin, `prev_state.pinned` is
        # already `None`), so the unconditional assignment is safe here.
        if isinstance(flow, SelfPinningFlow):
            state_for_next_turn.pinned = await flow.pin_request(resolved, context)
        self._session_state.set(session_id, state_for_next_turn)
        self._memory.append(session_id, "user", query)
        self._memory.append(session_id, "assistant", reply)
        return reply


__all__ = ["AnsweringModel", "Orchestrator", "load_strategy"]
