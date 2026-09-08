"""recommendation flow: clarifies usage/budget-style information, then runs
`SqlTopNFlow`'s existing single-sided core (`run_pass`) with a SINGLE query.

## Clarification dialog, built on the pinned-flow mechanism

When the user says "recommend me something", the flow -- whichever product
family/constraint information is missing -- asks a clarifying question AND
pins the session to itself (the `PinnedFlow`/`SelfPinningFlow` pair, the same
pattern as `ComparisonFlow`). The next turn is the user's answer -- that
answer ALONE (e.g. "$500") is meaningless and gains sense only merged with
information collected in PREVIOUS turns. `Flow.run(query, on_trace)` can't
cover this by itself (it only sees this turn's raw query) -- so an optional
capability was added: `PinAwareRunFlow.run_pinned(query, pinned, on_trace)`
(see flows/base.py). `Orchestrator._answer_pinned` calls it INSTEAD of
`run()` when the flow implements it; flows that don't (e.g.
`ComparisonFlow`) see zero behavior change. The collected answers + which
question was asked are carried inside `PinnedFlow.expected` -- the
"already-given information is NEVER ASKED AGAIN" requirement is met exactly
by this field.

## The question pool is now derived DYNAMICALLY from the schema, not hardcoded

**The old form** (the `_ATTRIBUTE_SLOTS` constant 2-attribute tuple, which
carried a key like `number_of_channels` that did NOT even exist in the
dictionary -- drift found by audits) could drift away from `spec_keys.yaml`
and was family-blind (the SAME 2 questions in every family). **The new
design** combines two code-level (LLM-free) pieces with one LLM decision:

  1. `slot_candidates.load_attribute_glossary` -- reads the `question` text of
     ALL keys from the `attribute_glossary` field of `facts/db/schema.yaml`
     (`question` is no longer a DB column; `facts` embeds it structured into
     schema.yaml precisely for this kind of consumption) -- ZERO
     invention/drift risk, the dictionary itself is the source.
  2. `slot_candidates.discover_informative_slots` -- finds, with a
     deterministic SQL query, the keys that REALLY carry >=2 distinct values
     in `spec_value` for the given family (i.e. worth asking -- not every
     product gives the same answer); no need to have an LLM WRITE this
     fixed-shape aggregation.
  3. `slot_selection.SlotSelector` -- a small LLM decision on WHICH of those
     candidates is worth asking in this conversation context (the exact same
     Protocol+graceful-degrade pattern as `continuation.py`).

If none of `db_path`/`schema_yaml_path`/`slot_selector` is given (or
`slot_selector` returns `None`/errors/no candidates are found), the flow falls
back to `_FALLBACK_ATTRIBUTE_SLOTS` (a small, static, emergency pool) -- zero
behavior break.

## Once clarification ends, the EXISTING `sql_topn` pattern is used as-is

No new retrieval mechanism was INVENTED -- the collected answers are merged
into a single natural-language query (`_synthesize_query`) handed verbatim to
`SqlTopNFlow.run_pass` (the same SQL-filter + top_n enrichment core SHARED
with `ComparisonFlow`/`AggregationFlow`).

## Multi-product handoff (simple, ONE-SHOT fields)

If the SQL result contains 2+ distinct product models
(`metadata["product_code"]`), the flow fills
`FlowContext.suggested_next_flow`/`suggested_next_candidates` (see
flows/base.py) -- the `Orchestrator` copies it into
`SessionState.suggested_next_flow` and, if the user says "yes" on the NEXT
turn, routes straight to `ComparisonFlow` (candidates PRE-FILLED; see
orchestrator.py::`_answer_handoff`). This is NOT a persistent "flow-to-flow
handoff" MECHANISM of the kind `PinnedFlow` generalizes -- deliberately kept
narrow/one-shot (an explicit requirement).

## NO ANSWER-COUNT THRESHOLD: if any spec is in hand, the filter runs (REVISED)

**The old form (`min_answered`, removed):** `_advance` compared the NUMBER of
collected answers to a threshold (default 2); below it, SQL was never reached.
It produced two distinct failures:

  1. Even if the user gave real constraints on the first turn (e.g. a channel
     count + a voltage range), the FILTER DIDN'T START because the counter
     hadn't reached 2 -- another question was asked while the information in
     hand went unused.
  2. An answer to the `family` question that didn't fit the surface-form
     dictionary (`_FAMILY_SURFACE_FORMS`) was DISCARDED ENTIRELY
     (`parse_recommendation_answer` returned `None`) -- the concrete
     constraints the user gave in that message vanished, the counter never
     advanced, and the flow kept asking questions.

**The new form ("a silly threshold; no such threshold should exist -- if the
specs in hand allow filtering, filtering must start and the user should be
steered by what remains"):** state is now managed by the CANDIDATE SET IN
HAND, not by an answer count.

  - Whenever `collected` is non-empty, `run_pass` (SQL + top_n) runs every
    turn -- filtering with whatever is in hand. Only when it's completely
    empty (first turn, no constraints at all) does just `top_n_pass` run,
    because there's nothing to filter.
  - Clarification ends when the candidate set narrows to a PRESENTABLE size
    (`few_max`, `config/default.toml [result_shape]` -- no new threshold
    invented, reusing the value `result_shape` already uses) or when the
    question pool is exhausted. `remaining == 0` also ends it: if the
    constraints match no product at all, one more question can only narrow
    the set further; the model should suggest relaxing a constraint.
  - The next question is chosen against WHAT REMAINS:
    `discover_informative_slots` is now called with `product_codes` (see
    slot_candidates.py) -- the key asked discriminates the ACTUAL remaining
    candidates, not the whole family. Side benefit: dynamic discovery works
    even when `family` was never resolved (a phrasing outside the
    dictionary).
  - A non-matching but meaningful answer to the `family` question is no
    longer DISCARDED -- it's kept as a free-text constraint under
    `_FREE_TEXT_SLOT_KEY` and carried into the SQL query by
    `_synthesize_query`. A non-matching family is still never INVENTED
    (`collected["family"]` is not written); only the user's own words are
    preserved.

The SQL/top_n priority rule of `strategies/recommendation.md` (`[SQL]` wins on
conflict) is UNCHANGED."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medrag.api.continuation import ContinuationChecker
from medrag.api.flows.base import FlowContext, compute_result_shape
from medrag.api.flows.sql_topn import SqlTopNFlow
from medrag.api.pin_relevance import looks_relevant_to_pin
from medrag.api.session_state import PinnedFlow
from medrag.api.slot_candidates import (
    GlossaryEntry,
    discover_informative_slots,
    load_attribute_glossary,
)
from medrag.api.slot_selection import SlotSelector
from medrag.api.trace import TraceFn

_FAMILY_SLOT_KEY = "family"

# The key under which an answer to the `family` question that fits NO surface
# form but is meaningful (not a skip word) is stored (see the threshold
# revision section of the module docstring) -- NOT a real
# `attribute_glossary` key, so it never enters `asked` and is never looked up
# as a key in slot discovery; only `_synthesize_query` carries it as free
# text.
_FREE_TEXT_SLOT_KEY = "_notes"

# `product.family` (facts/db/schema.yaml) -- 12 real values + surface forms
# (only a field that GENUINELY exists in the schema). Same pattern as
# `ComparisonFlow`'s `_ANGLE_KEYWORDS`: a small, hand-picked, extensible
# dictionary. The `family` question was not made dynamic -- it isn't an
# `attribute` key but `product.family` itself, and it doesn't live in the
# glossary as an "askable question". Matching data: keep values verbatim.
_FAMILY_SURFACE_FORMS: dict[str, tuple[str, ...]] = {
    "PXI EXPRESS SYSTEMS": ("pxi express", "pxie", "pxi"),
    "MIL-STD-1553 BUS COUPLERS": ("1553", "mil-std-1553", "mil std 1553", "bus coupler"),
    "SIGNAL CONDITIONING & SIMULATION SYSTEMS": (
        "sinyal şartlandırma", "sinyal sartlandirma", "signal conditioning", "simulation",
    ),
    "TEST SYSTEM COMPONENTS": ("test sistemi", "test system"),
    "EMBEDDED BOARDS & MODULES": ("gömülü", "gomulu", "embedded board", "embedded module"),
    "ETHERNET CONTROLLED INSTRUMENTS (LXI)": (
        "lxi", "ethernet controlled", "ethernet kontrollü", "ethernet kontrollu",
    ),
    "AVIONICS INTERFACES": ("aviyonik", "avionics"),
    "DAQ SYSTEMS": ("daq", "veri toplama"),
    "SOFTWARE PRODUCTS": ("yazılım", "yazilim", "software"),
    "PCI EXPRESS SYSTEMS": ("pci express", "pcie"),
    "NAVIGATION SYSTEM PRODUCTS": ("navigasyon", "navigation", "gps", "gnss"),
    "FMC MODULES": ("fmc",),
}

_FAMILY_QUESTION = (
    "Which product family/use case are you interested in (e.g. data "
    "acquisition (DAQ), PXI Express, signal conditioning, avionics "
    "interfaces, embedded boards, navigation...)?"
)


@dataclass(frozen=True)
class ClarifyingSlot:
    key: str
    question: str


# EMERGENCY POOL (engaged only when dynamic discovery is entirely
# unconfigured/failed) -- `channel_count` is the corrected form of the
# `number_of_channels` drift found in audits; per the naming rule ("no
# number_of_/num_ prefix") this is the real key. It is NOT in the `core`
# block (it lives in analog_io/digital_io/battery_simulator, family-specific)
# -- the whole reason this list exists is "a family-blind approximation", and
# while the dynamic path is active this limitation stops mattering.
_FALLBACK_ATTRIBUTE_SLOTS: tuple[ClarifyingSlot, ...] = (
    ClarifyingSlot(
        "operating_temperature",
        "What operating environment temperature, or any special "
        "environmental condition, does the product need to handle?",
    ),
    ClarifyingSlot(
        "channel_count",
        "Roughly how many channels/inputs-outputs do you need?",
    ),
    ClarifyingSlot(
        "power_consumption",
        "Do you have any constraint on power consumption or supply voltage?",
    ),
)

# Short answers showing the user deliberately wants to SKIP a question -- the
# slot counts as "asked" (NEVER ASKED AGAIN) but is NOT STORED as a constraint
# (same pattern as `ComparisonFlow`'s `_CANCEL_WORDS`). Matching data: keep
# values verbatim.
_SKIP_WORDS = {
    "farketmez", "fark etmez", "bilmiyorum", "emin değilim", "emin degilim",
    "yok", "önemli değil", "onemli degil", "geç", "gec",
}

# The SAME small/deterministic set as `ComparisonFlow`'s `_CANCEL_WORDS` --
# kept as a separate constant (importing from comparison.py would create a
# dependency where the two flows must stay INDEPENDENT). Now only a FALLBACK
# engaged when `_continuation_checker` is never injected (dev/test, endpoint
# unconfigured) -- the primary decision mechanism is
# `continuation.ContinuationChecker` (see the "REVISED" note in the
# flows/base.py::PinnableFlow docstring). Matching data: keep values verbatim.
_CANCEL_WORDS = {
    "iptal", "boşver", "bosver", "geç", "gec", "vazgeç", "vazgec",
    "unut", "hayır", "hayir", "boş ver", "bos ver",
}

# Every attribute slot in the dynamic/static pools inherently expects NUMERIC
# answers -- the fallback relevance gate is a heuristic built on top of the
# minimal/existing question text (no new dictionary was invented): if the
# message contains a digit, it counts as "an attempt to answer this slot".
# This does NOT apply to the `family` slot -- that one is already handled
# separately by `parse_family_query`'s own surface-form matching.
_NUMERIC_ANSWER_RE = re.compile(r"\d")


def _question_for(slot_key: str, glossary: dict[str, GlossaryEntry]) -> str:
    if slot_key == _FAMILY_SLOT_KEY:
        return _FAMILY_QUESTION
    entry = glossary.get(slot_key)
    if entry is not None:
        return entry.question
    for slot in _FALLBACK_ATTRIBUTE_SLOTS:
        if slot.key == slot_key:
            return slot.question
    raise KeyError(slot_key)  # a key not in any pool -- a programming error


def _match_family(text: str) -> str | None:
    family, _residual, _near_miss = parse_family_query(text)
    return family


def _is_skip(text: str) -> bool:
    normalized = text.strip().lower()
    return not normalized or normalized in _SKIP_WORDS


def parse_family_query(query: str) -> tuple[str | None, str, None]:
    """PURE function (for testability) -- the core of the surface-form match
    also used by `_match_family`, extracting the CONSUMED part from the query
    and returning the residual. Returns: (found family or `None`, unconsumed
    residual text, near-miss note).

    The third element (`near_miss`) always returns `None` for this flow --
    there is no "looks-like-a-code-but-invalid" surface form defined for
    `product.family` the way `ComparisonFlow`/`DocDownloadFlow` have the
    product-code SHAPE (DE+digits) (see the dynamic-pool section of the module
    docstring: the pool is already limited to a subset that GENUINELY exists
    in the schema) -- the signature is kept as (matched, residual, near_miss)
    only for CONSISTENCY with the other two flows."""
    lowered = query.lower()
    for family, forms in _FAMILY_SURFACE_FORMS.items():
        for form in forms:
            idx = lowered.find(form)
            if idx != -1:
                residual = query[:idx] + query[idx + len(form) :]
                residual = re.sub(r"\s+", " ", residual).strip()
                return family, residual, None
    return None, query.strip(), None


def parse_recommendation_answer(slot_key: str, reply: str) -> str | None:
    """PURE function -- the interpretation core of
    `RecommendationFlow._record_answer`, side-effect free: `reply` (this turn's
    raw query) returns the value to be STORED as the answer of `slot_key` (or
    `None` -- if it's a skip word, empty, or `family` didn't match). Returning
    `None` means "no answer" -- the caller still keeps the slot in `asked`
    (never asked again) and simply doesn't write it into `collected` (a
    non-matching family is never INVENTED)."""
    if _is_skip(reply):
        return None
    if slot_key == _FAMILY_SLOT_KEY:
        family, _residual, _near_miss = parse_family_query(reply)
        return family
    text = reply.strip()
    return text or None


class RecommendationFlow:
    """`flow_name`/`intent_key` (same rationale as `ComparisonFlow`) must
    exactly equal the flow name registered in the `Router` and the
    `strategies/<intent_key>.md` key -- `PinnedFlow` doesn't validate this
    (known limitation, see session_state.py)."""

    def __init__(
        self,
        recommendation_pass: SqlTopNFlow,
        *,
        flow_name: str = "recommendation",
        intent_key: str = "recommendation",
        handoff_flow_name: str = "comparison",
        min_handoff_candidates: int = 2,
        max_handoff_candidates: int = 4,
        continuation_checker: ContinuationChecker | None = None,
        slot_selector: SlotSelector | None = None,
        db_path: str | None = None,
        schema_yaml_path: str | Path | None = None,
        few_max: int = 5,
    ) -> None:
        self._pass = recommendation_pass
        self._flow_name = flow_name
        self._intent_key = intent_key
        self._handoff_flow_name = handoff_flow_name
        self._min_handoff_candidates = min_handoff_candidates
        self._max_handoff_candidates = max_handoff_candidates
        self._continuation_checker = continuation_checker
        # Dynamic question-pool discovery -- active only when all three are
        # given; if any is missing, fall back to `_FALLBACK_ATTRIBUTE_SLOTS`
        # (graceful degrade, the same principle as `continuation_checker`).
        self._slot_selector = slot_selector
        self._db_path = db_path
        self._glossary: dict[str, GlossaryEntry] = (
            load_attribute_glossary(schema_yaml_path) if schema_yaml_path else {}
        )
        # Threshold for `FlowContext.result_shape` -- injected from `config/
        # default.toml [result_shape] few_max` (see
        # factory.py::build_flows_from_env). The SAME value is also used as the
        # END criterion for clarification ("has the candidate set narrowed to a
        # presentable size") -- deliberate: both answer "how many results can
        # be shown to the user at once", and a separate threshold would just
        # keep the same number in two places.
        self._few_max = few_max

    def _dynamic_enabled(self) -> bool:
        return self._slot_selector is not None and self._db_path is not None and bool(self._glossary)

    def _slot_selection_context(self, raw_query: str, collected: dict[str, str]) -> str:
        parts = [f'The user wants a product recommendation. Original request: "{raw_query}".']
        family = collected.get(_FAMILY_SLOT_KEY)
        if family:
            parts.append(f"Product family already identified: {family}.")
        extra = {k: v for k, v in collected.items() if k != _FAMILY_SLOT_KEY}
        if extra:
            parts.append(
                "Already collected answers: " + ", ".join(f"{k}={v}" for k, v in extra.items()) + "."
            )
        return " ".join(parts)

    def _legacy_pool_keys(self) -> list[str]:
        return [_FAMILY_SLOT_KEY, *(s.key for s in _FALLBACK_ATTRIBUTE_SLOTS)]

    async def _next_slot_key(
        self, raw_query: str, collected: dict[str, str], asked: list[str], on_trace: TraceFn | None,
        *, product_codes: Sequence[str] = (),
    ) -> str | None:
        """The key of the next clarifying question -- `family` is ALWAYS asked
        first; after that dynamic discovery (if active) is tried, and if it
        fails/returns empty/is off, the flow drops to the static emergency
        pool. Network/DB errors are SWALLOWED here (unlike continuation.py) --
        failing to CHOOSE a clarifying question must not BREAK
        `RecommendationFlow`'s core flow; it should only fall back to a coarser
        question.

        `product_codes` are the candidates the filter so far has LEFT in hand
        -- if given, discovery looks only at keys that discriminate those
        candidates instead of the whole family (see slot_candidates.py). This
        also makes dynamic discovery possible while `family` is still
        unresolved."""
        if _FAMILY_SLOT_KEY not in asked:
            return _FAMILY_SLOT_KEY

        family = collected.get(_FAMILY_SLOT_KEY)
        if self._dynamic_enabled() and (product_codes or family):
            try:
                candidates = discover_informative_slots(
                    self._db_path, family=family, glossary=self._glossary,
                    exclude_keys=set(asked), limit=8,
                    product_codes=list(product_codes) or None,
                )
                if candidates:
                    context = self._slot_selection_context(raw_query, collected)
                    chosen = await self._slot_selector.choose(context, candidates)
                    if on_trace:
                        on_trace("recommendation_slot_candidates", [c.key for c in candidates])
                        on_trace("recommendation_slot_chosen", chosen)
                    if chosen is not None:
                        return chosen
            except Exception as exc:  # noqa: BLE001 -- bkz. docstring: kirlenmeden dus
                if on_trace:
                    on_trace("recommendation_slot_selection_failed", str(exc)[:200])

        return next((k for k in self._legacy_pool_keys() if k not in asked), None)

    def _record_answer(self, slot_key: str, reply: str, collected: dict[str, str]) -> None:
        """`reply` (this turn's raw query) is interpreted as the answer to
        `slot_key` -- if it is a skip word (or empty) nothing is RECORDED
        (the slot still stays in `asked` and is never asked again). The
        interpretation itself is delegated to `parse_recommendation_answer`.

        An answer to the `family` question that matches NO surface form but is
        meaningful (not a skip word) is no longer DISCARDED -- a mismatched
        family is still never INVENTED (`collected["family"]` is not written),
        but the user's text is preserved as a free-text constraint under
        `_FREE_TEXT_SLOT_KEY` and carried into the SQL filter via
        `_synthesize_query`. Previously the concrete constraints the user gave
        in that message (channel count, voltage, ...) silently vanished."""
        value = parse_recommendation_answer(slot_key, reply)
        if value is not None:
            collected[slot_key] = value
            return
        if slot_key == _FAMILY_SLOT_KEY and not _is_skip(reply):
            note = reply.strip()
            existing = collected.get(_FREE_TEXT_SLOT_KEY)
            collected[_FREE_TEXT_SLOT_KEY] = f"{existing} {note}".strip() if existing else note

    def _synthesize_query(self, raw_query: str, collected: dict[str, str]) -> str:
        """Merges the collected answers into ONE natural-language query --
        `run_pass` (and the SQL/top_n chain inside it) already has its own
        natural-language understanding, so NO extra parsing happens here.
        ALL keys of `collected` are carried over (no longer limited to a
        static tuple -- they can be any dynamically selected key).
        `_FREE_TEXT_SLOT_KEY` (see `_record_answer`) is appended as raw text
        rather than `key: value` because it is NOT a schema key."""
        parts = [raw_query]
        family = collected.get(_FAMILY_SLOT_KEY)
        if family:
            parts.append(family)
        for key, value in collected.items():
            if key == _FAMILY_SLOT_KEY:
                continue
            parts.append(value if key == _FREE_TEXT_SLOT_KEY else f"{key}: {value}")
        return " | ".join(parts)

    def _remaining_codes(self, results) -> list[str]:
        """The DISTINCT product codes appearing in the results (first-seen
        order) -- the "candidate set left on hand". Reads the same
        `product_code` metadata field as `_detect_handoff` (see the fix note
        there); extracted into a shared helper so it is defined in one place."""
        seen: dict[str, None] = {}
        for r in results:
            code = r.metadata.get("product_code")
            if code:
                seen.setdefault(str(code), None)
        return list(seen.keys())

    def _detect_handoff(self, results) -> tuple[str | None, list[str]]:
        """Counts the distinct product codes from the SQL rows'
        `product_code` metadata -- if 2+, a handoff is offered (see module
        docstring). **Fix:** `"model"` was NEVER a real column name
        (`retrieval.modules.db_query.mapping.row_to_result` fills metadata
        with the SQL's OWN column names -- the real name is `product_code`),
        so the handoff never fired and `seen` stayed empty."""
        candidates = self._remaining_codes(results)
        if len(candidates) >= self._min_handoff_candidates:
            return self._handoff_flow_name, candidates[: self._max_handoff_candidates]
        return None, []

    async def _advance(
        self,
        raw_query: str,
        collected: dict[str, str],
        asked: list[str],
        on_trace: TraceFn | None,
        *,
        residual: str | None = None,
    ) -> FlowContext:
        # No threshold on the NUMBER of answers (see module docstring). If any
        # constraint is on hand, the SQL filter runs IMMEDIATELY; if there is
        # none (nothing to filter), only top_n runs.
        synthesized = self._synthesize_query(raw_query, collected)
        if collected:
            if on_trace:
                on_trace("recommendation_query", synthesized)
            results = await self._pass.run_pass(synthesized, on_trace)
        else:
            results = await self._pass.top_n_pass(synthesized, on_trace)

        remaining = self._remaining_codes(results)
        if on_trace and collected:
            on_trace("recommendation_remaining", len(remaining))

        # Clarification ends when the CANDIDATE SET is narrowed enough to
        # present (0 included: if no product is left, one more question can
        # only narrow it further -- the model should suggest relaxing
        # constraints). When no filter has run yet (`collected` empty), this
        # criterion does not apply.
        narrowed_enough = bool(collected) and len(remaining) <= self._few_max
        if not narrowed_enough:
            next_key = await self._next_slot_key(
                raw_query, collected, asked, on_trace, product_codes=remaining
            )
            if next_key is not None:
                asked = [*asked, next_key]
                if on_trace:
                    on_trace("recommendation_clarify", next_key)
                return FlowContext(
                    results=results,
                    scratch={
                        "collected": collected, "asked": asked,
                        "asking": next_key, "raw_query": raw_query,
                    },
                    result_shape=compute_result_shape(len(results), few_max=self._few_max),
                    residual=residual,
                )
            # The question pool is exhausted (the user skipped all of them /
            # no discriminating key left to ask) -- finish with what we have,
            # WITHOUT getting stuck; fall through.

        handoff_flow, candidates = self._detect_handoff(results)
        if on_trace and handoff_flow:
            on_trace("recommendation_handoff_offer", candidates)
        return FlowContext(
            results=results, scratch={},
            suggested_next_flow=handoff_flow, suggested_next_candidates=candidates,
            result_shape=compute_result_shape(len(results), few_max=self._few_max),
            residual=residual,
        )

    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext:
        """Unpinned (first) turn: no collected answers yet, but the raw
        query may already contain a family name (e.g. "suggest a DAQ
        system") -- captured opportunistically (information already given
        is NEVER ASKED AGAIN)."""
        collected: dict[str, str] = {}
        asked: list[str] = []
        family, residual, _near_miss = parse_family_query(query)
        if family:
            collected[_FAMILY_SLOT_KEY] = family
            asked.append(_FAMILY_SLOT_KEY)
        return await self._advance(query, collected, asked, on_trace, residual=residual or None)

    async def run_pinned(
        self, query: str, pinned: PinnedFlow, on_trace: TraceFn | None = None
    ) -> FlowContext:
        """`PinAwareRunFlow`: this turn's raw query is the ANSWER to the
        question asked in the PREVIOUS turn (`pinned.expected["asking"]`) --
        it is recorded first, then clarification continues where it left
        off."""
        expected: dict[str, Any] = pinned.expected or {}
        collected: dict[str, str] = dict(expected.get("collected", {}))
        asked: list[str] = list(expected.get("asked", []))
        asking = expected.get("asking")
        raw_query = expected.get("raw_query", query)
        if asking:
            self._record_answer(asking, query, collected)
        return await self._advance(raw_query, collected, asked, on_trace)

    async def check_pin(self, query: str, pinned: PinnedFlow) -> bool:
        """`PinnableFlow`: while pinned, a clarification answer is EXPECTED.
        If `_continuation_checker` is injected (see factory.py), the decision
        is delegated to it -- the expected question
        (`pinned.expected["asking"]`) is rendered into natural language and
        given as context; the LLM decides whether the user is still answering
        that question or has changed/cancelled the topic. If no checker is
        injected (dev/test, endpoint not configured), it falls back to
        `pin_relevance.looks_relevant_to_pin` -- a `parse_family_query` match
        counts as a signal for the `family` slot; for an attribute slot
        (temperature/channels/power, whether picked from the dynamic or the
        static pool) a numeric answer TRACE or a skip word (`_is_skip`)
        counts as a signal; if there is none (and it is not a cancel word),
        the pin DROPS."""
        if not query.strip():
            return False
        if self._continuation_checker is None:
            asking = (pinned.expected or {}).get("asking")
            if asking == _FAMILY_SLOT_KEY:
                # A numeric trace also counts as a signal -- the user may
                # answer the family question with a concrete constraint
                # (e.g. a channel count/voltage) and that is no longer
                # DISCARDED (see `_record_answer`), so dropping the pin
                # here would be wrong too.
                family, _residual, _near_miss = parse_family_query(query)
                has_signal = (
                    family is not None
                    or _is_skip(query)
                    or bool(_NUMERIC_ANSWER_RE.search(query))
                )
            elif asking is not None:
                has_signal = bool(_NUMERIC_ANSWER_RE.search(query)) or _is_skip(query)
            else:
                has_signal = True  # which slot was asked is unknown -- old behavior preserved
            return looks_relevant_to_pin(query, cancel_words=_CANCEL_WORDS, has_signal=has_signal)
        asking = (pinned.expected or {}).get("asking")
        context = (
            f"The user was asked this clarifying question: \"{_question_for(asking, self._glossary)}\" "
            "(to get a product recommendation). The user is expected to "
            "answer or give up." if asking else
            "The user is expected to provide clarifying information for a "
            "product recommendation."
        )
        return await self._continuation_checker.check(query, context)

    async def pin_request(self, query: str, context: FlowContext) -> PinnedFlow | None:
        """`SelfPinningFlow` (pattern applied to recommendation): reads what
        `run()`/`run_pinned()` left in `context.scratch` for this turn -- if
        a question is still being asked (`scratch["asking"]` non-empty) the
        pin is KEPT/UPDATED; when clarification is done (retrieval ran,
        `scratch` empty) the pin is CLEARED."""
        asking = context.scratch.get("asking")
        if not asking:
            return None
        return PinnedFlow(
            flow=self._flow_name, intent=self._intent_key,
            expected={
                "collected": context.scratch["collected"],
                "asked": context.scratch["asked"],
                "asking": asking,
                "raw_query": context.scratch["raw_query"],
            },
        )


__all__ = [
    "ClarifyingSlot",
    "RecommendationFlow",
    "parse_family_query",
    "parse_recommendation_answer",
]
