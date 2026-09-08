"""L0 — session task-state: the user's ONGOING goal (different from
turn-intent).

Turn-intent says "what is this message asking" (retrieval routing); session
task-state holds "what is the user generally trying to achieve" (answer
framing). E.g. a user comparing two products may ask for individual specs --
every turn is `product_fact` retrieval but the ONGOING goal is still
comparison; this layer carries that goal so the answer doesn't lose context.

Representation is HYBRID: a structural skeleton (task/subjects/constraints) +
a free-text "focus". The skeleton gives inspectable boundaries/dedup; focus
carries the slow evolution of the goal. Against stickiness: subject
recency-decay (entities untouched for N turns drop) + a suspend stack on
switch (no deletion).

`SessionStateStore` follows the same principle as `ConversationMemory`: RAM
ONLY, NO persistence. `last_results` is subject to the same rule -- lives only
in RAM, survives until the next turn.
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, Field

from medrag.api.retrieval.core import RetrievalResult

# Short notes carrying this turn's transition (produced by the reconciler) to
# the answering model. NO note for `continue`/`evolve`: the goal already
# flows smoothly, an extra sentence would just be noise -- only the break
# moments (switch/digress) should change the model's framing.
#
# `pinned_continuation`: on pinned/handoff paths the reconciler NEVER runs, so
# there is no real `rec.transition` -- since those paths are structurally
# always "continuation of the earlier clarification", a FIXED, honest note is
# used instead of a computed transition (without ADDING a new SessionState
# field; the cheaper option).
_TRANSITION_NOTES = {
    "switch": "NOTE: The user switched to a new goal this turn; the goal "
              "above is the NEW goal, the previous one was dropped -- do "
              "not revert to the old context.",
    "digress": "NOTE: This turn is a temporary digression from the main "
               "goal; the ongoing goal above is still preserved -- answer "
               "this turn but don't forget the goal.",
    "pinned_continuation": "NOTE: This turn continues the earlier "
                           "clarification; keep answering in that context.",
    "resume": "NOTE: The user is returning to an earlier topic that was set "
              "aside; the goal above is that earlier topic, now resumed -- "
              "pick up where it left off, don't treat it as brand new.",
}


class ConstraintDelta(BaseModel):
    """Concrete record of what changed in `constraints` THIS TURN -- derived
    from the reconciler's `dropped_constraints` field, layered on top of
    `_merge_constraints`' plain-union behavior. This turn only; NOT carried
    into `TaskSnapshot`/`suspended` (same principle as `pinned`/`last_results`
    -- the next turn's correction produces its own delta; the old one is
    meaningless)."""

    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)


def _constraint_delta_note(delta: ConstraintDelta) -> str:
    """Converts a `ConstraintDelta` into a short correction sentence the model
    can read -- it CANNOT be a fixed dict like `_TRANSITION_NOTES` because the
    content depends on this turn's actual data."""
    bits: list[str] = []
    if delta.added:
        bits.append("added " + ", ".join(f"'{c}'" for c in delta.added))
    if delta.removed:
        bits.append("dropped " + ", ".join(f"'{c}'" for c in delta.removed) + " (superseded)")
    return "Correction this turn: " + "; ".join(bits) + "."


class Subject(BaseModel):
    """A product/entity on stage + the last turn it was mentioned (for
    decay)."""

    entity: str
    last_turn: int


class PinnedFlow(BaseModel):
    """State in which the session is "pinned" to a flow. When a flow pins
    itself (currently NO flow does -- a later-stage feature), the Orchestrator
    skips the whole reconcile+classify chain next turn and goes directly to
    this flow's cheap "continue or broken" check
    (`flows/base.py::PinnableFlow.check_pin`).

    `flow`: the flow name registered in the `Router` (resolved via
    `Router.flow_by_name` -- DELIBERATELY INDEPENDENT of the `routing.intents`
    mapping, since a pin binds to a flow instance, not to an intent).
    `intent`: the `strategies/<intent>.md` key to use for the lifetime of this
    pin (since classify is skipped while pinned, the intent is CARRIED HERE).
    `expected`: a flow-SPECIFIC, free-schema "expected situation" (e.g. two
    product codes for comparison) -- the generic mechanism doesn't know or
    interpret its content; it passes it verbatim to the relevant flow's
    `check_pin`."""

    flow: str
    intent: str
    expected: dict[str, Any] = Field(default_factory=dict)


class TaskSnapshot(BaseModel):
    """Lightweight task snapshot pushed onto the suspend stack (`suspended`)
    -- doesn't carry its own `suspended` (prevents infinite nesting)."""

    task: str | None = None
    subjects: list[Subject] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    focus: str = ""


class SessionState(BaseModel):
    """The session's ongoing goal. Empty (`task is None`) = no goal detected
    yet (first turns or after a clean switch)."""

    task: str | None = None
    subjects: list[Subject] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    focus: str = ""
    suspended: list[TaskSnapshot] = Field(default_factory=list)
    turn: int = 0
    confidence: float | None = None
    pinned: PinnedFlow | None = None
    """See the `PinnedFlow` docstring. `None` = normal state, every turn goes
    through the full L0-L4 chain (default, backward compatible). Deliberately
    NOT carried into `TaskSnapshot`/`suspended` -- a switch/digress already
    makes Reconciler's `_apply` produce a fresh `SessionState` WITHOUT the
    `pinned` field (see reconciler.py), so dropping to the full pipeline
    naturally clears the pin; the `Orchestrator` additionally resets it
    explicitly in the escape hatch (see orchestrator.py, after `_check_pin`)."""
    last_results: list[RetrievalResult] = Field(default_factory=list)
    """The previous turn's retrieved data (SQL rows, top_n hits) -- an exact
    copy of `FlowContext.results`; `Orchestrator.handle` OVERWRITES it every
    turn (doesn't accumulate). Not carried into `TaskSnapshot` -- after a
    switch/digress the old goal's retrieval results are meaningless for the new
    goal (deliberately dropped in `snapshot()`, like `suspended`/`turn`)."""
    suggested_next_flow: str | None = None
    """A flow's marking, for the next turn, that after `run()` it offered
    something like "want to compare these?" (today only `RecommendationFlow`,
    when it suggests several similar products -- see flows/recommendation.py).
    This is NOT a "flow-to-flow handoff" MECHANISM of the kind `PinnedFlow`
    generalizes -- deliberately NARROW and ONE-SHOT: it only carries "if the
    user says 'yes' on the next turn, which flow with which candidates", with
    no check_pin/pin_request lifecycle (`Orchestrator.handle` reads it once
    and CLEARS it no matter what the user says -- acceptance or rejection --
    see orchestrator.py). Resolved via `Router.flow_by_name` (same principle
    as `PinnedFlow.flow` -- a flow name, not an intent label). `None` = no
    offer (default, backward compatible). Not carried into
    `TaskSnapshot`/`suspended` (same rationale as `pinned`)."""
    suggested_next_candidates: list[str] = Field(default_factory=list)
    """Product codes to pre-fill the target flow with when
    `suggested_next_flow` is set (e.g. the N products `ComparisonFlow` will
    compare) -- see `Orchestrator._answer_handoff`."""
    constraint_delta: ConstraintDelta | None = None
    """This turn's correction: filled by `_apply` when the reconciler returns
    `dropped_constraints` and a real add/removal happened; otherwise `None`
    (zero behavior change -- old, correction-less turns are unaffected).
    Computed only on `continue`/`evolve` (`switch` restarts from scratch,
    `digress` never merges). Like `last_results`/`pinned`, THIS TURN ONLY --
    not carried into `TaskSnapshot`/`suspended`."""

    def available_followups(self, limit: int = 8) -> list[str]:
        """Deterministically derive "want more specific info?" candidates from
        `last_results` -- NO model call, plain code. Grounded strictly in the real
        `key` column that `db_query` SQL rows carry in `metadata`
        (FK -> `spec_key.key`, see the `spec_value`/`spec_chunk` tables in
        `facts/db/schema.yaml`) -- never free invention, only spec keys this turn /
        the last turn actually returned. `top_n` hits have no such field in
        metadata (different schema, chunk payload) -- they are skipped silently,
        NO exception is raised.

        Only rows with `status == "present"` (or no `status` at all, e.g.
        old/simple rows) become candidates -- suggesting an `absent`/
        `not_specified` feature as "more specific info" would be misleading.
        **`conflicting` also FAILS this filter and this is INTENTIONAL** ("if it
        conflicts, it waits"): when two documents state different values for the
        same feature, that feature is NOT offered as a follow-up -- it waits for
        human approval. Don't mistake this for a bug and widen the filter to show
        `conflicting` -- the sibling filter in `flows/comparison.py` is part of the
        same decision. If `product_code` is in the metadata, the candidate is
        returned product-qualified as `"{product_code}: {key}"` (so in
        multi-product turns like comparison/aggregation it stays clear which
        product it belongs to); otherwise just `key`. Order is first-seen, deduped,
        clipped by `limit` (an unbounded list would be noise for the answering
        model).

        **Why `product_code`:** this field used to be read as `"model"` -- the real
        `spec_value` column is `product_code`
        (`retrieval.modules.db_query.mapping.row_to_result` fills metadata with the
        SQL's OWN column names; "model" was never a real column name) -- so the
        product-qualified prefix ("PN1015: weight") was never produced, always
        falling back to the bare `key`."""
        seen: dict[str, None] = {}
        for result in self.last_results:
            metadata = result.metadata
            key = metadata.get("key")
            if not key:
                continue
            status = metadata.get("status")
            if status is not None and status != "present":
                continue
            model = metadata.get("product_code")
            candidate = f"{model}: {key}" if model else str(key)
            seen.setdefault(candidate, None)
            if len(seen) >= limit:
                break
        return list(seen.keys())

    def available_document_followups(self, limit: int = 8) -> list[str]:
        """SIBLING of `available_followups` -- looks NOT at the `key`/`spec_key`
        fields in `last_results` but at the `doc_id`/`file_name`/`doc_type`/
        `requested` fields the `doc_download` flow (see flows/doc_download.py)
        adds to every row. Only rows with `requested == False` are candidates --
        these are other REAL active files of the SAME product(s) that the user did
        NOT explicitly ask for this turn (rows with `requested == True` were
        already sent/shown; we don't ask "would you like it" again). `top_n`/
        `spec_value` rows lack these fields -- they are skipped silently, NO
        exception is raised (same principle as `available_followups` skipping
        top_n).

        `file_name` is SAFE to show (`source_path` simply doesn't exist on these
        rows -- see flows/doc_download.py::_to_result, a type-level guarantee);
        the candidate is returned as `"{file_name} ({doc_type})"`. Order is
        first-seen, deduped by `doc_id` (against fan-out: the same file can appear
        on multiple `model` rows), clipped by `limit`."""
        seen: dict[str, None] = {}
        seen_doc_ids: set[str] = set()
        for result in self.last_results:
            metadata = result.metadata
            doc_id = metadata.get("doc_id")
            file_name = metadata.get("file_name")
            if not doc_id or not file_name:
                continue
            if metadata.get("requested", True):
                continue  # already requested/shown file -- not offered again
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            doc_type = metadata.get("doc_type")
            candidate = f"{file_name} ({doc_type})" if doc_type else str(file_name)
            seen.setdefault(candidate, None)
            if len(seen) >= limit:
                break
        return list(seen.keys())

    def followups_prompt(self) -> str:
        """A SINGLE follow-up suggestion handed to the answering model (L4) --
        picks only the FIRST candidate from
        `available_followups`/`available_document_followups` in a DETERMINISTIC
        priority order (spec/attribute candidates first, document candidates only
        if none). The previous behavior (TWO separate sentences, up to 8 spec + 8
        document candidates -- 16 in theory) gave the model the feel of "a fixed
        suggestion menu"; one single, natural suggestion removes that feel. Returns
        `""` when there are no candidates (then nothing is appended -- same
        pattern as `as_goal_prompt`'s empty-goal behavior)."""
        candidates = self.available_followups(limit=1)
        if candidates:
            return (
                "The data returned last turn also had this attribute the "
                "user hasn't asked about yet -- suggest it briefly only if "
                "it fits the flow of the conversation and the user wants it, "
                "DON'T FORCE it: " + candidates[0]
            )
        doc_candidates = self.available_document_followups(limit=1)
        if doc_candidates:
            return (
                "This other file is also available for the same product(s) "
                "but wasn't sent/shown last turn -- suggest it briefly only "
                "if it fits the flow of the conversation and the user wants "
                "it (e.g. \"I can also send you X if you'd like\"), DON'T "
                "FORCE it: " + doc_candidates[0]
            )
        return ""

    def snapshot(self) -> TaskSnapshot:
        return TaskSnapshot(
            task=self.task, subjects=list(self.subjects),
            constraints=list(self.constraints), focus=self.focus,
        )

    def as_goal_prompt(self, transition: str | None = None) -> str:
        """Readable framing text for the answering model (L4). Returns an empty
        string when there is no goal (then no framing is appended).

        If `transition` (this turn's continue/evolve/switch/digress transition)
        is given, a short note is appended to the framing -- previously this
        information was only used in trace + internal state-merge and NEVER
        reached the answering model; as a result the model could frame a
        `switch`/`digress` wrongly without seeing that the previous goal was
        dropped/suspended. No extra LLM call -- only the already-produced
        `rec.transition` is carried here."""
        # If both task and focus are missing there is no goal yet --
        # subjects/constraints alone don't count as a "goal" (behavior
        # preserved); a transition note would be meaningless too.
        if not self.task and not self.focus:
            return ""
        parts: list[str] = []
        if self.task:
            parts.append(f"User's ongoing goal: {self.task}")
        if self.subjects:
            parts.append("Products in scope: " + ", ".join(s.entity for s in self.subjects))
        if self.constraints:
            parts.append("Constraints gathered: " + "; ".join(self.constraints))
        if self.focus:
            parts.append(f"Current focus: {self.focus}")

        note = _TRANSITION_NOTES.get(transition or "")
        # A transition note is only meaningful when a goal framing EXISTS
        # (saying "switched to a new goal" in an empty goal is meaningless).
        # No goal and no note -> returns "".
        if note and parts:
            parts.append(note)
        if self.constraint_delta and parts:
            parts.append(_constraint_delta_note(self.constraint_delta))
        return "\n".join(parts)


class SessionStateStore:
    """Per-session `SessionState`, RAM only (see module docstring),
    single-process dict -- same boundary as `ConversationMemory`. `_lock`:
    multiple WhatsApp users can issue concurrent `/mesaj` requests within the
    same process, so the public methods are guarded by a `threading.Lock`."""

    def __init__(self) -> None:
        self._states: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        """Returns a COPY of the stored state (so the caller mutating it
        doesn't corrupt the store) -- or an empty `SessionState` if none."""
        with self._lock:
            state = self._states.get(session_id)
            return state.model_copy(deep=True) if state is not None else SessionState()

    def set(self, session_id: str, state: SessionState) -> None:
        with self._lock:
            self._states[session_id] = state

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)


__all__ = [
    "ConstraintDelta",
    "PinnedFlow",
    "SessionState",
    "SessionStateStore",
    "Subject",
    "TaskSnapshot",
]
