"""Combined L0+L1 core: updates session state AND turns this turn's (likely
elliptical/context-embedded) message into an independent query.

The two jobs are ONE reasoning step: resolving "what about the other one"
already requires understanding the goal. Hence the reconciler produces
`resolved_query` (de-ellipsis), `new_state` (current goal) and `transition`
(continue/evolve/switch/digress) together -- L1 is embedded here instead of
being a separate call (hop reduction).

Inertia + snap: state is derived FROM SCRATCH every turn but PRIME-D with the
previous state -- evolution slides smoothly, a clear switch breaks hard. The
anti-stickiness mechanics (subject decay, suspend stack) are LLM-INDEPENDENT
pure code (`_apply`); the LLM produces transition + fields, code applies them
deterministically.

Model: shares the default `LLM_*` (the small intent model) -- a single loaded
model, VRAM-friendly; overridable via `RECONCILER_*`. Thinking is controlled
by `RECONCILER_THINKING` (see thinking.py).

Any failure (network, unparseable JSON) is GRACEFUL: `resolved_query = raw
query`, `transition = continue`, state preserved with only the turn counter
incremented -- a reconciler error NEVER drops the turn.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from medrag.api.answering_model import ProviderError, _post_json
from medrag.api.memory import Message
from medrag.api.reply_contract import extract_json_object
from medrag.api.retrieval.modules.intent_classification.prompt import (
    intent_catalogue_block,
)
from medrag.api.session_state import (
    ConstraintDelta,
    SessionState,
    Subject,
    TaskSnapshot,
)

_TRANSITIONS = ("continue", "evolve", "switch", "digress")
_MAX_SUSPENDED = 3
_HISTORY_TURNS = 6  # number of recent messages taken into the prompt (context suffices, doesn't bloat)


def _build_system_prompt(intent_labels: Sequence[str] | None) -> str:
    """Builds the reconciler system prompt. If `intent_labels` is given
    (fusion), the model is ALSO asked to produce this turn's intent label --
    merging L1 (reconcile) + L2 (classify) into ONE call, cutting a full LLM
    round-trip. If `None` (fusion off), intent isn't asked and the old
    two-call behavior (orchestrator calls a separate classifier) is preserved.

    Labels are injected from outside (factory passes
    `retrieval.core.IntentLabel` values) -- the reconciler isn't hard-wired to
    retrieval and stays testable.

    When fusion is on, the label set is NO LONGER a bare name list but the
    definition+few-shot block from
    `retrieval.modules.intent_classification.intent_catalogue_block()` --
    a bare name list gives the model no signal on how one label differs from
    another (e.g. recommendation vs. aggregation), which contributed to
    `recommendation` being over-dominant in production; the same single-source
    definition/example block the standalone classifier already uses
    (`prompt.py::build_system_prompt`) is reused here."""
    jobs = [
        "1) Track the user's ONGOING goal (across turns, don't look at a single message).",
        (
            "2) Rewrite this turn's message into an independent, self-contained query\n"
            "   (if elliptical/context-embedded; e.g. \"what about the other one\" -> \"de1100's power\").\n"
            "   CRITICAL: resolved_query MUST stay in the SAME language the user actually wrote\n"
            "   this turn's message in -- never translate it to English (or any other language)\n"
            "   even though your own reasoning/output schema here is in English."
        ),
    ]
    fields = [
        (
            '  "resolved_query": "this turn\'s message, self-contained, in the user\'s OWN '
            'language -- unchanged if already clear, NEVER translated",'
        ),
        '  "transition": "continue | evolve | switch | digress",',
        '  "task": "short label for the goal (e.g. \'de1000 vs de1100 comparison\') or null",',
        '  "subjects": ["product/entity codes in scope this turn"],',
        '  "constraints": ["constraints gathered so far (e.g. \'cold environment\')"],',
        (
            '  "dropped_constraints": ["constraints the user just said are no longer '
            'true/superseded this turn (e.g. user said a NEW number/value for '
            'something previously stated)"],'
        ),
        '  "focus": "one sentence: what the user is after right now",',
        '  "confidence": 0.0',
    ]
    intent_hint = ""
    if intent_labels:
        jobs.append(
            "3) Classify the RESOLVED query's intent (what this message wants) "
            "with a single label."
        )
        # Append a comma to the "confidence" line, then insert the intent field.
        fields[-1] = '  "confidence": 0.0,'
        fields.append('  "intent": "exactly one of the labels below"')
        intent_hint = (
            "\n\nintent must be exactly one of the labels defined in the "
            "catalogue below (write nothing else):\n\n" + intent_catalogue_block()
        )
    return (
        "You are a conversation-state tracker. You do these jobs AT THE SAME TIME:\n"
        + "\n".join(jobs)
        + "\n\nYou are given: the previous state (JSON), recent conversation history, this turn's raw message."
        + "\n\nYour output must be ONLY this JSON object, nothing else:\n{\n"
        + "\n".join(fields)
        + "\n}\n\ntransition meanings:\n"
        "- continue: same goal, adding information\n"
        "- evolve:   same goal but focus/constraints shifted\n"
        "- switch:   moved to a new goal (previous goal dropped)\n"
        "- digress:  a one-off aside unrelated to the goal; goal is PRESERVED\n\n"
        "confidence: your confidence that the goal is still active (0.0-1.0)."
        + intent_hint
    )


class ReconcileResult(BaseModel):
    resolved_query: str
    new_state: SessionState
    transition: str
    confidence: float | None = None
    # Only filled when JSON parsing fails -- makes visible what the model
    # actually returned (see module docstring, "graceful error" note). The
    # orchestrator emits `on_trace("reconcile_raw", ...)` when it sees this.
    raw_text: str | None = None
    # Fusion: this turn's intent label when the reconciler also produced it
    # (intent_labels injected AND the model returned a valid label). `None` =
    # fusion off or invalid/missing label -> the orchestrator falls back to a
    # separate classifier (graceful fallback).
    intent: str | None = None


@runtime_checkable
class Reconciler(Protocol):
    async def reconcile(
        self, state: SessionState, history: Sequence[Message], query: str
    ) -> ReconcileResult: ...


def _merge_subjects(
    prev: list[Subject], new_entities: list[str], turn: int, decay_window: int
) -> list[Subject]:
    """Stamps new entities with this turn, merges with previous ones, drops
    those untouched for N turns (recency-decay). Order is preserved: oldest
    first."""
    last: dict[str, int] = {s.entity: s.last_turn for s in prev}
    for e in new_entities:
        if e:
            last[e] = turn
    return [Subject(entity=e, last_turn=t) for e, t in last.items() if turn - t <= decay_window]


def _find_resumable(
    suspended: list[TaskSnapshot], task: str | None, new_entities: list[str]
) -> TaskSnapshot | None:
    """Looks in the suspend stack for a `TaskSnapshot` matching this turn's new
    `task` or `subjects` (the "resume it" decision). On a match, `_apply`'s
    switch branch BASES itself on this snapshot instead of rebuilding from
    scratch -- pure code, independent of the LLM (module docstring's "Inertia
    + snap" principle).

    Task matching is a case-insensitive substring (so abbreviations like
    "let's go back to de1000" after "de1100 comparison" are also caught);
    subject matching is exact entity overlap. The LAST pushed (most recent)
    matching snapshot wins -- `suspended` runs old->new, so `reversed` looks
    from the most recently suspended backwards."""
    task_norm = (task or "").strip().lower()
    entities_norm = {e.strip().lower() for e in new_entities if e.strip()}
    for snap in reversed(suspended):
        snap_task = (snap.task or "").strip().lower()
        if task_norm and snap_task and (task_norm in snap_task or snap_task in task_norm):
            return snap
        snap_entities = {s.entity.strip().lower() for s in snap.subjects}
        if entities_norm and snap_entities and entities_norm & snap_entities:
            return snap
    return None


def _merge_constraints(prev: list[str], new: list[str]) -> list[str]:
    out = list(prev)
    for c in new:
        if c and c not in out:
            out.append(c)
    return out


def _apply_dropped_constraints(
    merged: list[str], prev: list[str], dropped: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Drops, from the output of `_merge_constraints`, the constraints the
    model said are "no longer true/superseded" this turn (pure function).
    Comparison is case-insensitive/strip (exact string matching would be
    fragile since the LLM produces free text).

    Returns: `(remaining, added, removed)` -- `added` = items in `merged` but
    NOT in `prev` (genuinely added this turn), `removed` = items matching
    `dropped` and ACTUALLY removed from `merged` (only existing ones -- a
    constraint the model invented / that never existed is not "dropped")."""
    dropped_norm = {d.strip().lower() for d in dropped if d.strip()}
    prev_norm = {p.strip().lower() for p in prev}
    remaining: list[str] = []
    removed: list[str] = []
    for c in merged:
        if c.strip().lower() in dropped_norm:
            removed.append(c)
        else:
            remaining.append(c)
    added = [c for c in remaining if c.strip().lower() not in prev_norm]
    return remaining, added, removed


def _extract_json(text: str) -> dict | None:
    """Extracts and parses the first {...} block from the model response; None
    if absent.

    The logic moved to `reply_contract.extract_json_object` (so there would be
    no two copies once the answering role got the same guarantee) -- this name
    remains as a thin wrapper for existing callers/tests; behavior is
    identical."""
    return extract_json_object(text)


def _apply(
    state: SessionState, query: str, data: dict | None, decay_window: int,
    *, raw_text: str | None = None, intent_labels: Sequence[str] | None = None,
) -> ReconcileResult:
    """Deterministic state mechanics: reconciles the LLM's transition + fields
    with the previous state. `data=None` (LLM error) means the graceful
    fallback -- if `raw_text` is filled (the model returned something but it
    couldn't be parsed) it is carried into the result; on a pure network error
    (no content at all) it stays `None`.

    If `intent_labels` is given (fusion on), the model's returned `intent`
    field is validated against that set; not in the set or absent leaves
    `intent=None` -> the orchestrator falls back to a separate classifier (no
    invented label leaks through)."""
    turn = state.turn + 1

    if data is None:
        fallback = state.model_copy(deep=True)
        fallback.turn = turn
        return ReconcileResult(resolved_query=query, new_state=fallback,
                               transition="continue", confidence=None,
                               raw_text=raw_text)

    resolved = str(data.get("resolved_query") or query).strip() or query
    transition = data.get("transition")
    if transition not in _TRANSITIONS:
        transition = "continue"
    task = data.get("task")
    task = str(task).strip() if task else None
    new_entities = [str(e).strip() for e in (data.get("subjects") or []) if str(e).strip()]
    new_constraints = [str(c).strip() for c in (data.get("constraints") or []) if str(c).strip()]
    dropped_constraints = [
        str(c).strip() for c in (data.get("dropped_constraints") or []) if str(c).strip()
    ]
    focus = str(data.get("focus") or "").strip()
    raw_conf = data.get("confidence")
    confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else None
    # Fusion: the intent is accepted only if it's in the injected label set.
    intent = None
    if intent_labels:
        cand = str(data.get("intent") or "").strip()
        intent = cand if cand in set(intent_labels) else None

    if transition == "digress":
        # The aside is temporary: the active goal is PRESERVED, only the turn
        # advances.
        new = state.model_copy(deep=True)
        new.turn = turn
        new.confidence = confidence
        return ReconcileResult(resolved_query=resolved, new_state=new,
                               transition=transition, confidence=confidence,
                               intent=intent)

    if transition == "switch":
        suspended = list(state.suspended)
        resumable = _find_resumable(suspended, task, new_entities)
        if resumable is not None:
            suspended.remove(resumable)
        if state.task is not None:
            suspended.append(state.snapshot())
            suspended = suspended[-_MAX_SUSPENDED:]
        if resumable is not None:
            # Resume: NOT from scratch -- the suspended old goal is taken as
            # the BASE and merged with this turn's new information (same
            # mechanics as _merge_subjects/_merge_constraints -- symmetric with
            # continuing; the only difference is the starting point).
            new = SessionState(
                task=task or resumable.task,
                subjects=_merge_subjects(resumable.subjects, new_entities, turn, decay_window),
                constraints=_merge_constraints(resumable.constraints, new_constraints),
                focus=focus or resumable.focus,
                suspended=suspended,
                turn=turn,
                confidence=confidence,
            )
            return ReconcileResult(resolved_query=resolved, new_state=new,
                                   transition="resume", confidence=confidence,
                                   intent=intent)
        new = SessionState(
            task=task,
            subjects=_merge_subjects([], new_entities, turn, decay_window),
            constraints=new_constraints,
            focus=focus,
            suspended=suspended,
            turn=turn,
            confidence=confidence,
        )
        return ReconcileResult(resolved_query=resolved, new_state=new,
                               transition=transition, confidence=confidence,
                               intent=intent)

    # continue / evolve: merge with the previous state (inertia), then apply
    # this turn's correction (dropped_constraints).
    merged_constraints = _merge_constraints(state.constraints, new_constraints)
    constraints, added, removed = _apply_dropped_constraints(
        merged_constraints, state.constraints, dropped_constraints
    )
    new = SessionState(
        task=task or state.task,
        subjects=_merge_subjects(state.subjects, new_entities, turn, decay_window),
        constraints=constraints,
        focus=focus or state.focus,
        suspended=list(state.suspended),
        turn=turn,
        confidence=confidence,
        constraint_delta=ConstraintDelta(added=added, removed=removed) if (added or removed) else None,
    )
    return ReconcileResult(resolved_query=resolved, new_state=new,
                           transition=transition, confidence=confidence,
                           intent=intent)


def _reconciler_endpoint(env: Mapping[str, str]) -> tuple[str, str, str | None]:
    """`RECONCILER_*` override, else `LLM_*` fallback (shares the small intent
    model by default, VRAM-friendly)."""
    base = (env.get("RECONCILER_BASE_URL") or env.get("LLM_BASE_URL") or "").strip()
    model = (env.get("RECONCILER_MODEL") or env.get("LLM_MODEL") or "").strip()
    api_key = env.get("RECONCILER_API_KEY") or env.get("LLM_API_KEY") or None
    return base, model, api_key


def _reconciler_provider(env: Mapping[str, str]) -> str:
    """`RECONCILER_PROVIDER` override, else `LLM_PROVIDER` fallback, else
    `"openai"` (same default as `answering_model._common`) -- the
    reconciler/continuation had no provider concept until now (only
    base_url/model/api_key); it was added together with the num_ctx need."""
    return (env.get("RECONCILER_PROVIDER") or env.get("LLM_PROVIDER") or "openai").strip().lower()


def _reconciler_num_ctx(env: Mapping[str, str], provider: str) -> int | None:
    """`RECONCILER_NUM_CTX` override, else `LLM_NUM_CTX` fallback -- the same
    `RECONCILER_*` -> `LLM_*` pattern followed by BASE_URL/MODEL/API_KEY. If
    both are empty AND `provider == "ollama"`, fall back to
    `answering_model._DEFAULT_OLLAMA_NUM_CTX` and write it back to
    `RECONCILER_NUM_CTX` for observability (same pattern as
    parser/facts/answering_model). An invalid value raises `ProviderError`."""
    from medrag.api.answering_model import _DEFAULT_OLLAMA_NUM_CTX

    raw = (env.get("RECONCILER_NUM_CTX") or env.get("LLM_NUM_CTX") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise ProviderError(f"RECONCILER_NUM_CTX/LLM_NUM_CTX must be an integer, got {raw!r}")
    if provider == "ollama":
        num_ctx = _DEFAULT_OLLAMA_NUM_CTX
        os.environ["RECONCILER_NUM_CTX"] = str(num_ctx)
        return num_ctx
    return None


class LLMReconciler:
    """Implements the `Reconciler` protocol against a remote,
    OpenAI-compatible endpoint. Blocking HTTP runs via `asyncio.to_thread`
    without blocking the main loop (same pattern as answering_model /
    db_query)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        provider: str | None = None,
        num_ctx: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        decay_window: int = 6,
        intent_labels: Sequence[str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # When `provider == "ollama" and num_ctx` are set, the request goes to
        # native `/api/chat` (see `_call` -- the same distinction as
        # answering_model.py).
        self._provider = provider
        self._num_ctx = num_ctx
        self._extra_body = dict(extra_body or {})
        self._decay_window = decay_window
        # Fusion: if given, the system prompt also requests the intent, and
        # `_apply` validates the returned one against this set. None = fusion
        # off.
        self._intent_labels = tuple(intent_labels) if intent_labels else None
        self._system_prompt = _build_system_prompt(self._intent_labels)

    def _build_user(self, state: SessionState, history: Sequence[Message], query: str) -> str:
        state_json = state.model_dump_json(exclude={"suspended"})
        recent = history[-_HISTORY_TURNS:]
        lines = [f"{m.role}: {m.content}" for m in recent] or ["(no history, first turn)"]
        return (
            f"Previous state (JSON):\n{state_json}\n\n"
            f"Recent conversation history:\n" + "\n".join(lines) + "\n\n"
            f"This turn's raw message:\n{query}"
        )

    def _call(self, user: str) -> tuple[dict | None, str | None]:
        """Returns `(parsed_data, raw_text)`. On a network/endpoint error both
        are `None` (there is no content to show). If the model returned
        something that couldn't be parsed as JSON, `(None, raw_text)` --
        `reconcile()` carries it into `ReconcileResult.raw_text` (makes the
        intermediate model's output visible)."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]
        try:
            if self._provider == "anthropic":
                resp = self._call_anthropic(messages)
            elif self._provider == "ollama" and self._num_ctx:
                resp = self._call_native(messages)
            else:
                resp = self._call_openai_compat(messages)
        except ProviderError:
            return None, None  # network/endpoint error -> graceful fallback
        try:
            content = str(resp["content"] or "")
        except (KeyError, TypeError):
            return None, None
        data = _extract_json(content)
        return (data, None) if data is not None else (None, content)

    def _call_anthropic(self, messages: list[dict[str, str]]) -> dict:
        """Called when `self._provider == "anthropic"` -- see the docstring of
        `core/llm/anthropic.py::call_anthropic`. Returns the SAME shape as
        `_call_openai_compat`/`_call_native` (`{"content": ...}`) so `_call`'s
        caller doesn't branch."""
        from medrag.core.llm.anthropic import call_anthropic

        content = call_anthropic(
            messages, model=self._model, base_url=self._base_url,
            api_key=self._api_key, timeout=self._timeout,
        )
        return {"content": content}

    def _call_openai_compat(self, messages: list[dict[str, str]]) -> dict:
        base_payload = {"model": self._model, "messages": messages}
        url = f"{self._base_url}/chat/completions"
        try:
            resp = _post_json(url, {**base_payload, **self._extra_body},
                              api_key=self._api_key, timeout=self._timeout)
        except ProviderError as exc:
            if self._extra_body and any(str(k).lower() in str(exc).lower() for k in self._extra_body):
                resp = _post_json(url, base_payload, api_key=self._api_key, timeout=self._timeout)
            else:
                raise
        try:
            return {"content": resp["choices"][0]["message"]["content"]}
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"unexpected chat completion response shape: {str(resp)[:500]}"
            ) from exc

    def _call_native(self, messages: list[dict[str, str]]) -> dict:
        """Only called when `provider == "ollama" and num_ctx` are set --
        Ollama's native `/api/chat` (see answering_model.py::_call_native, the
        same distinction). The `reasoning_effort` key in `extra_body` signals
        that thinking was EXPLICITLY disabled -- on the native path it is
        translated to `think:false`."""
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self._num_ctx},
        }
        if "reasoning_effort" in self._extra_body:
            payload["think"] = False
        url = f"{self._base_url.removesuffix('/v1')}/api/chat"
        resp = _post_json(url, payload, api_key=self._api_key, timeout=self._timeout)
        try:
            return {"content": resp["message"]["content"]}
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                f"unexpected chat completion response shape: {str(resp)[:500]}"
            ) from exc

    async def reconcile(
        self, state: SessionState, history: Sequence[Message], query: str
    ) -> ReconcileResult:
        user = self._build_user(state, history, query)
        data, raw_text = await asyncio.to_thread(self._call, user)
        return _apply(state, query, data, self._decay_window,
                      raw_text=raw_text, intent_labels=self._intent_labels)


__all__ = [
    "LLMReconciler",
    "ReconcileResult",
    "Reconciler",
    "_build_system_prompt",
    "_reconciler_endpoint",
]
