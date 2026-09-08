"""Answers soft-skill continuation/confirmation decisions (is the pin still
valid, is an offer accepted or rejected) with a small LLM call instead of
hardcoded word sets -- cancel-word logic was weak: hardcoded and inelegant,
for both triggering and ending.

Root-cause research found four call sites falling into the SAME pattern:
`RecommendationFlow.check_pin`, `ComparisonFlow.check_pin`,
`DocDownloadFlow.check_pin`, `Orchestrator._is_affirmative`. All four were
really natural-language interpretation tasks (did the user change the
subject, did they accept the offer) imitated with a fixed word list -- no
expression outside the list ("actually I was going to ask something else",
"uh-huh") could ever be classified correctly. This module collapses the four
into ONE shared protocol + implementation.

Deliberately generic: `check(query, context)` -- `context` is the caller's
natural-language description of "what is expected right now", carrying NO
flow-specific schema/field names (flows stay independent of each other and
only inject this shared service -- same pattern as `SqlTopNFlow` being shared
by three flows, see factory.py::build_flows_from_env).

The old `flows/base.py::PinnableFlow` docstring principle "deliberately does
NOT require an LLM" is CONSCIOUSLY REVISED here (see the updated note in that
file) -- the cheapness of the pinned path was traded for correctness after
complaints that "both triggering and ending are too dominant".

Network/error problems are NOT swallowed here -- `ProviderError`/unexpected
response shape propagates as-is; the caller already falls back to its own
safe default (`Orchestrator._check_pin` treats the exception as "pin broken",
`Orchestrator._is_affirmative` likewise as "no/rejected") -- this module does
NOT REPEAT its own try/except (DIFFERENT from reconciler's "an error turn
never drops the turn" principle: there an EXISTING escape hatch is being
reused, not inventing a second one)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

_SYSTEM_PROMPT = (
    "You are a short yes/no classifier. You are given a CONTEXT (an answer/"
    "confirmation currently expected) and the user's latest message. Decide "
    "whether the user's message is RELEVANT TO/CONTINUES this context (yes) "
    "or changes the subject/rejects/cancels it (no). Answer with ONLY 'yes' "
    "or 'no' -- no other word, punctuation, or explanation."
)


@runtime_checkable
class ContinuationChecker(Protocol):
    """`context` is a natural-language description of "what is expected right
    now" (e.g. "the product family was asked; the user is expected to answer
    or give up"). Returns `True` = the message is relevant to / continues this
    context, `False` = subject changed / rejected / cancelled."""

    async def check(self, query: str, context: str) -> bool: ...


class LLMContinuationChecker:
    """Implements `ContinuationChecker` against a remote, OpenAI-compatible
    endpoint -- SAME `_post_json` client as
    `answering_model.OpenAICompatAnsweringModel`/`reconciler.LLMReconciler`,
    SAME "retry without body if extra_body rejected" pattern."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 15.0,
        provider: str | None = None,
        num_ctx: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        # When `provider == "ollama" and num_ctx` are set, the request goes to
        # native `/api/chat` (see `_call` -- same distinction as
        # answering_model.py/reconciler.py).
        self._provider = provider
        self._num_ctx = num_ctx
        self._extra_body = dict(extra_body or {})

    def _call(self, query: str, context: str) -> bool:
        # Deferred import: `chatbot.answering_model` imports this module at
        # module level (via the flows package, see flows/comparison.py) --
        # importing at top level would create a circular import
        # (answering_model imports `chatbot.flows.base` before defining its
        # own `ProviderError`). Importing at call time (where it's needed
        # nowhere else until `ProviderError` is raised) breaks the cycle.
        from medrag.api.answering_model import ProviderError

        user = f"Context: {context}\n\nUser's message: {query}"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        if self._provider == "anthropic":
            content = self._call_anthropic(messages)
        elif self._provider == "ollama" and self._num_ctx:
            content = self._call_native(messages)
        else:
            content = self._call_openai_compat(messages)
        normalized = content.strip().lower()
        if normalized.startswith("yes"):
            return True
        if normalized.startswith("no"):
            return False
        raise ProviderError(f"unexpected yes/no answer: {content!r}")

    def _call_anthropic(self, messages: list[dict[str, str]]) -> str:
        """Called when `self._provider == "anthropic"` -- see the docstring of
        `core/llm/anthropic.py::call_anthropic`."""
        from medrag.core.llm.anthropic import call_anthropic

        return call_anthropic(
            messages, model=self._model, base_url=self._base_url,
            api_key=self._api_key, timeout=self._timeout,
        )

    def _call_openai_compat(self, messages: list[dict[str, str]]) -> str:
        from medrag.api.answering_model import ProviderError, _post_json

        base_payload = {"model": self._model, "messages": messages}
        url = f"{self._base_url}/chat/completions"
        try:
            resp = _post_json(
                url, {**base_payload, **self._extra_body},
                api_key=self._api_key, timeout=self._timeout,
            )
        except ProviderError as exc:
            if self._extra_body and any(
                str(k).lower() in str(exc).lower() for k in self._extra_body
            ):
                resp = _post_json(url, base_payload, api_key=self._api_key, timeout=self._timeout)
            else:
                raise
        try:
            return str(resp["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"unexpected chat completion response shape: {str(resp)[:500]}"
            ) from exc

    def _call_native(self, messages: list[dict[str, str]]) -> str:
        """Only called when `provider == "ollama" and num_ctx` are set --
        Ollama's native `/api/chat` (same distinction as
        answering_model.py/reconciler.py). The `reasoning_effort` key in
        `extra_body` signals that thinking was EXPLICITLY disabled -- on the
        native path it is translated to `think:false`."""
        from medrag.api.answering_model import ProviderError, _post_json

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
            return str(resp["message"]["content"] or "")
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                f"unexpected chat completion response shape: {str(resp)[:500]}"
            ) from exc

    async def check(self, query: str, context: str) -> bool:
        return await asyncio.to_thread(self._call, query, context)


def _continuation_endpoint(env: Mapping[str, str]) -> tuple[str, str, str | None]:
    """`CONTINUATION_*` override, else `LLM_*` fallback -- the exact same
    pattern as `reconciler._reconciler_endpoint` (shared small model: a single
    loaded model, VRAM-friendly)."""
    base = (env.get("CONTINUATION_BASE_URL") or env.get("LLM_BASE_URL") or "").strip()
    model = (env.get("CONTINUATION_MODEL") or env.get("LLM_MODEL") or "").strip()
    api_key = env.get("CONTINUATION_API_KEY") or env.get("LLM_API_KEY") or None
    return base, model, api_key


def _continuation_provider(env: Mapping[str, str]) -> str:
    """`CONTINUATION_PROVIDER` override, else `LLM_PROVIDER` fallback, else
    `"openai"` -- the exact same pattern as `reconciler._reconciler_provider`."""
    return (env.get("CONTINUATION_PROVIDER") or env.get("LLM_PROVIDER") or "openai").strip().lower()


def _continuation_num_ctx(env: Mapping[str, str], provider: str) -> int | None:
    """`CONTINUATION_NUM_CTX` override, else `LLM_NUM_CTX` fallback -- the
    exact same pattern as `reconciler._reconciler_num_ctx`: if both are empty
    AND `provider == "ollama"`, fall back to
    `answering_model._DEFAULT_OLLAMA_NUM_CTX` and write it back to
    `CONTINUATION_NUM_CTX` for observability. An invalid value raises
    `ProviderError`."""
    from medrag.api.answering_model import _DEFAULT_OLLAMA_NUM_CTX, ProviderError

    raw = (env.get("CONTINUATION_NUM_CTX") or env.get("LLM_NUM_CTX") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise ProviderError(f"CONTINUATION_NUM_CTX/LLM_NUM_CTX must be an integer, got {raw!r}")
    if provider == "ollama":
        num_ctx = _DEFAULT_OLLAMA_NUM_CTX
        os.environ["CONTINUATION_NUM_CTX"] = str(num_ctx)
        return num_ctx
    return None


def build_continuation_checker_from_env(
    env: Mapping[str, str] | None = None,
) -> LLMContinuationChecker | None:
    """Returns `None` when the endpoint is unconfigured (base_url/model empty)
    -- the caller then falls back to the old (hardcoded word list) behavior
    (the SAME graceful-degrade principle as `build_reconciler_from_env`
    returning `None`, see factory.py)."""
    from medrag.api.thinking import thinking_off_body

    resolved_env = env if env is not None else os.environ
    base_url, model, api_key = _continuation_endpoint(resolved_env)
    if not base_url or not model:
        return None
    # Reload/cold-start fix (same note as reconciler.py::
    # build_reconciler_from_env): since CONTINUATION_* may also share LLM_*,
    # switch to the native path so a num_ctx-less `/v1` call doesn't collide
    # with text2sql linking's native num_ctx-carrying call.
    provider = _continuation_provider(resolved_env)
    num_ctx = _continuation_num_ctx(resolved_env, provider)
    return LLMContinuationChecker(
        base_url=base_url, model=model, api_key=api_key,
        provider=provider, num_ctx=num_ctx,
        extra_body=thinking_off_body(resolved_env, "CONTINUATION"),
    )


__all__ = [
    "ContinuationChecker",
    "LLMContinuationChecker",
    "build_continuation_checker_from_env",
]
