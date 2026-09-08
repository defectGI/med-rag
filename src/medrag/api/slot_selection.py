"""A small LLM call that decides WHICH clarification question the
`recommendation` flow will ASK -- the exact same pattern as
`continuation.py` (Protocol + real implementation + graceful degrade to
`None` when the endpoint is absent), applied to a different decision:
instead of "is the user continuing", "which attribute is most useful to ask
about for this product family".

Why a separate module (instead of EXTENDING continuation.py): the two have
different input/output shapes (yes/no vs. a key choice + `None`) and need
different prompts -- same rationale as `ComparisonFlow`/`RecommendationFlow`
NOT SHARING their own `_CANCEL_WORDS` copies (module docstring,
flows/recommendation.py): two different decisions must not become coupled.

The candidates themselves (`slot_candidates.discover_informative_slots`) are
LLM-free, produced by deterministic SQL -- this module only makes the
"which of the given candidates is most useful in this conversation context"
decision; it does NOT write new SQL."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from medrag.api.slot_candidates import SlotCandidate

_SYSTEM_PROMPT = (
    "You help a product-recommendation chat pick the SINGLE most useful "
    "clarifying question to ask next. You are given the conversation context "
    "(what the user is looking for, what has already been asked/answered) and "
    "a JSON list of CANDIDATE attributes -- each with a `key`, a `question` "
    "(the exact question text to ask), and `n_distinct` (how many different "
    "values this attribute takes among matching products -- higher means "
    "asking about it narrows the choice down more). Pick the candidate whose "
    "question is most relevant to the user's stated need; when relevance is "
    "similar, prefer higher `n_distinct`. If NONE of the candidates are worth "
    "asking (e.g. the user already implied the answer, or none are relevant "
    "at all), say so. Respond with ONLY a JSON object: {\"key\": \"<the "
    "chosen candidate's key>\"} or {\"key\": null} -- no other text, no "
    "explanation, no markdown fence."
)


@runtime_checkable
class SlotSelector(Protocol):
    """`candidates` is a non-empty list of `SlotCandidate`. Return: the chosen
    `key` (must be a `key` present in `candidates`) or `None` (none worth
    asking)."""

    async def choose(self, context: str, candidates: list[SlotCandidate]) -> str | None: ...


class LLMSlotSelector:
    """Implements `SlotSelector` against a remote, OpenAI-compatible endpoint
    -- SAME `_post_json` client as `continuation.LLMContinuationChecker`, SAME
    native-ollama/bodyless-retry-if-`extra_body`-rejected pattern."""

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
        self._provider = provider
        self._num_ctx = num_ctx
        self._extra_body = dict(extra_body or {})

    def _call(self, context: str, candidates: list[SlotCandidate]) -> str | None:
        from medrag.api.answering_model import ProviderError

        payload_candidates = [
            {"key": c.key, "question": c.question, "n_distinct": c.n_distinct}
            for c in candidates
        ]
        user = (
            f"Context: {context}\n\n"
            f"Candidates:\n{json.dumps(payload_candidates, ensure_ascii=False)}"
        )
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

        candidate_keys = {c.key for c in candidates}
        raw = content.strip()
        try:
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            chosen = data.get("key")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(f"unexpected slot-selection answer: {content!r}") from exc
        if chosen is None:
            return None
        if chosen not in candidate_keys:
            # The model invented a key OUTSIDE the candidates -- accepting it
            # would be carrying an invention into the DB (fail-loud); drop to
            # the ProviderError below.
            raise ProviderError(f"slot selector chose a key outside the candidate list: {chosen!r}")
        return chosen

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

    async def choose(self, context: str, candidates: list[SlotCandidate]) -> str | None:
        return await asyncio.to_thread(self._call, context, candidates)


def _slot_selector_endpoint(env: Mapping[str, str]) -> tuple[str, str, str | None]:
    """`SLOT_SELECTOR_*` override, else `LLM_*` fallback -- the exact same
    pattern as `continuation._continuation_endpoint`."""
    base = (env.get("SLOT_SELECTOR_BASE_URL") or env.get("LLM_BASE_URL") or "").strip()
    model = (env.get("SLOT_SELECTOR_MODEL") or env.get("LLM_MODEL") or "").strip()
    api_key = env.get("SLOT_SELECTOR_API_KEY") or env.get("LLM_API_KEY") or None
    return base, model, api_key


def _slot_selector_provider(env: Mapping[str, str]) -> str:
    return (env.get("SLOT_SELECTOR_PROVIDER") or env.get("LLM_PROVIDER") or "openai").strip().lower()


def _slot_selector_num_ctx(env: Mapping[str, str], provider: str) -> int | None:
    """`SLOT_SELECTOR_NUM_CTX` override, else `LLM_NUM_CTX` fallback -- the
    exact same pattern as `continuation._continuation_num_ctx` (if both are
    empty AND provider=ollama, fall back to the default and write it back)."""
    from medrag.api.answering_model import _DEFAULT_OLLAMA_NUM_CTX, ProviderError

    raw = (env.get("SLOT_SELECTOR_NUM_CTX") or env.get("LLM_NUM_CTX") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise ProviderError(f"SLOT_SELECTOR_NUM_CTX/LLM_NUM_CTX must be an integer, got {raw!r}")
    if provider == "ollama":
        num_ctx = _DEFAULT_OLLAMA_NUM_CTX
        os.environ["SLOT_SELECTOR_NUM_CTX"] = str(num_ctx)
        return num_ctx
    return None


def build_slot_selector_from_env(env: Mapping[str, str] | None = None) -> LLMSlotSelector | None:
    """Returns `None` when the endpoint is unconfigured (base_url/model empty)
    -- the caller (`RecommendationFlow`) then falls back to the old
    hardcoded-3-key behavior (the SAME graceful-degrade principle as
    `build_continuation_checker_from_env`)."""
    from medrag.api.thinking import thinking_off_body

    resolved_env = env if env is not None else os.environ
    base_url, model, api_key = _slot_selector_endpoint(resolved_env)
    if not base_url or not model:
        return None
    provider = _slot_selector_provider(resolved_env)
    num_ctx = _slot_selector_num_ctx(resolved_env, provider)
    return LLMSlotSelector(
        base_url=base_url, model=model, api_key=api_key,
        provider=provider, num_ctx=num_ctx,
        extra_body=thinking_off_body(resolved_env, "SLOT_SELECTOR"),
    )


__all__ = ["LLMSlotSelector", "SlotSelector", "build_slot_selector_from_env"]
