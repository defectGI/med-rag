"""Production wiring: build a real chat model from ``.env``.

Kept apart from :mod:`classifier` so the classifier stays provider-agnostic and
unit-testable without ``langchain-openai`` installed. ``langchain-openai`` is
imported lazily *inside* the function, so importing this module (or the
classifier) never requires the ``intent_classification`` extra.

GPU/inference rule (ARCHITECTURE.md #10): this only points at a remote
OpenAI-compatible endpoint. It performs no local inference. Calling the returned
model reaches the endpoint — do that on the inference machine, not here.
"""

from __future__ import annotations

import json
import os

from langchain_core.language_models import BaseChatModel

# Thinking/reasoning suppression body -- OPT-IN, only when LLM_THINKING is
# explicitly "off". A hybrid-reasoning model left thinking ON can spend the
# whole token budget on a hidden <think> block and return empty/unparseable
# content for a short-answer task like a single intent label. A consumer's own
# startup preflight should probe this same endpoint with the same body before
# traffic starts. Suppression is opt-in (unset/on => inject nothing) because
# unlike a raw-HTTP answering model, langchain ChatOpenAI has NO "retry without
# the body on rejection" hook -- a default-injected body a server rejects would
# hard-fail every classify call.
_DEFAULT_THINKING_OFF_BODY: dict = {"reasoning_effort": "none"}


def _thinking_extra_body() -> dict:
    """Suppression body when LLM_THINKING is explicitly `off`; empty (neutral)
    when unset or on."""
    if (os.getenv("LLM_THINKING") or "").strip().lower() != "off":
        return {}
    raw = (os.getenv("LLM_THINKING_OFF_BODY") or "").strip()
    if not raw:
        return dict(_DEFAULT_THINKING_OFF_BODY)
    body = json.loads(raw)  # invalid JSON -> loud failure (config error)
    if not isinstance(body, dict):
        raise ValueError("LLM_THINKING_OFF_BODY must be a JSON object")  # noqa: TRY004 -- config error, not a type error; same ValueError contract as the shared thinking helper
    return body


def _build_anthropic_chat_model() -> BaseChatModel:
    """The branch `build_default_chat_model` takes when `LLM_PROVIDER=anthropic`.

    The Anthropic Messages API does not speak the OpenAI-compatible `/v1`
    contract `ChatOpenAI` expects, so the official `langchain-anthropic` package
    is used -- the LangChain counterpart of `medrag.core.llm.anthropic` (the
    stdlib-urllib translation the consumer's own roles use), reading the same
    environment variables (`LLM_MODEL`/`LLM_API_KEY`/`LLM_TIMEOUT`/
    `LLM_MAX_RETRIES`). `request_logprobs` and `LLM_THINKING_OFF_BODY` are NOT
    applied here -- both are specific to OpenAI-compatible endpoints and have no
    Anthropic equivalent."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "langchain-anthropic is required for LLM_PROVIDER=anthropic. "
            "Install the module extra:  pip install -e '.[intent_classification]'"
        ) from exc

    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL must be set (see .env.example).")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY must be set when LLM_PROVIDER=anthropic.")
    timeout = float(os.getenv("LLM_TIMEOUT", "60"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # The `temperature=0` hardcode is rejected with an API 400 by some Claude
    # models (the extended-thinking-required ones: "`temperature` is deprecated
    # for this model"), so it is not passed here. Determinism is lost for those
    # models, which still beats not working at all; a model-specific gate can be
    # added here later if needed.
    return ChatAnthropic(
        model=model, api_key=api_key,
        timeout=timeout, max_retries=max_retries,
    )


def build_default_chat_model(*, request_logprobs: bool = True) -> BaseChatModel:
    """Construct a chat model pointed at the configured endpoint --
    ``ChatOpenAI`` by default, ``ChatAnthropic`` when ``LLM_PROVIDER=anthropic``
    (see :func:`_build_anthropic_chat_model`).

    Reads ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` (and optional
    ``LLM_TIMEOUT`` / ``LLM_MAX_RETRIES`` / ``LLM_THINKING`` /
    ``LLM_THINKING_OFF_BODY``) from the environment, loading a local ``.env`` if
    present. ``temperature`` is pinned to 0 for deterministic classification.

    ``request_logprobs`` asks the endpoint for token logprobs (used to populate
    ``IntentResult.confidence``). If the endpoint ignores or omits them the
    classifier degrades gracefully to ``confidence=None`` — so leaving this on
    is safe even against endpoints without logprobs support. Ignored for the
    Anthropic branch (no logprobs support there).

    ``LLM_THINKING`` (default off) controls whether a reasoning model is asked
    to think; when off, ``LLM_THINKING_OFF_BODY`` (default
    ``{"reasoning_effort": "none"}``) is merged into every request body.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is a base dep, but don't hard-fail
        pass

    if (os.getenv("LLM_PROVIDER") or "").strip().lower() == "anthropic":
        return _build_anthropic_chat_model()

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "langchain-openai is required to build the default chat model. "
            "Install the module extra:  pip install -e '.[intent_classification]'"
        ) from exc

    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if not base_url or not model:
        raise RuntimeError(
            "LLM_BASE_URL and LLM_MODEL must be set (see .env.example). "
            f"Got LLM_BASE_URL={base_url!r}, LLM_MODEL={model!r}."
        )
    api_key = os.getenv("LLM_API_KEY", "EMPTY")  # OpenAI clients require non-empty
    timeout = float(os.getenv("LLM_TIMEOUT", "60"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

    kwargs: dict = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": 0,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if request_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5

    extra_body = _thinking_extra_body()
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


__all__ = ["build_default_chat_model"]
