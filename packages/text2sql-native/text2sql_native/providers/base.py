"""Provider-agnostic LLM client layer.

An :class:`LLMProvider` turns a prompt into text. Two entry points:

* :meth:`complete` — free-form text. Used by Stage 2.
* :meth:`complete_json` — output forced to a JSON object that matches a given
  JSON schema. Used by Stage 1. Providers with native structured output
  (``response_format`` / tool-calling) should use it. The base class gives a
  plain-text fallback for the rest.

Adding a provider is one subclass plus the ``@register_provider`` decorator.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..logging_utils import get_logger
from ..types import LLMResponse

logger = get_logger("providers")


class LLMProvider(ABC):
    """Base class for all providers.

    Args:
        model: Model id for this instance.
        api_key: Secret key. Never logged.
        base_url: Optional endpoint override, for OpenAI-compatible gateways.
        timeout: Request timeout in seconds.
    """

    #: Registry name. Set by ``@register_provider``.
    name: str = ""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        # Only meaningful to providers that talk to a local Ollama server
        # (see ollama_provider.py) -- other providers accept and ignore it,
        # so build_provider() can pass it uniformly without a per-provider
        # branch.
        self._num_ctx = num_ctx

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Return a free-form text completion."""
        raise NotImplementedError

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Return a completion forced to a JSON object.

        Default: append the JSON schema to the prompt and rely on text parsing.
        Providers with native structured output should override this for a
        stronger guarantee.
        """
        augmented_user = (
            f"{user}\n\n"
            f"Respond with a single JSON object that matches this JSON schema. "
            f"Do not wrap it in markdown fences and do not add any comment.\n"
            f"JSON schema:\n{json.dumps(json_schema)}"
        )
        return self.complete(
            system=system,
            user=augmented_user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


# Registry / factory.
_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(name: str) -> Callable[[type[LLMProvider]], type[LLMProvider]]:
    """Class decorator. Registers a provider under ``name``."""

    def decorator(cls: type[LLMProvider]) -> type[LLMProvider]:
        cls.name = name
        _REGISTRY[name.lower()] = cls
        return cls

    return decorator


def registered_providers() -> list[str]:
    """Return the registered provider names, sorted."""
    return sorted(_REGISTRY)


def get_provider_class(name: str) -> type[LLMProvider]:
    """Look up a provider class by name.

    Raises:
        ProviderError: If no provider is registered under ``name``.
    """
    from ..errors import ProviderError

    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ProviderError(
            f"Unknown provider '{name}'. Registered providers: "
            f"{', '.join(registered_providers()) or '(none)'}."
        )
    return cls
