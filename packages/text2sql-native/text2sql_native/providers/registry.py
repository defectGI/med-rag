"""Factory that builds a provider from a :class:`ProviderSelection`.

Importing this module registers the built-in providers as a side effect.
"""

from __future__ import annotations

from ..config import ProviderSelection
from .base import LLMProvider, get_provider_class, registered_providers

# Importing these runs their @register_provider decorators.
from . import anthropic_provider as _anthropic  # noqa: F401
from . import ollama_provider as _ollama  # noqa: F401
from . import openai_provider as _openai  # noqa: F401


def build_provider(
    selection: ProviderSelection, *, timeout: float, num_ctx: int | None = None
) -> LLMProvider:
    """Build the provider described by ``selection``.

    Args:
        selection: Provider/model/base_url/api_key from ``.env``.
        timeout: Request timeout in seconds, from the stage config.
        num_ctx: Context window to force (``ollama`` provider only, from
            ``config.toml``'s ``[linking]``/``[generation]`` -- see
            ``StageSettings.num_ctx``). Every other provider accepts and
            ignores it.

    Returns:
        A ready :class:`LLMProvider`.

    Raises:
        ProviderError: If the provider name is not registered.
    """
    cls = get_provider_class(selection.provider)
    return cls(
        model=selection.model,
        api_key=selection.api_key,
        base_url=selection.base_url,
        timeout=timeout,
        num_ctx=num_ctx,
    )


__all__ = ["build_provider", "registered_providers"]
