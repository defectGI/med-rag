"""Anthropic provider.

Uses the Messages API. Structured output is forced with tool-calling: one tool
whose ``input_schema`` is the JSON schema, plus ``tool_choice`` to force it. The
``anthropic`` SDK is imported lazily.
"""

from __future__ import annotations

import json
from typing import Any

from ..errors import ProviderError
from ..logging_utils import get_logger
from ..types import LLMResponse, TokenUsage
from .base import LLMProvider, register_provider

logger = get_logger("providers.anthropic")


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """Provider backed by the Anthropic Messages API."""

    def _client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with 'pip install anthropic'."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": self._api_key, "timeout": self._timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return Anthropic(**kwargs)

    @staticmethod
    def _usage(response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage()
        prompt = getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "output_tokens", 0) or 0
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._client()
        logger.debug("Anthropic completion request (model=%s)", self.model)
        try:
            response = client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(text=text, usage=self._usage(response), model=self.model, raw=response)

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
        client = self._client()
        tool = {
            "name": schema_name,
            "description": "Return the structured result with this tool.",
            "input_schema": json_schema,
        }
        logger.debug("Anthropic structured (tool-use) request (model=%s)", self.model)
        try:
            response = client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=[tool],
                tool_choice={"type": "tool", "name": schema_name},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic structured request failed: {exc}") from exc

        # Take the tool-use block's input as the JSON payload.
        for block in response.content:
            if getattr(block, "type", "") == "tool_use":
                text = json.dumps(block.input)
                return LLMResponse(
                    text=text, usage=self._usage(response), model=self.model, raw=response
                )

        # No tool call came back. Return whatever text there is, for the parser.
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(text=text, usage=self._usage(response), model=self.model, raw=response)
