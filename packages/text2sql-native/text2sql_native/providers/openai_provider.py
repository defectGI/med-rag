"""OpenAI-compatible provider.

Works with the OpenAI API and any OpenAI-compatible gateway (via ``base_url``).
Uses ``response_format`` JSON-schema mode for structured output when available.
The ``openai`` SDK is imported lazily, so the library imports without it.
"""

from __future__ import annotations

from typing import Any

from ..errors import ProviderError
from ..logging_utils import get_logger
from ..types import LLMResponse, TokenUsage
from .base import LLMProvider, register_provider

logger = get_logger("providers.openai")


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """Provider backed by the OpenAI Chat Completions API."""

    def _client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with 'pip install openai'."
            ) from exc
        # base_url=None lets the SDK use its default endpoint.
        return OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)

    @staticmethod
    def _usage(response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
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
        logger.debug("OpenAI completion request (model=%s)", self.model)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - normalize SDK errors
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        text = response.choices[0].message.content or ""
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
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            },
        }
        logger.debug("OpenAI structured request (model=%s)", self.model)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as exc:  # noqa: BLE001
            # Some OpenAI-compatible gateways do not support json_schema mode.
            # Fall back to schema-in-prompt + text parsing.
            logger.warning(
                "OpenAI structured output failed (%s). Falling back to text parsing.",
                exc,
            )
            return super().complete_json(
                system=system,
                user=user,
                json_schema=json_schema,
                schema_name=schema_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, usage=self._usage(response), model=self.model, raw=response)
