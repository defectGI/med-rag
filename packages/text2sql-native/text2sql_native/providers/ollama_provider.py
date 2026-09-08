"""Native Ollama provider.

Talks to Ollama's own ``/api/chat`` endpoint (not the OpenAI-compatible
``/v1`` one). This is the one place `base_url`/`num_ctx` interact: Ollama's
``/v1`` endpoint has no per-request way to raise the context window
(``num_ctx``) -- it's a server/Modelfile-only setting, and an over-budget
prompt is silently truncated rather than rejected
(github.com/ollama/ollama/issues/5356). ``/api/chat`` accepts
``options.num_ctx`` per request, so a stage that needs a large schema/prompt
in context can force it without touching the model's own Modelfile.

Uses stdlib ``urllib`` only -- no extra HTTP dependency, no SDK guard needed
(unlike the ``openai``/``anthropic`` providers, which lazily import their
SDKs). ``base_url`` may include a trailing ``/v1`` or not; either way this
provider talks to ``{root}/api/chat``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..errors import ProviderError
from ..logging_utils import get_logger
from ..types import LLMResponse, TokenUsage
from .base import LLMProvider, register_provider

logger = get_logger("providers.ollama")

_DEFAULT_BASE_URL = "http://localhost:11434"


@register_provider("ollama")
class OllamaProvider(LLMProvider):
    """Provider backed by Ollama's native ``/api/chat``."""

    def _root(self) -> str:
        base = (self._base_url or _DEFAULT_BASE_URL).rstrip("/")
        return base.removesuffix("/v1")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._root()}/api/chat"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"cannot reach {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"non-JSON response from {url}: {exc}") from exc

    def _options(self, *, temperature: float, top_p: float) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature, "top_p": top_p}
        if self._num_ctx:
            options["num_ctx"] = self._num_ctx
        return options

    @staticmethod
    def _usage(reply: dict[str, Any]) -> TokenUsage:
        if "prompt_eval_count" not in reply and "eval_count" not in reply:
            return TokenUsage()
        prompt = reply.get("prompt_eval_count") or 0
        completion = reply.get("eval_count") or 0
        return TokenUsage(
            prompt_tokens=prompt, completion_tokens=completion,
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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {**self._options(temperature=temperature, top_p=top_p),
                       "num_predict": max_tokens},
        }
        logger.debug("Ollama completion request (model=%s)", self.model)
        reply = self._post(payload)
        usage = self._usage(reply)
        logger.info(
            "Ollama prompt token count (model=%s, role=complete): %d",
            self.model, usage.prompt_tokens,
        )
        try:
            text = reply["message"]["content"] or ""
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected chat response shape: {str(reply)[:500]}") from exc
        return LLMResponse(text=text, usage=usage, model=self.model, raw=reply)

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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": json_schema,
            "options": {**self._options(temperature=temperature, top_p=top_p),
                       "num_predict": max_tokens},
        }
        logger.debug("Ollama structured request (model=%s)", self.model)
        try:
            reply = self._post(payload)
        except ProviderError as exc:
            # Older Ollama servers / some models reject the `format` field --
            # fall back to schema-in-prompt + text parsing (openai_provider's
            # same fallback for gateways that don't support json_schema mode).
            logger.warning(
                "Ollama structured output failed (%s). Falling back to text parsing.", exc,
            )
            return super().complete_json(
                system=system, user=user, json_schema=json_schema, schema_name=schema_name,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            )
        usage = self._usage(reply)
        logger.info(
            "Ollama prompt token count (model=%s, role=complete_json): %d",
            self.model, usage.prompt_tokens,
        )
        try:
            text = reply["message"]["content"] or ""
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected chat response shape: {str(reply)[:500]}") from exc
        return LLMResponse(text=text, usage=usage, model=self.model, raw=reply)
