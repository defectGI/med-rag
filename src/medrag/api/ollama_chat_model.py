"""Native-Ollama `BaseChatModel` -- forces `options.num_ctx` per request.

Only exists so a langchain-based consumer that only accepts a `BaseChatModel`
(currently `retrieval.modules.intent_classification.LLMIntentClassifier`,
see `retrieval/AGENTS.md`'s provider-agnostic rule -- that repo will never
grow Ollama-specific code) can still get the same native `/api/chat` +
`options.num_ctx` guarantee this component's other Ollama-backed roles have
(`answering_model.py`/`reconciler.py`/`continuation.py`/`slot_selection.py`).
Ollama's OpenAI-compatible `/v1` endpoint has no per-request way to raise
`num_ctx` (github.com/ollama/ollama/issues/5356); an over-budget prompt is
silently truncated rather than rejected, and -- the failure mode this class
exists to fix -- a role sharing weights with another `num_ctx`-forcing role
never converges on one context size, so Ollama reloads the model on every
call that alternates between them (cold-start thrash).

Opt-in only (`INTENT_PROVIDER=ollama` in `chatbot/.env`, see
`factory.py::_build_intent_chat_model`) -- unset, the classifier keeps using
retrieval's own `build_default_chat_model()` (plain `/v1`, no `num_ctx`),
zero behavior change for deployments that don't need this.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from medrag.api.answering_model import ProviderError

_ROLE_TO_OLLAMA = {"human": "user", "ai": "assistant", "system": "system", "tool": "user"}


def _to_ollama_message(message: BaseMessage) -> dict[str, str]:
    return {
        "role": _ROLE_TO_OLLAMA.get(message.type, "user"),
        "content": str(message.content),
    }


class NativeOllamaChatModel(BaseChatModel):
    """Talks to Ollama's native `/api/chat`, forcing `options.num_ctx`.

    Mirrors the raw-HTTP native-Ollama path in `reconciler.py`/
    `continuation.py`, wrapped in langchain-core's `BaseChatModel` interface
    so it plugs into any langchain-based consumer without that consumer
    needing to know about Ollama at all.
    """

    ollama_base_url: str
    ollama_model: str
    num_ctx: int
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "native-ollama"

    def _root(self) -> str:
        return self.ollama_base_url.rstrip("/").removesuffix("/v1")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.ollama_model,
            "messages": [_to_ollama_message(m) for m in messages],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0},
        }
        url = f"{self._root()}/api/chat"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"cannot reach {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"non-JSON response from {url}: {exc}") from exc
        try:
            text = reply["message"]["content"] or ""
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected chat response shape: {str(reply)[:500]}") from exc
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])


__all__ = ["NativeOllamaChatModel"]
