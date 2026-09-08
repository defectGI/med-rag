"""Intent classifier: query in, ``IntentResult`` out. Nothing more.

Per ARCHITECTURE.md #7 this module ONLY assigns a label — it does not know or
decide what retrieval strategy an intent maps to. That mapping, and each
intent's strategy, live entirely outside this module.

The classifier depends only on the ``langchain-core`` ``BaseChatModel``
abstraction, never on a concrete provider. That keeps it fully testable with a
fake chat model (no network, no GPU here) while production wires in a real
OpenAI-compatible endpoint via :func:`factory.build_default_chat_model`.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from medrag.api.retrieval.core import IntentResult
from medrag.api.retrieval.modules.intent_classification.parsing import (
    confidence_from_logprobs,
    parse_label_verbose,
)
from medrag.api.retrieval.modules.intent_classification.prompt import (
    build_system_prompt,
)


@runtime_checkable
class IntentClassifier(Protocol):
    """The seam the module exposes.

    v1 is LLM-prompt based; future backends (embedding-similarity, a small
    fine-tuned model) implement this same protocol so consuming code does not
    change when the backend does.
    """

    async def classify(self, query: str) -> IntentResult: ...


class LLMIntentClassifier:
    """Prompt-based classifier over any ``BaseChatModel``."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._model = chat_model
        self._system = system_prompt or build_system_prompt()

    def _messages(self, query: str) -> list:
        return [SystemMessage(content=self._system), HumanMessage(content=query)]

    def _to_result(self, ai_message: BaseMessage) -> IntentResult:
        content = ai_message.content
        if isinstance(content, list):  # some models return content blocks
            content = " ".join(str(c) for c in content)
        content = str(content)
        label, was_fallback = parse_label_verbose(content)
        confidence = confidence_from_logprobs(getattr(ai_message, "response_metadata", None))
        return IntentResult(
            label=label, confidence=confidence,
            raw_text=content if was_fallback else None,
        )

    async def classify(self, query: str) -> IntentResult:
        ai_message = await self._model.ainvoke(self._messages(query))
        return self._to_result(ai_message)


def classify_sync(classifier: IntentClassifier, query: str) -> IntentResult:
    """Call an async :class:`IntentClassifier` from synchronous code.

    The sync mirror of :func:`retrieval.core.run_sync` for retrievers: the
    module keeps a single async protocol surface (just ``classify``) and offers
    this free helper alongside it, so the sync story is identical across the
    package. Prefer awaiting ``classifier.classify(...)`` directly in async
    contexts. Raises ``RuntimeError`` if called while an event loop is already
    running (use ``await`` there instead).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — safe to start one
    else:
        raise RuntimeError(
            "classify_sync() cannot be called from a running event loop; "
            "await classifier.classify(...) directly instead."
        )
    return asyncio.run(classifier.classify(query))


__all__ = ["IntentClassifier", "LLMIntentClassifier", "classify_sync"]
