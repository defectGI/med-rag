"""Core interfaces shared by every retrieval module.

Only *shapes* live here — no implementation, no module-specific logic
(ARCHITECTURE.md #2). Every module depends on this; this depends on no module.

Retrieval is async-first: every real backend in this layer is network I/O
(remote OpenAI-compatible embedding/LLM endpoints, vector stores), and later
modules (query_rewriting multi-query, agentic multi-step) fan out concurrent
calls that only pay off under async. ``run_sync`` is a thin convenience for
callers in a plain synchronous context.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Retrieval result + retriever protocol
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """A single scored item returned by any retriever.

    Deliberately generic so every backend (top_n, raptor, db_query, agentic)
    returns the same type. ``id`` is the backend's identifier for the item
    (e.g. a ``ChunkNode.node_id`` or a product ``node_id``); ``metadata`` holds
    whatever the backend wants to surface for citation/display without forcing
    it into the shared shape.
    """

    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


@runtime_checkable
class Retriever(Protocol):
    """The one method every retrieval backend implements.

    Async by contract. ``k`` is the maximum number of results to return;
    implementations may return fewer.
    """

    async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]: ...


def run_sync(retriever: Retriever, query: str, k: int = 10) -> list[RetrievalResult]:
    """Call an async :class:`Retriever` from synchronous code.

    Convenience only — prefer awaiting ``retriever.retrieve(...)`` directly in
    async contexts. Raises ``RuntimeError`` if called while an event loop is
    already running (use ``await`` there instead).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — safe to start one
    else:
        raise RuntimeError(
            "run_sync() cannot be called from a running event loop; "
            "await retriever.retrieve(...) directly instead."
        )
    return asyncio.run(retriever.retrieve(query, k))


# ---------------------------------------------------------------------------
# Intent classification result
# ---------------------------------------------------------------------------


class IntentLabel(str, Enum):
    """Intent set v1 (9 classes).

    The intent_classification module returns one of these and nothing more —
    the label -> retrieval-method mapping is the consuming project's job
    (ARCHITECTURE.md #7).
    """

    PRODUCT_FACT = "product_fact"
    AGGREGATION = "aggregation"
    DOC_QUESTION = "doc_question"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    VISUAL_REQUEST = "visual_request"
    DOC_DOWNLOAD = "doc_download"
    QUOTE_OR_CONTACT = "quote_or_contact"
    OUT_OF_SCOPE = "out_of_scope"


class IntentResult(BaseModel):
    """Output of an intent classifier for a single query."""

    label: IntentLabel
    confidence: float | None = None
    rationale: str | None = None
    # Set ONLY when the label is a silent fallback -- the model's raw text
    # didn't match any known label and `out_of_scope` was substituted. Lets a
    # caller tell "the model genuinely said out_of_scope" apart from "the model
    # said something unparseable and it defaulted" for tracing/observability,
    # without changing behavior for existing callers.
    raw_text: str | None = None


__all__ = [
    "IntentLabel",
    "IntentResult",
    "RetrievalResult",
    "Retriever",
    "run_sync",
]
