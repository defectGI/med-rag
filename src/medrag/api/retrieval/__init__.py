"""mkd-retriever — a modular, project-agnostic RAG retrieval layer.

Independent retrieval strategies (intent classification, text-to-SQL db query,
and more to come) live side by side without depending on one another. They share
only a small, stable ``retrieval.core`` (interfaces + ingestion-contract
schemas); each strategy is an installable extra so a consumer pulls in only the
dependencies it uses. See ``ARCHITECTURE.md`` for the hard rules.

Importing this top-level package pulls in only ``retrieval.core`` (base
dependencies). Concrete strategies are imported from their own subpackages,
e.g. ``retrieval.modules.intent_classification`` or ``retrieval.modules.db_query``,
and require that module's extra to be installed.
"""

from __future__ import annotations

from medrag.api.retrieval.core import (
    IntentLabel,
    IntentResult,
    RetrievalResult,
    Retriever,
    run_sync,
)

__version__ = "0.4.1"

__all__ = [
    "IntentLabel",
    "IntentResult",
    "RetrievalResult",
    "Retriever",
    "__version__",
    "run_sync",
]
