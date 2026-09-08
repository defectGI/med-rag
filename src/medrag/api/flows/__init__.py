"""Intent-routed retrieval paths ("flows"). Each flow owns everything
between an already-determined intent and the final merged context handed to
the Orchestrator -- see `base.py` for the interface to implement.

Checklist when adding a new flow (equivalent of retrieval/ARCHITECTURE.md's
per-module checklist):
- one file here implementing `Flow`
- registered under a name in the wiring code that builds the `Router`
- `config/default.toml` [routing] points at least one intent to it (or leaves
  it as the `default_flow` fallback)
- its own tests under `tests/`
"""

from medrag.api.flows.aggregation import AggregationFlow, AggregationSplitter
from medrag.api.flows.base import (
    Flow,
    FlowContext,
    PinAwareRunFlow,
    PinnableFlow,
    ResultShape,
    SelfPinningFlow,
    compute_result_shape,
)
from medrag.api.flows.comparison import (
    ComparisonAngleSplitter,
    ComparisonFlow,
    ComparisonSplitter,
    DeterministicComparisonSplitter,
)
from medrag.api.flows.default_topn import DefaultTopNFlow
from medrag.api.flows.doc_download import (
    DocDownloadFlow,
    DocumentLookup,
    DocumentRow,
    SqliteDocumentLookup,
)
from medrag.api.flows.no_retrieval import NoRetrievalFlow
from medrag.api.flows.recommendation import ClarifyingSlot, RecommendationFlow
from medrag.api.flows.sql_topn import KeywordExtractor, Rewriter, SqlTopNFlow

__all__ = [
    "AggregationFlow",
    "AggregationSplitter",
    "ClarifyingSlot",
    "ComparisonAngleSplitter",
    "ComparisonFlow",
    "ComparisonSplitter",
    "DefaultTopNFlow",
    "DeterministicComparisonSplitter",
    "DocDownloadFlow",
    "DocumentLookup",
    "DocumentRow",
    "Flow",
    "FlowContext",
    "KeywordExtractor",
    "NoRetrievalFlow",
    "PinAwareRunFlow",
    "PinnableFlow",
    "RecommendationFlow",
    "ResultShape",
    "Rewriter",
    "SelfPinningFlow",
    "SqlTopNFlow",
    "SqliteDocumentLookup",
    "compute_result_shape",
]
