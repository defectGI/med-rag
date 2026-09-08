"""retrieval.eval — golden-set format, metrics, and runners.

Cross-cutting evaluation infrastructure, not a retrieval method: it depends
only on ``retrieval.core`` and never imports ``retrieval.modules.*``. Every
module is scored through this harness against a user-supplied golden set.
"""

from medrag.api.retrieval.eval.golden import (
    IntentGoldenItem,
    RetrievalGoldenItem,
    load_intent_golden,
    load_retrieval_golden,
)
from medrag.api.retrieval.eval.metrics import (
    ClassScore,
    IntentReport,
    RetrievalReport,
    intent_metrics,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
)
from medrag.api.retrieval.eval.runner import (
    Classifier,
    evaluate_intent,
    evaluate_retrieval,
)

# RUF022: sorted alphabetically; grouping comments dropped (each name's source
# module is already clear from the import blocks above).
__all__ = [
    "ClassScore",
    "Classifier",
    "IntentGoldenItem",
    "IntentReport",
    "RetrievalGoldenItem",
    "RetrievalReport",
    "evaluate_intent",
    "evaluate_retrieval",
    "intent_metrics",
    "load_intent_golden",
    "load_retrieval_golden",
    "recall_at_k",
    "reciprocal_rank",
    "retrieval_metrics",
]
