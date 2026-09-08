"""intent_classification — query -> intent label (v1: LLM prompt based).

Public API. The module returns only a ``core.IntentResult``; mapping a label to
a retrieval strategy is the consuming project's job (ARCHITECTURE.md #7).

Typical use::

    from medrag.api.retrieval.modules.intent_classification import (
        LLMIntentClassifier, build_default_chat_model,
    )

    clf = LLMIntentClassifier(build_default_chat_model())   # on the inference machine
    result = await clf.classify("de1000 fiyati ne kadar")   # -> IntentResult

In tests, inject a fake ``BaseChatModel`` instead of ``build_default_chat_model``.
"""

from medrag.api.retrieval.modules.intent_classification.classifier import (
    IntentClassifier,
    LLMIntentClassifier,
    classify_sync,
)
from medrag.api.retrieval.modules.intent_classification.factory import (
    build_default_chat_model,
)
from medrag.api.retrieval.modules.intent_classification.parsing import (
    confidence_from_logprobs,
    parse_label,
)
from medrag.api.retrieval.modules.intent_classification.prompt import (
    build_system_prompt,
    intent_catalogue_block,
    labels_list,
)

__all__ = [
    "IntentClassifier",
    "LLMIntentClassifier",
    "build_default_chat_model",
    "build_system_prompt",
    "classify_sync",
    "confidence_from_logprobs",
    "intent_catalogue_block",
    "labels_list",
    "parse_label",
]
