"""Shared LLM/embedding provider error type.

Was defined separately in `urun/api/answering_model.py`,
`urun/api/retrieval/modules/top_n/embedder.py` and
`urun/pipeline/vectorize/embedder/openai_compat.py` -- three byte-identical
`class ProviderError(RuntimeError)` bodies. Consolidated here.

NOT consolidated (deliberately out of scope):
  - `packages/text2sql-native/text2sql_native/errors.py::ProviderError`
    subclasses `Text2SQLError`, not `RuntimeError` -- a different hierarchy,
    own package (it was moved out of chatbot/vendor without touching
    behavior).
  - `urun/pipeline/parser/llm/base.py::LLMError` -- a different name and
    hierarchy entirely (`Exception`, not `RuntimeError`), 15+ call sites;
    merging it is a separate decision, not this one's.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Provider configuration or call failed."""
