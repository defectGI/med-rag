"""Tokenizer abstraction: `core/` counts tokens ONLY through this interface.

The interface is deliberately a single method (`count(text) -> int`): the
engine's only requirement from a tokenizer is size measurement.
`encode`/`decode` are not exposed — chunk text is carried as text, not
as token ids, so the engine has no need for them, and keeping the
interface narrow makes new backends (HuggingFace, model-specific
counters...) cheap to add.

`Protocol` was chosen (not ABC): same plugability rationale as
parser-agnosticism — ANY object providing `count` and `name` is a
tokenizer, no inheritance from this package required. `name` is for
diagnostics (log/output metadata); it doesn't affect counting.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Token counter contract.

    - `count` must be deterministic: the same text must yield the same
      count on every call (splitting decisions stay reproducible).
    - Empty text must return 0.
    """

    name: str
    """Diagnostic label, e.g. ``"tiktoken:cl100k_base"`` or ``"fake"``."""

    def count(self, text: str) -> int:
        """Token count of `text` (>= 0)."""
        ...