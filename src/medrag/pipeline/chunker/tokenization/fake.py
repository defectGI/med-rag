"""Deterministic fake tokenizer — for tests and offline end-to-end runs.

The counting rule is deliberately the simplest predictable one: count of
whitespace-separated pieces (words). This lets splitting-engine tests set
up text at the exact token count they want ("a b c" = 3 tokens) without
guessing what a real BPE would produce. No network, no model files, no
cache needed.

Not intended for production; `TOKENIZER=fake` is for test/offline
scenarios only (see `factory.py`).
"""

from __future__ import annotations


class FakeTokenizer:
    """Deterministic counter: 1 word = 1 token.

    `str.split()` semantics: any whitespace (space, tab, newline) is a
    separator; consecutive separators count as one; empty / whitespace-only
    text is 0 tokens.
    """

    name = "fake"

    def count(self, text: str) -> int:
        return len(text.split())