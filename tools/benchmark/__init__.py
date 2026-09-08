"""Parser-quality benchmark.

Scores an automated parser's Markdown output against its source PDF on
three dimensions -- accuracy, coverage, clarity -- with an optional
scope, using a provider-agnostic vision LLM as judge.

Run it with:  python -m benchmark -f <folder> [-s scope.md]
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv=None) -> int:
    from .cli import main as _main

    return _main(argv)
