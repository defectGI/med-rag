"""Surrounding-document context for a describable block.

Every describable block type needs the same thing: a block sitting in a
document is rarely self-explanatory on its own. A spec table's units live in
the paragraph above it; a block diagram's subject is named by the sentence
that introduces it; a technical drawing's part number is in its section
heading. So the model gets the block PLUS its neighborhood.

This used to live inside `tables/table_describe.py`, which made it look like a
table-specific feature (config even called it `[table.context]`). It never was:
nothing here knows what a table is. The input is the block list and an index,
and the output is text — identical for a table, a chart or a flowchart. It now
lives here, once, and every strategy in `describe/` uses it.

Why gathering runs AFTER the whole document is parsed: a classification call
sees one cropped region and nothing else. Context only exists once the block's
neighbors exist, so the describe pass is a separate pass over the finished IR,
not something folded into parse-time or classification-time work.
"""

from __future__ import annotations

from medrag.pipeline.parser.parsers.base import HeadingBlock, ParagraphBlock

__all__ = ["build_context"]


def _heading_breadcrumb(blocks: list, idx: int) -> str:
    """Ancestor heading chain preceding the block at `idx`, e.g. 'A > B > C'."""
    chain: list[str] = []
    needed_level: int | None = None
    for block in reversed(blocks[:idx]):
        if isinstance(block, HeadingBlock) and (
                needed_level is None or block.level < needed_level):
            chain.append(block.text.strip())
            needed_level = block.level
            if block.level == 1:
                break
    chain.reverse()
    return " > ".join(t for t in chain if t)


def _collect_paragraphs(blocks: list, indices, count: int, max_chars: int) -> list[str]:
    """Up to `count` non-empty paragraph texts in the order `indices` is walked."""
    out: list[str] = []
    if count <= 0:
        return out
    for i in indices:
        block = blocks[i]
        if isinstance(block, ParagraphBlock) and block.text.strip():
            out.append(block.text.strip()[:max_chars])
            if len(out) >= count:
                break
    return out


def build_context(blocks: list, idx: int, max_chars: int,
                  before: int = 1, after: int = 1) -> str:
    """Heading breadcrumb + up to `before`/`after` surrounding paragraphs.

    Paragraphs are always listed in document order (nearest ones included first
    when the budget is small).
    """
    parts = []
    breadcrumb = _heading_breadcrumb(blocks, idx)
    if breadcrumb:
        parts.append(f"Section: {breadcrumb}")

    # Walk outward from the block, then restore document order for readability.
    prev = _collect_paragraphs(blocks, range(idx - 1, -1, -1), before, max_chars)
    for text in reversed(prev):
        parts.append(f"Text before: {text}")

    nxt = _collect_paragraphs(blocks, range(idx + 1, len(blocks)), after, max_chars)
    for text in nxt:
        parts.append(f"Text after: {text}")

    return "\n".join(parts)
