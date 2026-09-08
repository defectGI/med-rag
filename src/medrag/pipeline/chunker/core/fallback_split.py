"""Fallback splitters — paragraph/code/list when the hard ceiling is still
overflowed after flex.

Over-sized non-table blocks are split at their NATURAL boundary too —
paragraph by sentence, code by line (fence + language repeated per part),
list by top-level item (nested children travel with their parent item;
item groups are not split). Overlap comes from config: `[fallback_split]`
— paragraph default 2 sentences, code/list 0 (overlap is misleading in
code/list).

The table splitter shares the same principles (see `table_split.py`):
* Parts are packed to `target_tokens` (the flex allowance exists to
  AVOID splitting; once we're splitting there's no reason to exceed
  target).
* Part size is measured from the FINAL rendered text (code: fence included).
* Even a single-unit part can exceed target ("unsplittable part" — long
  sentence/line/item); the part goes out as-is with an honest
  `token_count` and is not sub-split.
* The part's block id is derived as `{id}#p{n}`; source traceability
  via the original block id is the packer's job.

The sentence-boundary heuristic is deliberately simple: `.`/`!`/`?`/`…`
followed by whitespace. Abbreviations ("e.g.", "Dr.") may be cut wrong —
splitting is already the last resort and wrong cuts soften with overlap;
not worth pulling in an NLP dependency.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from medrag.pipeline.chunker.config import ChunkerConfig
from medrag.pipeline.chunker.core.document import Code, ListBlock, ListItem, Paragraph
from medrag.pipeline.chunker.core.markdown import render_code, render_list
from medrag.pipeline.chunker.core.split_common import Span, pack_spans
from medrag.pipeline.chunker.tokenization.base import Tokenizer

# Sentence boundary: punctuation + whitespace. Punctuation stays in the sentence.
_CUMLE_AYRACI = re.compile(r"(?<=[.!?…])\s+")


class BlockPart(BaseModel):
    """Shared part output of the fallback splitters (sibling of
    `table_split.TablePart`). The unit of `overlap_units` depends on
    block type: paragraph=sentences, code=lines, list=top-level items;
    the overlap actually applied is what's written."""

    model_config = ConfigDict(extra="forbid")

    block: Annotated[Paragraph | ListBlock | Code,
                     Field(discriminator="kind")]
    text: str
    token_count: int = Field(ge=0)
    split_index: int = Field(ge=1)   # 1-based
    split_total: int = Field(ge=1)
    overlap_units: int = Field(ge=0)


def _parcala(
    n_units: int,
    overlap: int,
    build: Callable[[int, Span], Paragraph | ListBlock | Code],
    render: Callable[[Paragraph | ListBlock | Code], str],
    cfg: ChunkerConfig,
    tokenizer: Tokenizer,
) -> list[BlockPart]:
    """Common skeleton: pack spans, produce block + final text per span."""
    target = cfg.limits.target_tokens

    def fits(span: Span) -> bool:
        return tokenizer.count(render(build(0, span))) <= target

    spans = pack_spans(n_units, overlap, fits)
    parts: list[BlockPart] = []
    for i, span in enumerate(spans, start=1):
        blok = build(i, span)
        text = render(blok)
        parts.append(BlockPart(
            block=blok, text=text, token_count=tokenizer.count(text),
            split_index=i, split_total=len(spans), overlap_units=span.overlap))
    return parts


def sentences(text: str) -> list[str]:
    """Split text by the sentence-boundary heuristic (module docstring)."""
    return [c for c in _CUMLE_AYRACI.split(text.strip()) if c]


def split_paragraph(
    block: Paragraph, cfg: ChunkerConfig, tokenizer: Tokenizer,
) -> list[BlockPart]:
    cumleler = sentences(block.text)

    def build(i: int, span: Span) -> Paragraph:
        return block.model_copy(update={
            "id": f"{block.id}#p{i}",
            "text": " ".join(cumleler[span.start:span.end])})

    return _parcala(len(cumleler), cfg.fallback_split.paragraph_overlap_sentences,
            build, lambda b: b.text, cfg, tokenizer)


def split_code(
    block: Code, cfg: ChunkerConfig, tokenizer: Tokenizer,
) -> list[BlockPart]:
    satirlar = block.text.rstrip("\n").split("\n")

    def build(i: int, span: Span) -> Code:
        return block.model_copy(update={
            "id": f"{block.id}#p{i}",
            "text": "\n".join(satirlar[span.start:span.end])})

    # Size measured from the render: fence + language tag are part of the
    # budget per part.
    return _parcala(len(satirlar), cfg.fallback_split.code_overlap_lines,
                    build, render_code, cfg, tokenizer)


def _ust_seviye_gruplar(items: list[ListItem]) -> list[list[ListItem]]:
    """Group items into unsplittable groups: a top-level (level=0) item
    plus any deeper items that follow it stay together. If the data
    starts with level>0, the leading run is its own group."""
    gruplar: list[list[ListItem]] = []
    for item in items:
        if item.level == 0 or not gruplar:
            gruplar.append([item])
        else:
            gruplar[-1].append(item)
    return gruplar


def split_list(
    block: ListBlock, cfg: ChunkerConfig, tokenizer: Tokenizer,
) -> list[BlockPart]:
    gruplar = _ust_seviye_gruplar(block.items)

    def build(i: int, span: Span) -> ListBlock:
        return block.model_copy(update={
            "id": f"{block.id}#p{i}",
            "items": [it for grup in gruplar[span.start:span.end] for it in grup]})

    return _parcala(len(gruplar), cfg.fallback_split.list_overlap_items,
                    build, render_list, cfg, tokenizer)