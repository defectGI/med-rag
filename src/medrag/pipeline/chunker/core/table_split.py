"""Table splitter — splits a table that doesn't fit the hard ceiling with
k-row overlap.

The packer's relation to tables is two calls:

* `compose_table_text(table, cfg)` — UNSPLIT table's chunk text:
  (`inject_description` on) description on top + table render +
  (`inject_facts` on) all fact sentences below the table. The packer
  sizes from THIS text and leaves the table whole if it fits
  (measured == stored). `description` used to be injected only into
  split parts — revised (most tables in the corpus never split, so the
  VLM's only natural-language summary of the table was systematically
  left out).
* `split_table(table, cfg, tokenizer)` — splits the overflowing table
  into parts. The packer only calls this when compose's output overflows
  the effective limit; a single-part result means "couldn't be shrunk"
  (below).

Splitting rules
---------------
* Parts are packed to `target_tokens`, NOT the effective limit: the flex
  allowance exists to AVOID splitting a block; once we're splitting every
  row boundary is an equally good cut point — there's no reason to
  exceed target. Sole exception: a one-row part that exceeds target
  ("unsplittable part", below).
* Consecutive parts share `overlap_rows` rows; each part makes at least
  1 new row of progress (k is clipped to part size).
* When `repeat_header_rows` is on, header rows are repeated per part;
  off, they live only in the first part (later parts render with
  `header_rows=0`).
* When `[visual] inject_description` is on, description is prepended as
  a plain paragraph to EVERY part. This flag lives in `[visual]`, not
  `[table_split]` (same flag covers `Image.description` — a table is not
  a separate category; see `core/document.py` module docstring).
* When `inject_facts` is on, facts are matched to rows (below) and each
  fact is written into the part where its row FIRST appears (overlap
  repetition is not accompanied by fact repetition); table-wide facts
  (matching no row) go into EVERY part. Part size is measured from the
  final render: description + table + facts.

Fact→row matching heuristic
---------------------------
The parser gives facts as sentences without row binding. Heuristic: the
total length of body-row cell values that appear (casefold) in the fact
text is the row's score (single-character cells are ignored as noise);
the highest-scoring row wins; ties go to the first row. If no cell
matches, the fact is treated as table-wide. Header rows do not
contribute to the score: header names appear in almost every fact, body
values are what discriminates.

Unsplittable part
-----------------
Even a one-new-row part can exceed target (long row; header +
description + general-fact load too). The row is NOT sub-split — the
part goes out as-is with an honest `token_count`; the packer records /
flags the overflow (the only known exception to "the hard ceiling is
absolute" in the README).

Complexity note: the grow loop re-renders and re-counts each candidate
(O(n·part) renders). Tables are small in practice; the "measure from
final text" principle wins over simplicity.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from medrag.pipeline.chunker.config import ChunkerConfig
from medrag.pipeline.chunker.core.document import Table
from medrag.pipeline.chunker.core.markdown import _describable_placeholder, render_table
from medrag.pipeline.chunker.core.split_common import Span, pack_spans
from medrag.pipeline.chunker.tokenization.base import Tokenizer

# Min cell-value length that contributes to the fact-matching score;
# single characters ("4", "-") appear in almost every fact, treated as noise.
_MIN_CELL_MATCH_LEN = 2


class TablePart(BaseModel):
    """One part of `split_table` output.

    table
        The part's table (header repetition included); `id` is derived
        from the original as `{id}#p{n}` (the adapter's `L1#2` convention),
        other fields (heading_path, page, provenance...) are copies.
    text
        Final chunk text: description + table render + facts.
    token_count
        Tokenizer count of `text` — the packer derives its flex/over-limit
        diagnostic from this.
    overlap_units
        Rows shared with the previous part (0 for the first part). The
        ACTUALLY applied overlap is written (k can be clipped to part size).
    """

    model_config = ConfigDict(extra="forbid")

    table: Table
    text: str
    token_count: int = Field(ge=0)
    split_index: int = Field(ge=1)   # 1-based
    split_total: int = Field(ge=1)
    overlap_units: int = Field(ge=0)


def compose_table_text(table: Table, cfg: ChunkerConfig) -> str:
    """The unsplit table's chunk text; the packer sizes from this.

    When `inject_description` is on, the table description is included
    (revision — previously it was injected only into split parts, which
    systematically dropped the VLM's natural-language summary for the
    common case of tables that never split — the most valuable part for
    embedding/search). `description` is now subject to the same config
    flag regardless of split/unsplit.

    `table.excluded_at_parse` (parse-time `[visual] exclude_types` had
    "table" added) OR `"table" in [visual] exclude_types` (chunk-time,
    config-only, reversible without re-parse) means no table render —
    instead a placeholder, from the SAME helper used for images (a
    table is not a separate branch of this category; see
    `core/markdown.py` module docstring). The placeholder is single-
    line, so `split_table` is never reached (the packer already sees it
    as small) — the splitting engine leaves it untouched in that case.
    """
    if table.excluded_at_parse or "table" in cfg.visual.exclude_types:
        return _table_placeholder(table)
    parts = []
    if cfg.visual.inject_description and table.description:
        parts.append(table.description)
    parts.append(render_table(table))
    if cfg.table_split.inject_facts and table.facts:
        parts.append("\n".join(table.facts))
    return "\n\n".join(p for p in parts if p)


def _table_placeholder(table: Table) -> str:
    # `Table` has no `visual_type` field of its own — being a table IS
    # the type, so the literal "table" is passed (see parser IR's
    # DescribableBlock).
    return _describable_placeholder(entity="Blok", block_id=table.id,
                                    visual_type="table", ref=table.source_crop)


def _match_facts(
    facts: list[str], body: list[list[str]],
) -> tuple[list[str], dict[int, list[str]]]:
    """Match facts to body rows: (table-wide, row_idx -> facts)."""
    general: list[str] = []
    by_row: dict[int, list[str]] = {}
    for fact in facts:
        f = fact.casefold()
        best_row, best_score = None, 0
        for i, row in enumerate(body):
            score = sum(
                len(cell) for cell in (c.strip() for c in row)
                if len(cell) >= _MIN_CELL_MATCH_LEN and cell.casefold() in f)
            if score > best_score:
                best_row, best_score = i, score
        if best_row is None:
            general.append(fact)
        else:
            by_row.setdefault(best_row, []).append(fact)
    return general, by_row


def split_table(
    table: Table, cfg: ChunkerConfig, tokenizer: Tokenizer,
) -> list[TablePart]:
    """Split the table into parts packed to `target_tokens` with k-row overlap.

    Always returns at least one part; a header-only or one-row table
    returns one part which may exceed target (module docstring:
    "unsplittable part").
    """
    ts = cfg.table_split
    target = cfg.limits.target_tokens
    n_header = min(table.header_rows, len(table.cells))
    header = table.cells[:n_header]
    body = table.cells[n_header:]

    if ts.inject_facts:
        general_facts, row_facts = _match_facts(table.facts, body)
    else:
        general_facts, row_facts = [], {}
    description = table.description if cfg.visual.inject_description else None

    def part_table(index: int, rows: list[list[str]], with_header: bool) -> Table:
        return table.model_copy(update={
            "id": f"{table.id}#p{index}",
            "cells": (header if with_header else []) + rows,
            "header_rows": n_header if with_header else 0,
        })

    def part_text(pt: Table, new_start: int, new_end: int) -> str:
        facts = list(general_facts)
        for i in range(new_start, new_end):
            facts.extend(row_facts.get(i, []))
        pieces = [description or "", render_table(pt), "\n".join(facts)]
        return "\n\n".join(p for p in pieces if p)

    if not body:
        # No body to split: table as-is, one part (couldn't be shrunk).
        pt = part_table(1, [], with_header=True)
        text = part_text(pt, 0, 0)
        return [TablePart(table=pt, text=text, token_count=tokenizer.count(text),
                          split_index=1, split_total=1, overlap_units=0)]

    def fits(span: Span) -> bool:
        with_header = ts.repeat_header_rows or span.new_start == 0
        aday = part_text(
            part_table(0, body[span.start:span.end], with_header),
            span.new_start, span.end)
        return tokenizer.count(aday) <= target

    spans = pack_spans(len(body), ts.overlap_rows, fits)

    parts: list[TablePart] = []
    for i, span in enumerate(spans, start=1):
        with_header = ts.repeat_header_rows or i == 1
        pt = part_table(i, body[span.start:span.end], with_header)
        text = part_text(pt, span.new_start, span.end)
        parts.append(TablePart(
            table=pt, text=text, token_count=tokenizer.count(text),
            split_index=i, split_total=len(spans),
            overlap_units=span.overlap))
    return parts