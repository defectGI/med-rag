"""The TABLE strategy for producing a describable block's description.

One of three strategies behind the shared describe pass (`describe/core.py`);
the others are `describe/visual.py` (VLM reads a crop) and the parsers' own
structural extraction (chart/SmartArt XML, no model at all). What makes a table
different from a chart or a block diagram is NOT its category -- it is that a
table carries real cell data, so its description can be written from the cells
themselves rather than guessed from pixels, and its facts can be fidelity-checked
against those cells. That extra determinism is why this strategy alone produces
`facts` and can run a verify+retry loop.

Per table:
    1. render the structured table to plain text (merges noted),
    2. optionally gather nearby context (`describe/context.py`) when
       [describe.context] is on,
    3. ask the model for a short, plain-text description,
    4. optionally verify content + format with a second LLM call and retry up
       to [table] check_retries times, feeding the rejection reason back.

The description and check both reference the SAME canonical FORMAT_SPEC so the
writer and the checker agree on what "correct" means. Everything model- and
provider-specific lives behind `llm.LLMClient`.

Alongside the description, every table also gets `facts`: the table's content
restated as self-contained natural-language sentences, one per item the table
describes, each weaving that item's whole row of attribute values into a single
cartesian sentence ("Product X withstands temperatures from 20 to 50 C, weighs
10 kg and is made of aluminium.") so a retriever can answer about one item
without reconstructing the grid. For downstream retrieval/QA. JSON-only --
render/markdown.py deliberately doesn't emit it. Sentences are checked
deterministically, not by a second LLM call: every digit run ("150301", "40",
"5") in a sentence must literally occur in the table or its gathered context,
so a sentence with an invented number is dropped rather than stored. This gate
is exactly what a pixels-only visual can't offer -- hence no facts there.

Tuning lives in `config/default.toml`: `[table]` for what is genuinely
table-specific (max_rows, llm_check, facts, header_llm, check_retries) and
`[describe]`/`[describe.context]` for what every describable block shares
(concurrency, context window). Every knob keeps its historical env override
name (see `config.py`'s `_ENV_OVERRIDES`).
"""

from __future__ import annotations

import json
import re

from medrag.pipeline.parser.llm import LLMClient
from medrag.pipeline.parser.parsers.base import TableBlock, TableData

__all__ = ["FORMAT_SPEC", "describe_table_block", "render_table"]

# The single source of truth for what a good description looks like. Injected
# verbatim into both the writer prompt and the checker prompt.
FORMAT_SPEC = (
    "The description must be:\n"
    "- Plain text only: no markdown, no line breaks, no bullet points, no code.\n"
    "- Written in the same language as the table and its context.\n"
    "- 1 to 3 sentences, at most about 60 words.\n"
    "- A summary of what the table is about and its main dimensions "
    "(what the rows and columns represent).\n"
    "- Faithful: never invent numbers, totals or facts that are not in the table, "
    "and do not enumerate every cell."
)

_DESCRIBE_SYSTEM = (
    "You write concise descriptions of tables so they can be found by search. "
    "Output only the description itself, with no preamble or quotes.\n" + FORMAT_SPEC
)

_CHECK_SYSTEM = (
    "You verify a candidate description of a table. Judge two things "
    "independently: content (is it faithful to the table and does it correctly "
    "say what the table is about?) and format (does it obey the rules below?).\n"
    + FORMAT_SPEC
    + '\n\nReturn ONLY a JSON object, no other text: '
    '{"content_ok": true/false, "format_ok": true/false, "reason": "<short reason if anything is false>"}.'
)

_FACTS_SYSTEM = (
    "You restate a table's content as natural-language facts so they can be "
    "used to answer questions without seeing the table. Output one "
    "declarative sentence per line, nothing else.\n"
    "- First identify the axis that lists the distinct items the table is "
    "about (each row is usually one item -- a product, part, pin, model -- but "
    "if the table is transposed, with one column per item and one row per "
    "attribute, treat each such column as the item instead).\n"
    "- Write exactly ONE sentence per item, and weave ALL of that item's "
    "attribute values from the other columns into that single sentence "
    "(e.g. 'Product X withstands temperatures from 20 to 50 C, weighs 10 kg "
    "and is made of aluminium.'). Do not split one item across several "
    "sentences and do not write one sentence per attribute.\n"
    "- Each sentence must be self-contained: name its subject (the product, "
    "part, pin, ... it is about) instead of saying 'this table' or 'the row'.\n"
    "- Cover every item; skip a header or unit row that names no item.\n"
    "- Write in the same language as the table and its context.\n"
    "- Plain text only: no markdown, no bullets, no numbering, no preamble.\n"
    "- Be strictly faithful: never invent numbers, units or facts that are "
    "not in the table or its context."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_DIGIT_RUN = re.compile(r"\d+")


def _digit_runs(text: str) -> set[str]:
    """Maximal digit sequences in `text` ("0°C-+40°C" -> {"0", "40"}) -- the
    unit of the facts fidelity check, robust to the LLM re-punctuating or
    re-spacing a value ("0 °C to +40 °C") since only the digits themselves
    have to reoccur."""
    return set(_DIGIT_RUN.findall(text))


def render_table(table: TableData, max_rows: int | None = None) -> str:
    """Render a TableData to a compact pipe-delimited text block.

    Covered (merged-away) cells render empty; merge regions are listed after the
    grid so the model can reason about spans without us duplicating content.

    `max_rows` (when > 0) caps how many rows are rendered: a huge table would
    otherwise overflow the model's context window, and Ollama truncates an
    over-budget prompt silently, so the model would see an arbitrary prefix
    without knowing it. A capped render instead states the cut explicitly so
    the model never mistakes the prefix for the whole table; merge notes for
    rows beyond the cap are dropped with them.
    """
    rows = table.cells
    total = len(rows)
    truncated = max_rows is not None and 0 < max_rows < total
    if truncated:
        rows = rows[:max_rows]
    lines = []
    for row in rows:
        lines.append(" | ".join(
            "" if cell is None else cell.plain_text() for cell in row))
    text = "\n".join(lines)
    merges = ([m for m in table.merges if m.row < max_rows]
              if truncated else table.merges)
    if merges:
        spans = "; ".join(
            f"cell (row {m.row}, col {m.col}) spans {m.rowspan}x{m.colspan}"
            for m in merges
        )
        text += f"\n[merged: {spans}]"
    if truncated:
        text += (f"\n[table truncated: only the first {max_rows} of {total} "
                 f"rows are shown above]")
    return text


def _describe(client: LLMClient, table_text: str, context: str, hint: str) -> str:
    user = ""
    if context:
        user += f"Context around the table:\n{context}\n\n"
    user += f"Table:\n{table_text}\n\nWrite the description now."
    if hint:
        user += (
            f"\n\nA previous attempt was rejected for this reason: {hint}\n"
            "Write a new description that fixes it."
        )
    return client.complete(system=_DESCRIBE_SYSTEM, user=user, max_tokens=300).strip()


def _table_facts(client: LLMClient, table_text: str, context: str) -> list[str] | None:
    """Fact sentences for one table, digit-checked (see `_digit_runs`):
    a sentence asserting a number that occurs in neither the table nor its
    context is dropped -- an invented value must never be stored as a fact.
    Returns None when nothing survives (mirrors description's None-for-failed
    semantics)."""
    user = ""
    if context:
        user += f"Context around the table:\n{context}\n\n"
    user += f"Table:\n{table_text}\n\nWrite the fact sentences now."
    raw = client.complete(system=_FACTS_SYSTEM, user=user, max_tokens=1500)
    source_runs = _digit_runs(table_text) | _digit_runs(context)
    kept: list[str] = []
    for line in (raw or "").splitlines():
        # Tolerate a bulleted/numbered reply despite instructions.
        sentence = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+", "", line).strip()
        if sentence and _digit_runs(sentence) <= source_runs:
            kept.append(sentence)
    return kept or None


def _check(client: LLMClient, table_text: str, context: str, description: str) -> dict:
    user = ""
    if context:
        user += f"Context around the table:\n{context}\n\n"
    user += (
        f"Table:\n{table_text}\n\n"
        f"Candidate description:\n{description}\n\n"
        "Return your JSON verdict."
    )
    raw = client.complete(system=_CHECK_SYSTEM, user=user, max_tokens=200)
    match = _JSON_OBJECT.search(raw)
    if not match:
        return {"content_ok": False, "format_ok": False, "reason": "checker returned no JSON"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"content_ok": False, "format_ok": False, "reason": "checker returned invalid JSON"}
    return parsed


def describe_table_block(client: LLMClient, block: TableBlock, context: str, *,
                         use_check: bool, max_attempts: int,
                         use_facts: bool = True, max_rows: int = 0) -> None:
    """Describe (and optionally verify+retry) a single table, mutating `block`
    in place. Independent of every other block -- safe to run concurrently.

    `context` is already-gathered surrounding text (`describe/context.py`); the
    orchestrator owns gathering it, since that is shared with every other
    strategy and needs the block's position in the document.
    """
    table_text = render_table(block.table, max_rows=max_rows)
    if not table_text.strip():
        # No extractable cell text (e.g. a geometry-only region pdfplumber
        # mistook for a table) -- asking the LLM to describe nothing invites
        # a plausible-sounding fabrication instead of a faithful description,
        # and the checker can't verify content against an empty table either.
        block.description = None
        block.description_source = None
        block.describe_status = "empty"
        block.describe_attempts = 0
        return

    hint = ""
    passed = False
    attempts = 0
    description = ""
    while attempts < max_attempts:
        attempts += 1
        description = _describe(client, table_text, context, hint)
        if not use_check:
            break
        verdict = _check(client, table_text, context, description)
        passed = bool(verdict.get("content_ok")) and bool(verdict.get("format_ok"))
        if passed:
            break
        hint = str(verdict.get("reason") or "").strip()

    # An empty final attempt means no usable description was produced -- leave
    # the field None (unknown/failed) rather than "" (which to_dict's
    # `is not None` check would otherwise serialize as "successfully empty").
    block.description = description or None
    block.description_source = "llm-cells" if block.description else None
    if use_check:
        block.describe_status = "ok" if passed else "flagged"
        block.describe_attempts = attempts
    if use_facts:
        block.facts = _table_facts(client, table_text, context)
