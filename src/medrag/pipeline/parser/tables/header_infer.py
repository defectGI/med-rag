"""Fill `TableData.header_rows` via an LLM for tables whose source format
carried no header semantics.

Only a *fallback*: a table whose parser already set `header_rows` (html
<th>/<thead>, markdown pipe header, docx w:tblHeader, pptx firstRow, pdf band
header recovery) is never touched — the deterministic value always wins. The
LLM answers one narrow classification question per table ("how many leading
rows are header rows?"), the reply is validated against the table's shape, and
an accepted value is marked with a "header-llm" entry in
`TableBlock.table_flags` so consumers can tell an inferred header count from a
source-stated one. A rejected/failed reply leaves `header_rows` None — the
failure mode is the pre-existing behavior, never a wrong claim.

This is a table-shape question, not a description, so it stays here rather than
in `describe/` -- but it reuses that package's shared pieces (context gathering,
table rendering) instead of keeping private copies.

Tuning (`config/default.toml`):
    [table] header_llm       false to skip the pass entirely (default on)
    [table] max_rows         table rows sent to the model (0 = unlimited;
                             shared with describe/table.py). Harmless here even
                             when it bites: header rows are leading rows, so
                             the classification never needs the tail.
    [describe.context]       surrounding-context injection (shared with every
                             describable block type; default off)
    [describe] concurrency   tables classified in parallel
`[table]`'s own knobs (TABLE_HEADER_LLM, TABLE_MAX_ROWS) keep their historical
env names; `[describe]`'s are DESCRIBE_CONTEXT*/DESCRIBE_CONCURRENCY (the old
TABLE_CONTEXT*/TABLE_CONCURRENCY names were retired, not aliased -- see
config.py's `_ENV_OVERRIDES`).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from medrag.pipeline.parser.config import get_config
from medrag.pipeline.parser.describe.context import build_context
from medrag.pipeline.parser.describe.table import render_table
from medrag.pipeline.parser.llm import LLMClient, get_client
from medrag.pipeline.parser.parsers.base import ParsedDocument, TableBlock

# More than a few header rows in a real table is vanishingly rare; an answer
# above this bound is far more likely a model misreading than a real layout.
_MAX_HEADER_ROWS = 3

_SYSTEM = (
    "You look at one table and decide how many of its LEADING rows are header "
    "rows (rows that label the columns, e.g. 'Name | Min | Max'), as opposed "
    "to data rows. Many tables have exactly one header row; some have none "
    "(every row is data); a few have two or more stacked header rows.\n"
    "Answer with a single integer (0, 1, 2, ...) and nothing else."
)

_INT = re.compile(r"-?\d+")


def _infer_one(client: LLMClient, doc: ParsedDocument, idx: int, *,
               use_context: bool, max_chars: int, before: int, after: int,
               max_rows: int = 0) -> None:
    """Classify a single table's header row count, mutating its block in
    place. Independent of every other table — safe to run concurrently."""
    block: TableBlock = doc.blocks[idx]
    table = block.table
    table_text = render_table(table, max_rows=max_rows)
    if not table_text.strip():
        return  # nothing to classify (geometry-only region)

    user = ""
    if use_context:
        context = build_context(doc.blocks, idx, max_chars, before, after)
        if context:
            user += f"Context around the table:\n{context}\n\n"
    user += (f"Table ({table.n_rows} rows x {table.n_cols} columns):\n"
             f"{table_text}\n\nHow many leading rows are header rows? "
             "Answer with a single integer.")

    try:
        raw = client.complete(system=_SYSTEM, user=user, max_tokens=50)
    except Exception:  # noqa: BLE001 -- best-effort inference: if the LLM blows up, header_rows stays None (unknown); the caller (to_markdown) warns separately
        return  # best-effort: header_rows stays None (unknown)

    match = _INT.search(raw or "")
    if not match:
        return
    value = int(match.group(0))
    # A header-only table (value == n_rows) leaves no data rows — treat any
    # such answer, and anything past the sanity cap, as a misread.
    if not 0 <= value <= min(_MAX_HEADER_ROWS, table.n_rows - 1):
        return
    table.header_rows = value
    block.table_flags = (block.table_flags or []) + ["header-llm"]


def infer_headers(doc: ParsedDocument, client: LLMClient | None = None) -> ParsedDocument:
    """Fill `header_rows` on tables where it is still None (unknown).

    Mutates `doc` in place and returns it. No-op when TABLE_HEADER_LLM is off
    or no table needs inference — so no client/network is required then.
    Failures are per-table and silent (the value simply stays None).
    """
    if not get_config().table.header_llm:
        return doc
    indices = [i for i, b in enumerate(doc.blocks)
               if isinstance(b, TableBlock)
               and b.table.header_rows is None
               and b.table.n_rows >= 2]
    if not indices:
        return doc

    if client is None:
        client = get_client()

    cfg = get_config()
    ctx = cfg.describe.context          # context/concurrency are shared by every
    use_context = ctx.enabled           # describable block type, not table-only
    max_chars = ctx.max_chars
    before = max(0, ctx.before)
    after = max(0, ctx.after)
    max_workers = max(1, cfg.describe.concurrency)
    max_rows = max(0, cfg.table.max_rows)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_infer_one, client, doc, idx,
                               use_context=use_context, max_chars=max_chars,
                               before=before, after=after, max_rows=max_rows)
                   for idx in indices]
        for f in futures:
            f.result()

    inferred = sum(1 for i in indices if doc.blocks[i].table.header_rows is not None)
    print(f"table headers: {inferred}/{len(indices)} inferred via LLM", flush=True)
    return doc
