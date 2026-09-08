"""Shared GFM pipe-table rule: what to render when a table has no header.

`medrag.pipeline.parser.render.markdown._render_gfm` and
`medrag.pipeline.chunker.core.markdown.render_table` each independently
implement the same non-obvious rule -- GFM requires a header line, but when
a table is KNOWN to have no header row (`header_rows == 0`), promoting the
first data row into that slot would misrepresent the table. Both renderers
instead emit an empty header row (all cells blank) and keep every row in the
body.

**Deliberately NOT a full merge**: the two renderers' input types differ too
much to unify further -- parser's `Cell` is block-structured with
`header_rows: int | None` (a third
"unknown" state that promotes the first row instead), chunker's cells are
plain strings with `header_rows: int` always concrete. Only this one rule is
shared; each renderer still builds its own row/line logic around it.
"""

from __future__ import annotations


def gfm_empty_header_row(ncols: int) -> str:
    """The header line GFM requires when a table has no real header
    (`header_rows == 0`): `ncols` empty cells, never a promoted data row."""
    return "| " + " | ".join([""] * ncols) + " |"
