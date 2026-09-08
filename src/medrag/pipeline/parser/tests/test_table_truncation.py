"""TABLE_MAX_ROWS: cap on table rows sent to the LLM (describe/table.py +
header_infer).

A huge table would otherwise overflow the model's context window, and Ollama
truncates an over-budget prompt silently -- the model would then describe a
table it only partially saw, with no trace anywhere. The cap renders only the
first N rows WITH an explicit "[table truncated: ...]" note, drops merge notes
for rows beyond the cap, and logs a warning; the facts digit-guard keeps
working against the truncated text, so a fact citing a dropped row's number is
discarded rather than stored unverifiable.
"""

from __future__ import annotations

from medrag.pipeline.parser.describe.core import describe_blocks
from medrag.pipeline.parser.describe.table import render_table
from medrag.pipeline.parser.parsers.base import (
    Merge,
    ParsedDocument,
    TableBlock,
    TableData,
    text_cell,
)
from medrag.pipeline.parser.tables.header_infer import infer_headers


def describe_tables(doc, client):
    """The table half of the shared describe pass (`stage="llm"`)."""
    return describe_blocks(doc, stage="llm", llm=client)


def _table(rows: list[list[str]], merges: list[Merge] | None = None) -> TableData:
    return TableData(n_rows=len(rows), n_cols=len(rows[0]),
                     cells=[[text_cell(v) for v in r] for r in rows],
                     merges=merges or [])


def _doc_with(table: TableData) -> ParsedDocument:
    doc = ParsedDocument(doc_id="d", source_path="x", fmt="test")
    doc.blocks = [TableBlock(id="t0", table=table)]
    return doc


class CapturingLLM:
    """Returns a canned reply and records every user prompt it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.users: list[str] = []

    def complete(self, *, system, user, max_tokens=1024):
        self.users.append(user)
        return self.reply


# --- render_table ------------------------------------------------------------


def test_render_uncapped_is_unchanged():
    t = _table([["a", "b"], ["c", "d"]])
    assert render_table(t) == "a | b\nc | d"
    assert render_table(t, max_rows=0) == "a | b\nc | d"      # 0 = unlimited
    assert render_table(t, max_rows=2) == "a | b\nc | d"      # cap not exceeded
    assert render_table(t, max_rows=99) == "a | b\nc | d"


def test_render_capped_keeps_prefix_and_says_so():
    t = _table([[f"r{i}", str(i)] for i in range(10)])
    out = render_table(t, max_rows=3)
    assert "r2 | 2" in out
    assert "r3" not in out
    assert "[table truncated: only the first 3 of 10 rows are shown above]" in out


def test_render_capped_drops_merges_beyond_the_cap():
    t = _table([[f"r{i}", str(i)] for i in range(10)],
               merges=[Merge(row=1, col=0, colspan=2),
                       Merge(row=8, col=0, colspan=2)])
    out = render_table(t, max_rows=3)
    assert "(row 1, col 0)" in out       # inside the rendered prefix: kept
    assert "(row 8, col 0)" not in out   # points at rows the model can't see


# --- describe/table.py ----------------------------------------------------------


def test_describe_sends_truncated_table(monkeypatch):
    monkeypatch.setenv("TABLE_MAX_ROWS", "2")
    monkeypatch.setenv("TABLE_FACTS", "0")
    monkeypatch.delenv("TABLE_LLM_CHECK", raising=False)
    doc = _doc_with(_table([[f"r{i}", str(i)] for i in range(5)]))
    llm = CapturingLLM("A table of rows.")

    describe_tables(doc, client=llm)

    assert doc.blocks[0].description == "A table of rows."
    (user,) = llm.users
    assert "r1 | 1" in user
    assert "r4" not in user
    assert "[table truncated" in user


def test_facts_digit_guard_uses_truncated_text(monkeypatch):
    """A fact citing a number only present in the dropped tail is discarded."""
    monkeypatch.setenv("TABLE_MAX_ROWS", "2")
    monkeypatch.setenv("TABLE_FACTS", "1")
    monkeypatch.delenv("TABLE_LLM_CHECK", raising=False)
    doc = _doc_with(_table([["pin", "volt"], ["p1", "33"], ["p2", "48"]]))
    llm = CapturingLLM("Pin p1 is 33 volts.\nPin p2 is 48 volts.")

    describe_tables(doc, client=llm)

    facts = doc.blocks[0].facts
    assert facts == ["Pin p1 is 33 volts."]  # "48" was cut away with row 2


def test_describe_unlimited_when_zero(monkeypatch):
    monkeypatch.setenv("TABLE_MAX_ROWS", "0")
    monkeypatch.setenv("TABLE_FACTS", "0")
    monkeypatch.delenv("TABLE_LLM_CHECK", raising=False)
    doc = _doc_with(_table([[f"r{i}", str(i)] for i in range(300)]))
    llm = CapturingLLM("A table of rows.")

    describe_tables(doc, client=llm)

    (user,) = llm.users
    assert "r299 | 299" in user
    assert "[table truncated" not in user


# --- header_infer ------------------------------------------------------------


def test_header_infer_caps_but_validates_against_real_shape(monkeypatch):
    """The prompt is capped, but the answer is still validated against the
    table's REAL row count (headers are leading rows, the tail is never
    needed)."""
    monkeypatch.setenv("TABLE_MAX_ROWS", "2")
    monkeypatch.delenv("TABLE_HEADER_LLM", raising=False)
    doc = _doc_with(_table([["H", "H2"]] + [[f"r{i}", str(i)] for i in range(9)]))
    llm = CapturingLLM("1")

    infer_headers(doc, client=llm)

    assert doc.blocks[0].table.header_rows == 1
    assert doc.blocks[0].table_flags == ["header-llm"]
    (user,) = llm.users
    assert "10 rows x 2 columns" in user  # real shape, not the capped one
    assert "r5" not in user
