"""heading_reconcile tests: pure block-list transformations, no PDFs.

Each test builds the block stream a parse would have produced (mixed
correct/incorrect levels, mistyped headings) and asserts the reconciled
levels/types -- signals (outline, toc, sizes) are passed as plain data.
"""

from __future__ import annotations

from medrag.pipeline.parser.parsers.base import HeadingBlock, ParagraphBlock, Span
from medrag.pipeline.parser.parsers.heading_reconcile import (
    num_path,
    reconcile_headings,
)


def H(text, level=1, page=1, bid="h"):
    return HeadingBlock(id=bid, span=Span(page=page), text=text, level=level)


def P(text, page=1, bid="p", **kw):
    return ParagraphBlock(id=bid, span=Span(page=page), text=text, **kw)


# ---------------------------------------------------------------------------
# num_path
# ---------------------------------------------------------------------------


def test_num_path_shapes():
    assert num_path("3. Signal Connections") == (3,)
    assert num_path("3.5. MIL-STD-1553 Connector") == (3, 5)
    assert num_path("2.2.1 Electrical") == (2, 2, 1)
    assert num_path("A.1 Connector Pinout") == (100, 1)
    assert num_path("IV. Results") == (4,)
    # not section numbering:
    assert num_path("192.168.0.100 Setup") is None          # IP-like component
    assert num_path("26.09.2025 View Revision History") is None  # date
    assert num_path("3.5") is None                          # no title text
    assert num_path("iv. results") is None                  # lowercase roman = list
    assert num_path("C. Something") is None                 # letter list, not roman 100
    assert num_path("Plain Title") is None
    # bare single number with no trailing "." or ")" is not chapter numbering
    # -- a spec-badge caption ("5 V / 20 W power output"), not "chapter 5":
    assert num_path("5 V / 20 W power output") is None
    assert num_path("8 CH 52 kS/s") is None
    assert num_path("5. Safety Guidelines") == (5,)          # bare number WITH punctuation is


# ---------------------------------------------------------------------------
# numbered backbone
# ---------------------------------------------------------------------------


def test_numbered_headings_get_depth_levels_regardless_of_source():
    blocks = [
        H("1. Description", level=3),        # code-route ladder said 3
        H("1.1. Key Features", level=4),
        H("3. Signal Connections", level=1),  # VLM said 1
        H("3.2. USB Connector", level=1),     # VLM said 1
        H("3.3. Ethernet Connector", level=2),
    ]
    out, stats = reconcile_headings(blocks)
    assert [b.level for b in out] == [1, 2, 1, 2, 2]
    assert stats["leveled_by_number"] == 5


def test_gap_fill_promotes_paragraph_with_missing_number():
    blocks = [
        H("3. Signal Connections"),
        P("3.1. Power Connector (USB Type-C)", bid="gap"),
        H("3.2. USB Connector"),
    ]
    out, stats = reconcile_headings(blocks)
    promoted = out[1]
    assert isinstance(promoted, HeadingBlock)
    assert promoted.level == 2
    assert promoted.id == "gap"
    assert stats["promoted"] == 1


def test_restarting_ordered_list_is_not_promoted():
    blocks = [
        H("4. Configuration"),
        P("1. First option of the ordered list"),
        P("2. Second option of the ordered list"),
        H("5. Safety Guidelines"),
    ]
    out, _ = reconcile_headings(blocks)
    assert isinstance(out[1], ParagraphBlock)
    assert isinstance(out[2], ParagraphBlock)


def test_toc_leftover_with_trailing_page_number_not_promoted():
    blocks = [
        H("3. Signal Connections"),
        P("3.1. Power Connector 11"),  # flattened TOC entry, not the heading
        H("3.2. USB Connector"),
    ]
    out, _ = reconcile_headings(blocks)
    assert isinstance(out[1], ParagraphBlock)


# ---------------------------------------------------------------------------
# outline
# ---------------------------------------------------------------------------


def test_outline_sets_levels_and_recovers_dropped_heading():
    blocks = [
        H("Hardware Overview", level=2),               # VLM dropped the "2."
        P("2.1. Circuitry", bid="c", list_id="l0"),    # VLM mistyped as list item
        H("2.2.2. Physical Specifications", level=2),  # VLM guessed 2
        P("Body prose that stays prose. It is long enough and ends in a period."),
    ]
    outline = [(1, "2. Hardware Overview"),
               (2, "2.1. Circuitry"),
               (3, "2.2.2. Physical Specifications")]
    out, stats = reconcile_headings(blocks, outline=outline)
    assert isinstance(out[0], HeadingBlock) and out[0].level == 1
    assert isinstance(out[1], HeadingBlock) and out[1].level == 2
    assert out[2].level == 3
    assert isinstance(out[3], ParagraphBlock)
    assert stats["outline_matched"] == 3
    assert stats["promoted"] == 1


def test_outline_matching_is_monotonic_for_repeated_titles():
    # "Electrical" exists under chapter 2 and chapter 5; in-order matching
    # must bind the first entry to the first occurrence, not both to one.
    blocks = [
        H("2. Alpha"), H("Electrical", level=4, bid="e1"),
        H("5. Beta"), H("Electrical", level=1, bid="e2"),
    ]
    outline = [(1, "2. Alpha"), (2, "Electrical"),
               (1, "5. Beta"), (2, "Electrical")]
    out, _ = reconcile_headings(blocks, outline=outline)
    assert out[1].level == 2 and out[3].level == 2


# ---------------------------------------------------------------------------
# printed TOC
# ---------------------------------------------------------------------------


def test_toc_levels_unnumbered_heading_without_touching_numbered():
    blocks = [
        H("Hardware Overview", level=4),
        H("3. Signal Connections", level=1),
    ]
    toc = [{"title": "Hardware Overview", "page": 3, "level": 1},
           {"title": "Signal Connections", "page": 5, "level": 1}]
    out, _ = reconcile_headings(blocks, toc=toc)
    assert out[0].level == 1
    assert out[1].level == 1  # from its own number, toc agrees


# ---------------------------------------------------------------------------
# unmatched entries surfaced for audit / further recovery
# ---------------------------------------------------------------------------


def test_unmatched_toc_entry_surfaced_with_its_page():
    # "Circuitry" never appears in any shape (the VLM skipped it entirely,
    # not just its number) -- nothing to promote, but the gap is reported
    # with the page the printed TOC itself named for it.
    blocks = [H("2. Hardware Overview", level=1)]
    toc = [{"title": "Hardware Overview", "page": 3, "level": 1},
           {"title": "Circuitry", "page": 3, "level": 2}]
    _, stats = reconcile_headings(blocks, toc=toc)
    assert stats["toc_unmatched"] == 1
    assert stats["unmatched_entries"] == [
        {"source": "toc", "title": "Circuitry", "level": 2, "page": 3}]


def test_unmatched_outline_entry_has_no_page():
    # Outline entries (see PdfParser._pdf_outline) carry no page destination
    # today -- the "page" key is simply absent, not None/0, since a caller
    # (e.g. a VLM2 cross-check) can only act on a page it actually knows.
    blocks = [H("Something Else", level=1)]
    outline = [(1, "Missing Chapter")]
    _, stats = reconcile_headings(blocks, outline=outline)
    assert stats["unmatched_entries"] == [
        {"source": "outline", "title": "Missing Chapter", "level": 1}]


def test_no_unmatched_key_when_everything_matches():
    blocks = [H("1. Alpha", level=1)]
    outline = [(1, "1. Alpha")]
    _, stats = reconcile_headings(blocks, outline=outline)
    assert "unmatched_entries" not in stats


# ---------------------------------------------------------------------------
# font anchoring
# ---------------------------------------------------------------------------


def test_font_anchor_levels_unnumbered_heading_from_backbone_size():
    blocks = [
        H("3. Signal Connections", level=1, page=1),
        H("Hardware Overview", level=4, page=2),   # same 16pt font, wrong level
        H("Fine Print Subtitle", level=1, page=2),  # smaller than every anchor
    ]
    line_sizes = {
        1: [("3. signal connections", 16.0)],
        2: [("hardware overview", 16.0), ("fine print subtitle", 11.0)],
    }
    out, stats = reconcile_headings(blocks, line_sizes=line_sizes)
    assert out[1].level == 1          # matches the level-1 anchor size
    assert out[2].level == 2          # below all anchors -> deepest + 1
    assert stats["leveled_by_font"] == 2


def test_ladder_fallback_when_no_backbone():
    # Unnumbered doc (brochure): VLM headings adopt the doc-wide ladder via
    # their text-layer size instead of keeping the model's guess.
    blocks = [
        H("Big Display Title", level=3, page=1),
        H("Smaller Subtitle", level=1, page=1),
    ]
    line_sizes = {1: [("big display title", 24.0), ("smaller subtitle", 14.0)]}
    out, _ = reconcile_headings(blocks, line_sizes=line_sizes,
                                heading_sizes=[24.0, 14.0])
    assert out[0].level == 1
    assert out[1].level == 2


def test_wrapped_heading_size_found_by_prefix():
    blocks = [
        H("1. Alpha", level=1, page=1),
        H("A very long heading title that wrapped onto a second line",
          level=5, page=1),
    ]
    line_sizes = {1: [("1. alpha", 15.0),
                      ("a very long heading title that", 15.0),
                      ("wrapped onto a second line", 15.0)]}
    out, _ = reconcile_headings(blocks, line_sizes=line_sizes)
    assert out[1].level == 1


# ---------------------------------------------------------------------------
# demotion
# ---------------------------------------------------------------------------


def test_url_and_running_line_headings_demoted():
    blocks = [
        H("www.example.com", level=5, bid="u"),
        H("ACME Proprietary", level=4, bid="r"),
        H("1. Description", level=1),
    ]
    out, stats = reconcile_headings(
        blocks, running_lines=frozenset({"acme proprietary"}))
    assert isinstance(out[0], ParagraphBlock)
    assert isinstance(out[1], ParagraphBlock)
    assert isinstance(out[2], HeadingBlock)
    assert stats["demoted"] == 2


def test_demoted_heading_never_becomes_font_anchor():
    blocks = [
        H("www.example.com", level=1, page=1),
        H("Real Title", level=3, page=1),
    ]
    line_sizes = {1: [("www.example.com", 20.0), ("real title", 20.0)]}
    out, _ = reconcile_headings(blocks, line_sizes=line_sizes,
                                heading_sizes=[20.0])
    assert isinstance(out[0], ParagraphBlock)
    assert out[1].level == 1  # ladder fallback, not poisoned by the URL


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------


def test_no_signals_leaves_blocks_untouched():
    blocks = [H("Mystery Heading", level=2), P("Prose.")]
    out, stats = reconcile_headings(blocks)
    assert out[0].level == 2
    assert not any(stats.values())
