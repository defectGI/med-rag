"""Document-wide heading reconciliation for the PDF parser.

Heading *levels* in the raw block stream come from two unrelated per-page
sources -- the code path ranks font sizes, the hybrid path copies the VLM's
own per-page guess -- so the same logical depth can surface as H1 on one page
and H4 on the next, and a heading the VLM mistyped as a paragraph/list item
vanishes from the hierarchy entirely. This module runs once over the finished
document (after TOC extraction, before `heading_path` assignment) and rebuilds
every heading's level from the document's own strongest available signals,
in strict priority order:

  1. The PDF's native outline (bookmarks): the author's explicit hierarchy,
     with exact titles and depths. Matched to blocks in reading order; an
     outline entry whose heading the VLM dropped recovers it by promoting the
     matching paragraph/list item.
  2. Numbered titles: "3.5. Ethernet" carries its own depth (dot components).
     A number missing from the sequence ("3." ... "3.2") recovers the
     paragraph that carries it ("3.1. ...") the same way.
  3. The document's printed table of contents (`metadata["toc"]`, already
     extracted): levels for unnumbered headings, recovery for dropped ones.
  4. Font size, re-anchored: sizes are mapped to levels through the headings
     already leveled by 1-3, so a one-off cover-page display font can no
     longer push every body heading down the ladder. Works for VLM headings
     too -- their text is layer-verified, so the physical line (and its size)
     is found in the page's own text layer.

Each signal is optional; a document with none of them (a brochure) keeps the
plain doc-wide font ladder, now applied to VLM headings as well, so at worst
the document is internally consistent on ONE scheme instead of two.

Deliberately NOT done here: restoring a number the VLM stripped from a
heading's text (would desync `runs`), and demoting headings on any signal
weaker than URL shape / page-number shape / running-line text -- aggressive
demotion eats real headings in unnumbered documents.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from itertools import pairwise
from typing import Any

from .base import Block, HeadingBlock, ParagraphBlock
from .heading_heuristics import CAPTION
from .running_lines import looks_like_page_number, normalize

# ---------------------------------------------------------------------------
# Numbered-title paths
# ---------------------------------------------------------------------------

# "3.", "3.5", "2.2.1." followed by actual title text. Components capped at
# 2 digits (99): section counters realistically stay far below that, while
# the things that would otherwise slip through -- IP addresses ("192.168…"),
# years -- carry a 3+ digit component somewhere. Split the same way
# `_BULLET_RE` splits its own numbering branches: a dotted path ("3.5",
# "5.4.2.1") is self-evidently section numbering from its internal dots
# alone, so the trailing "." is optional -- but a BARE single number needs
# one mandatory (else "5 V / 20 W power output", a spec badge caption, reads
# as "chapter 5, title 'V / 20 W power output'").
_NUM_TITLE_DOTTED = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){1,5})\.?\s+\S")
_NUM_TITLE_BARE = re.compile(r"^\s*(\d{1,2})[.)]\s+\S")
# Appendix numbering: "A.1", "B.2.3" (a bare "A. Title" is left to the other
# signals -- single letters collide with lettered list markers).
_APPENDIX_TITLE = re.compile(r"^\s*([A-Z])\.(\d{1,2}(?:\.\d{1,2})*)\.?\s+\S")
# "IV. Results" -- uppercase only: lowercase roman ("i.", "ii.") is a nested
# list-marker convention, not a chapter one.
_ROMAN_TITLE = re.compile(r"^\s*([IVXLCDM]{1,7})[.)]\s+\S")
_ROMAN_VALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

# First path component reserved for appendix letters: "A" -> 100, "B" -> 101,
# ... so appendix paths sort after every numeric chapter (they do in print,
# too) while staying plain int tuples that compare with `<`.
_APPENDIX_BASE = 100

_MAX_LEVEL = 6


def _roman_to_int(s: str) -> int | None:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        val = _ROMAN_VALS.get(ch)
        if val is None:
            return None
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


def num_path(text: str) -> tuple[int, ...] | None:
    """The dotted number path a title starts with, as an int tuple whose
    length is the hierarchy depth: "3. Signals" -> (3,), "3.5. Ethernet" ->
    (3, 5), "A.1 Pinout" -> (100, 1), "IV. Results" -> (4,). None when the
    text doesn't open with plausible section numbering."""
    m = _NUM_TITLE_DOTTED.match(text)
    if m:
        return tuple(int(c) for c in m.group(1).split("."))
    m = _NUM_TITLE_BARE.match(text)
    if m:
        return (int(m.group(1)),)
    m = _APPENDIX_TITLE.match(text)
    if m:
        letter = _APPENDIX_BASE + ord(m.group(1)) - ord("A")
        return (letter, *(int(c) for c in m.group(2).split(".")))
    m = _ROMAN_TITLE.match(text)
    if m:
        val = _roman_to_int(m.group(1))
        # Cap keeps genuine chapter numerals and rejects the letter-list
        # collisions ("C.", "D.", "M." are letters 100/500/1000, not chapters).
        if val is not None and val <= 49:
            return (val,)
    return None


def strip_num(text: str) -> str:
    """Title text with its leading number path (if any) removed. Shared with
    pdf_parser's TOC extraction: a split-shape TOC entry (see module
    docstring) keeps its heading's number in `.text`, but this document's
    other TOC entries are stored number-free (`_BULLET_RE` already strips a
    glued entry's leading number before it ever becomes a block) -- entries
    from both shapes need the same number-free convention to land in one
    consistent `metadata["toc"]` list."""
    for rx in (_NUM_TITLE_DOTTED, _NUM_TITLE_BARE, _APPENDIX_TITLE, _ROMAN_TITLE):
        m = rx.match(text)
        if m:
            # match ends at the first char of the real title (`\S`)
            return text[m.end() - 1:]
    return text


# ---------------------------------------------------------------------------
# Block predicates / conversions
# ---------------------------------------------------------------------------

_URLISH = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
# A flattened TOC entry that survived extraction: title glued to a bare
# trailing page number ("Configuration 11"). Never promotion material.
_TRAILING_PAGENO = re.compile(r"\s\d{1,4}\s*$")


def _promotable(b: Block) -> bool:
    """Could this paragraph plausibly be a heading the VLM mistyped? Shape
    checks only -- the caller still needs a positive identity signal (an
    outline/TOC entry match, or a number filling a sequence gap)."""
    if not isinstance(b, ParagraphBlock) or type(b) is not ParagraphBlock:
        return False
    t = b.text.strip()
    if not t or len(t.split()) > 16:
        return False
    if t.endswith((".", ";", ":", ",")) or CAPTION.match(t):
        return False
    return not _TRAILING_PAGENO.search(t)


def _promote(b: ParagraphBlock, level: int) -> HeadingBlock:
    return HeadingBlock(
        id=b.id, span=b.span, heading_path=b.heading_path,
        provenance=b.provenance, source_crop=b.source_crop,
        text=b.text, level=max(1, min(_MAX_LEVEL, level)), runs=b.runs)


def _demote(b: HeadingBlock) -> ParagraphBlock:
    return ParagraphBlock(
        id=b.id, span=b.span, heading_path=b.heading_path,
        provenance=b.provenance, source_crop=b.source_crop,
        text=b.text, runs=b.runs)


# ---------------------------------------------------------------------------
# Entry matching (outline / printed TOC)
# ---------------------------------------------------------------------------


def _keys(text: str) -> tuple[str, str]:
    """(normalized, normalized-with-number-stripped) match keys. The second
    key lets an outline title "2. Hardware Overview" find the heading whose
    number the VLM dropped ("Hardware Overview"), and vice versa."""
    return normalize(text), normalize(strip_num(text))


def _entry_match(blocks: list[Block], pos: int, title: str) -> int | None:
    """Index of the block at/after `pos` that carries this entry's title,
    or None. Existing headings are preferred over promotable paragraphs
    document-wide, not just first-in-reading-order: a printed TOC whose page
    numbers sit in a separate visual column leaves bare section titles as
    plain paragraphs on the TOC page, and those always precede the real
    heading -- first-match-wins would promote the TOC line and starve the
    genuine heading. A paragraph is promotion material only when no heading
    anywhere ahead carries the title (the VLM-dropped-heading case). Fuzzy
    similarity is likewise reserved for headings; a fuzzy paragraph match
    would promote real prose."""
    tkeys = [k for k in _keys(title) if k]
    if not tkeys:
        return None
    para_match: int | None = None
    for i in range(pos, len(blocks)):
        b = blocks[i]
        if isinstance(b, HeadingBlock):
            bkeys = [k for k in _keys(b.text) if k]
            if any(bk == tk for bk in bkeys for tk in tkeys):
                return i
            if SequenceMatcher(None, normalize(b.text),
                               normalize(title)).ratio() >= 0.88:
                return i
        elif para_match is None and _promotable(b):
            bkeys = [k for k in _keys(b.text) if k]
            if any(bk == tk for bk in bkeys for tk in tkeys):
                para_match = i
    return para_match


def _apply_entries(blocks: list[Block], entries: Sequence[Mapping[str, Any]],
                   leveled: dict[int, str], source: str,
                   stats: Counter, unmatched: list[dict[str, Any]]) -> None:
    """Walk `entries` ({"level", "title", page: optional}) in document order,
    aligning each to its block monotonically (an entry never matches before
    the previous match -- section names repeat across chapters, and reading
    order is what tells the copies apart). Matched headings get the entry's
    depth; matched promotable paragraphs are promoted to headings at that
    depth. An unmatched entry moves on without advancing the block cursor,
    and is recorded in `unmatched` (page carried through when the source
    entry has one, e.g. a printed TOC entry -- an outline entry currently
    doesn't, see `PdfParser._pdf_outline`) so a caller can act on exactly
    which titles the document promised but no block ever delivered (see
    `reconcile_headings`'s docstring) -- not deduplicated against the OTHER
    entry source possibly flagging the very same missing heading; a caller
    that cares (e.g. one entry per source is enough to act on) filters by
    "source" itself rather than this function guessing which one to keep."""
    pos = 0
    for entry in entries:
        depth, title = int(entry["level"]), str(entry["title"])
        idx = _entry_match(blocks, pos, title)
        if idx is None:
            stats[f"{source}_unmatched"] += 1
            rec: dict[str, Any] = {"source": source, "title": title, "level": depth}
            page = entry.get("page")
            if page is not None:
                rec["page"] = page
            unmatched.append(rec)
            continue
        b = blocks[idx]
        if isinstance(b, ParagraphBlock):
            blocks[idx] = _promote(b, depth)
            leveled[idx] = source
            stats["promoted"] += 1
        elif idx not in leveled:
            b.level = max(1, min(_MAX_LEVEL, depth))
            leveled[idx] = source
        stats[f"{source}_matched"] += 1
        pos = idx + 1


# ---------------------------------------------------------------------------
# Reconciliation stages
# ---------------------------------------------------------------------------


def _demote_noise(blocks: list[Block], running_lines: frozenset[str],
                  stats: Counter) -> None:
    """URL-shaped / page-number-shaped / running-line heading text is margin
    noise the VLM lifted into the flow -- the code path already drops these
    lines before classification (`_is_margin_noise`), so a heading still
    carrying one can only be a transcription artifact."""
    for i, b in enumerate(blocks):
        if not isinstance(b, HeadingBlock):
            continue
        t = b.text.strip()
        if (_URLISH.match(t) or looks_like_page_number(t)
                or normalize(t) in running_lines):
            blocks[i] = _demote(b)
            stats["demoted"] += 1


def _level_numbered(blocks: list[Block], leveled: dict[int, str],
                    stats: Counter) -> None:
    for i, b in enumerate(blocks):
        if i in leveled or not isinstance(b, HeadingBlock):
            continue
        path = num_path(b.text)
        if path:
            b.level = min(len(path), _MAX_LEVEL)
            leveled[i] = "number"
            stats["leveled_by_number"] += 1


def _promote_gap_fills(blocks: list[Block], leveled: dict[int, str],
                       stats: Counter) -> None:
    """A paragraph whose own number falls strictly between two neighboring
    numbered headings' paths ("3." < "3.1." < "3.2.") is that missing
    heading. Tuple order does the sequence math: (3,) < (3, 1) < (3, 2).
    Sentinels open the walk at both ends so a drop before the first / after
    the last numbered heading still recovers, while a restarting ordered
    list ("1., 2., ..." inside chapter 4) can never qualify -- its numbers
    sort before the surrounding anchors, not between them."""
    anchors = [(i, num_path(b.text)) for i, b in enumerate(blocks)
               if isinstance(b, HeadingBlock) and num_path(b.text)]
    if not anchors:
        return
    bounded = [(-1, ())] + anchors + [(len(blocks), (10 ** 6,))]
    for (i1, p1), (i2, p2) in pairwise(bounded):
        prev = p1
        for j in range(i1 + 1, i2):
            b = blocks[j]
            if not _promotable(b):
                continue
            q = num_path(b.text)
            if q and prev < q < p2:
                blocks[j] = _promote(b, len(q))
                leveled[j] = "gap"
                stats["promoted"] += 1
                prev = q


def _lookup_size(b: HeadingBlock,
                 page_lines: Sequence[tuple[str, float]] | None) -> float | None:
    """The heading's physical font size, found by locating its text in the
    page's own visual lines. Exact match first; then prefix containment
    either way, since a wrapped heading's block text spans several physical
    lines (line is a prefix of block) and a VLM line split does the reverse."""
    if not page_lines:
        return None
    t = normalize(b.text)
    if len(t) < 4:
        return None
    best: tuple[int, float] | None = None
    for lt, size in page_lines:
        if lt == t:
            return size
        if len(lt) >= 6 and (t.startswith(lt) or (len(t) >= 6 and lt.startswith(t))):
            n = min(len(lt), len(t))
            if best is None or n > best[0]:
                best = (n, size)
    return best[1] if best else None


def _level_by_font(blocks: list[Block], leveled: dict[int, str],
                   line_sizes: Mapping[int, Sequence[tuple[str, float]]],
                   heading_sizes: Sequence[float], stats: Counter) -> None:
    """Levels for the headings no stronger signal reached, from font size --
    but anchored to the already-leveled headings' sizes instead of the raw
    document-wide ladder, so the ladder's top steps aren't wasted on one-off
    cover-page display text. With no anchors at all (an unnumbered,
    outline-less document) the plain ladder still applies -- crucially now
    to VLM headings too, which used to keep the model's per-page guess."""
    anchor_votes: dict[float, Counter] = {}
    for i in leveled:
        b = blocks[i]
        if not isinstance(b, HeadingBlock):
            continue
        size = _lookup_size(b, line_sizes.get(b.span.page))
        if size is not None:
            anchor_votes.setdefault(round(size, 1), Counter())[b.level] += 1
    # majority level per size; ties break shallow (min)
    anchors: dict[float, int] = {}
    for sz, votes in anchor_votes.items():
        best_n = max(votes.values())
        anchors[sz] = min(lvl for lvl, n in votes.items() if n == best_n)

    ladder = sorted({round(s, 1) for s in heading_sizes}, reverse=True)
    for i, b in enumerate(blocks):
        if i in leveled or not isinstance(b, HeadingBlock):
            continue
        size = _lookup_size(b, line_sizes.get(b.span.page))
        if size is None:
            continue
        size = round(size, 1)
        if anchors:
            lo, hi = min(anchors), max(anchors)
            if size > hi + 0.3:
                level = max(1, anchors[hi] - 1)
            elif size < lo - 0.3:
                level = min(_MAX_LEVEL, anchors[lo] + 1)
            else:
                nearest = min(anchors, key=lambda s: abs(s - size))
                level = anchors[nearest]
        elif ladder:
            nearest = min(ladder, key=lambda s: abs(s - size))
            if abs(nearest - size) > 0.3:
                continue
            level = min(_MAX_LEVEL, ladder.index(nearest) + 1)
        else:
            continue
        b.level = level
        leveled[i] = "font"
        stats["leveled_by_font"] += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def reconcile_headings(
    blocks: list[Block],
    *,
    outline: Sequence[tuple[int, str]] = (),
    toc: Sequence[Mapping[str, Any]] = (),
    line_sizes: Mapping[int, Sequence[tuple[str, float]]] | None = None,
    heading_sizes: Sequence[float] = (),
    running_lines: frozenset[str] = frozenset(),
) -> tuple[list[Block], dict[str, Any]]:
    """Rebuild every heading's level from the document's strongest signals
    (see module docstring), promoting mistyped headings back into the
    hierarchy and demoting margin noise. Returns (blocks, stats); `blocks`
    is the same list with some elements replaced in place (promotions /
    demotions preserve id, span, runs and provenance). `stats` is mostly
    int counters, plus one list-valued key -- `stats["unmatched_entries"]`,
    present only when non-empty -- of every outline/TOC entry that named a
    heading no block ever delivered (see `_apply_entries`): the document
    promised this title exists (and, for a TOC entry, on which page) but the
    page's own transcription (VLM or otherwise) never produced it, so
    there's nothing here to promote -- surfaced for a caller to act on
    (audit, or a further recovery pass) rather than silently dropped."""
    blocks = list(blocks)
    line_sizes = line_sizes or {}
    stats: Counter = Counter()
    leveled: dict[int, str] = {}
    unmatched: list[dict[str, Any]] = []

    _demote_noise(blocks, running_lines, stats)
    _apply_entries(blocks, [{"level": lvl, "title": title} for lvl, title in outline],
                   leveled, "outline", stats, unmatched)
    _level_numbered(blocks, leveled, stats)
    _promote_gap_fills(blocks, leveled, stats)
    _apply_entries(blocks,
                   [{"level": int(e.get("level") or 1), "title": str(e.get("title") or ""),
                     "page": e.get("page")} for e in toc],
                   leveled, "toc", stats, unmatched)
    _level_by_font(blocks, leveled, line_sizes, heading_sizes, stats)
    result: dict[str, Any] = dict(stats)
    if unmatched:
        result["unmatched_entries"] = unmatched
    return blocks, result
