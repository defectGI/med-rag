"""Detects page-margin noise (running headers/footers, URLs, page numbers)
so `pdf_parser` can drop it before block classification.

Two independent, deliberately conservative signals, both gated on the same
`in_margin_band` position check so a one-off line that just happens to sit
near the page edge (real content) is never touched:

  1. `detect_running_lines` -- the exact same normalized text recurs, in the
     margin band, on at least `min_pages` distinct pages. Catches boilerplate
     that's identical everywhere (a URL, a "Caution" notice, ...); a phrase
     that repeats in the body but never sits in the margin is left alone.
  2. `looks_like_page_number` -- the line's own shape (a bare number, "N/M",
     "Page N", ...) marks it as a page number even though its *text* is
     different on every page and so can never repeat under signal 1.

This is intentionally its own module, not folded into `pdf_parser.py`'s
per-page classification: detection needs a whole-document view (one pass
over every page before any page is classified), while the thing it feeds --
"is this line, at this position, noise" -- is a single, format-agnostic
lookup any parser with a page/line notion could reuse.
"""

from __future__ import annotations

import re
from collections import defaultdict

BAND_FRAC = 0.08
MIN_PAGES = 3

_WS = re.compile(r"\s+")
_PAGE_NUMBER = re.compile(
    r"^\s*(page|sayfa)?\s*\d{1,4}\s*((/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def in_margin_band(top: float, bottom: float, page_height: float,
                   band_frac: float = BAND_FRAC) -> bool:
    if page_height <= 0:
        return False
    band = page_height * band_frac
    return top <= band or bottom >= page_height - band


def looks_like_page_number(text: str) -> bool:
    """A bare page-number line ("4", "4/10", "4 of 10", "Page 4", "Sayfa 4")
    -- never gated on repetition since the number differs on every page, only
    on `in_margin_band` at the call site, so a real "4/10" ratio sitting in
    the body of a page is never touched."""
    return bool(_PAGE_NUMBER.match(text))


def detect_running_lines(
    pages: list[tuple[float, list[tuple[float, float, str]]]],
    band_frac: float = BAND_FRAC, min_pages: int = MIN_PAGES,
) -> frozenset[str]:
    """`pages`: one (page_height, lines) entry per page; each line is
    (top, bottom, text) in that page's own coordinate space. Returns the
    normalized texts that repeat often enough, in the margin band, to be
    treated as running headers/footers/page numbers.

    `min_pages` is capped to the document's own page count so a short
    document (2-3 pages) can still flag a footer that repeats on every one
    of its pages, instead of requiring more repeats than there are pages.
    """
    threshold = max(2, min(min_pages, len(pages)))
    seen: dict[str, set[int]] = defaultdict(set)
    for pageno, (height, lines) in enumerate(pages):
        for top, bottom, text in lines:
            norm = normalize(text)
            if norm and in_margin_band(top, bottom, height, band_frac):
                seen[norm].add(pageno)
    return frozenset(t for t, on_pages in seen.items() if len(on_pages) >= threshold)
