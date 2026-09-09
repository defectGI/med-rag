"""Split a large document into page-aligned windows for chunked grading.

A single judge call carries every page image plus the whole Markdown; for a big
document that overflows the model's context, and an over-budget prompt is
truncated silently (see cli.py) -- the judge then scores pages it never saw. The
fix is to grade in page windows: a few pages at a time, with the Markdown SLICED
to exactly those pages so the judge is never asked to grade a slice against pages
it cannot see (which would wrongly tank coverage), and never shown pages the
slice doesn't cover.

Alignment is the crux. The rendered `.md` carries no page markers, so it cannot
be sliced directly. But the parser's IR JSON (which sits next to the `.md` --
see benchmark/discover.py) tags every block with the source page it came from
(`Span.page`, populated by the PDF parser). So we bucket the IR's blocks by page,
then re-render each window's blocks back to Markdown with the parser's own
`render.markdown.to_markdown`. The `.md` graded as a whole IS that renderer's
output over ALL blocks, so the concatenation of the window slices reproduces it
up to minor boundary effects (a list split across a window edge, a table's
trailing description). Re-rendering from the IR is the only way to get a
page-aligned slice at all; grading the exact `.md` byte-for-byte is impossible
without page markers it doesn't contain.

This module does NO model I/O -- it only turns (IR JSON + rendered page images)
into windows. The judge grades them (benchmark/judge.py) and the results are
aggregated page-weighted back into one score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageWindow:
    """One gradable slice: a contiguous page range, its Markdown, its images.

    page_lo/page_hi are 1-based inclusive source page numbers. `images` is the
    subset of the document's rendered page images for exactly this range (so
    `len(images) == page_hi - page_lo + 1` for a fully-rendered document).
    `markdown` is the parser output re-rendered from just the blocks the IR
    attributes to these pages -- possibly empty when the parser produced nothing
    for the range (blank/figure-only pages), which is itself gradable: coverage
    should notice content the pages carry but the Markdown dropped.
    """

    page_lo: int
    page_hi: int
    markdown: str
    images: list[tuple[str, bytes]]

    @property
    def n_pages(self) -> int:
        return len(self.images)


def pages_for_budget(*, markdown: str, scope_md: str | None, total_pages: int,
                     budget_tokens: float, cap: int | None = None) -> int:
    """A window height (in pages) whose estimated prompt cost fits `budget_tokens`.

    Uses the same rough estimator the pre-call warning does (judge.estimate_
    prompt_tokens), split into a fixed per-call cost (system + scope text) and a
    per-page cost (one page image + that page's share of the Markdown), then
    solves for how many pages fit. At least 1 (a single page always gets its own
    window even if that page alone is over budget -- there's nothing smaller to
    fall back to). `cap`, when given, is an upper bound (an explicit
    --chunk-pages ceiling)."""
    from tools.benchmark.judge import _TOKENS_PER_IMAGE, estimate_prompt_tokens

    fixed = estimate_prompt_tokens("", scope_md, 0)          # system + scope text
    md_tokens = max(0, estimate_prompt_tokens(markdown, scope_md, 0) - fixed)
    per_page = _TOKENS_PER_IMAGE + md_tokens / max(1, total_pages)
    pages = int((budget_tokens - fixed) / per_page) if per_page > 0 else total_pages
    pages = max(1, pages)
    if cap is not None:
        pages = min(pages, max(1, cap))
    return pages


def _block_page(block, fallback: int) -> int:
    """The 1-based source page a block belongs to, for windowing.

    Prefers `span.page`, then `span.page_start` (a block spanning pages is
    bucketed by where it starts). Blocks a parser left page-less (None) inherit
    the previous block's page -- reading order carries them along rather than
    dropping their content out of every window."""
    span = getattr(block, "span", None)
    if span is not None:
        for attr in ("page", "page_start"):
            val = getattr(span, attr, None)
            if isinstance(val, int) and val >= 1:
                return val
    return fallback


def build_windows(ir_json: Path, images: list[tuple[str, bytes]], *,
                  pages_per_window: int, warn=None) -> list[PageWindow] | None:
    """Turn a document into page-aligned windows, or None if it can't be windowed.

    `images` is the document's rendered page images in page order (images[i] is
    page i+1), as produced by benchmark/render.py. `pages_per_window` is the
    window height in pages.

    Returns None -- meaning "grade this as a single call, as before" -- when the
    IR can't be loaded, carries no per-block page numbers (e.g. a non-paginated
    source), or fits one window AND covers no pages beyond the rendered ones
    (when it does -- a --max-pages-truncated render of a longer PDF -- even a
    single window is built, because it slices the Markdown down to the rendered
    pages while the single-call fallback would send all of it). Never raises on a
    missing/corrupt IR: a benchmark must degrade to the old single-call path, not
    abort. `warn`, when given, is called with a one-line human-readable reason
    on every such fallback -- without it a failed windowing is indistinguishable
    from a document that was simply never over budget.
    """
    if pages_per_window < 1:
        raise ValueError("pages_per_window must be >= 1")

    def _fallback(reason: str) -> None:
        if warn is not None:
            warn(reason)

    total_pages = len(images)

    try:
        # 2026-08-27 fix: this used to import bare `parsers`/`render` on the
        # (stale) assumption that cli.py put `_PARSER_DIR` on sys.path -- it
        # never did (that var is only used for .env discovery, see cli.py),
        # so this import ALWAYS failed and page-aligned windowing silently
        # fell back to the single-call path on every real run, not just in
        # tests. `medrag` is an installed package (pyproject.toml
        # include=["medrag*","tools*"], see cli.py's own comment on that), so
        # the real dotted path always resolves without any sys.path hack --
        # kept as a lazy import (not a module-level one) only so this module
        # still imports cleanly in contexts where `medrag.pipeline.parser`'s
        # own heavy dependencies aren't installed.
        from medrag.pipeline.parser.parsers.base import ParsedDocument
        from medrag.pipeline.parser.render.markdown import to_markdown
    except Exception as exc:  # noqa: BLE001 -- any import failure => can't window
        _fallback(f"parser packages not importable ({exc!r})")
        return None

    try:
        doc = ParsedDocument.load(ir_json)
    except Exception as exc:  # noqa: BLE001 -- missing/corrupt IR => single-call fallback
        _fallback(f"IR JSON {ir_json.name} failed to load ({exc!r})")
        return None

    # Bucket blocks by source page (carry-forward for page-less blocks).
    by_page: dict[int, list] = {}
    last_page = 1
    saw_page = False
    for block in doc.blocks:
        page = _block_page(block, last_page)
        if getattr(getattr(block, "span", None), "page", None) or \
           getattr(getattr(block, "span", None), "page_start", None):
            saw_page = True
        last_page = page
        by_page.setdefault(page, []).append(block)

    if not saw_page:
        _fallback(f"IR JSON {ir_json.name} carries no per-block page numbers "
                  f"(no span.page/page_start on any of {len(doc.blocks)} block(s))")
        return None  # no page provenance at all -> can't align, single call

    # "Fits in one window" only means "nothing to gain" when the Markdown
    # covers no pages beyond the rendered ones. When rendering was truncated
    # (--max-pages on a longer PDF), the IR extends past `total_pages`, and the
    # single-call fallback would send the ENTIRE Markdown -- every unrendered
    # page's text -- alongside the few rendered images; that full text is
    # exactly what blew the context budget. A single window still slices the
    # Markdown down to the rendered pages, so build it.
    if total_pages <= pages_per_window and max(by_page) <= total_pages:
        _fallback(f"whole document ({total_pages} page(s)) fits one "
                  f"{pages_per_window}-page window")
        return None

    windows: list[PageWindow] = []
    lo = 1
    while lo <= total_pages:
        hi = min(lo + pages_per_window - 1, total_pages)
        # Blocks whose starting page falls in [lo, hi], kept in reading order.
        win_blocks = [b for p in range(lo, hi + 1) for b in by_page.get(p, [])]
        slice_doc = ParsedDocument(
            doc_id=doc.doc_id, source_path=doc.source_path, fmt=doc.fmt,
            blocks=win_blocks,
        )
        windows.append(PageWindow(
            page_lo=lo, page_hi=hi,
            markdown=to_markdown(slice_doc) if win_blocks else "",
            images=images[lo - 1:hi],
        ))
        lo = hi + 1
    return windows
