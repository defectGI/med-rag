"""Deterministic "band" table-grid builder for born-digital PDF pages.

Motivation (from the e2e table analysis, 2026-07):
    The ACME datasheet family -- the bulk of the corpus -- draws spec tables
    with NO left border line and often no per-row separators; the only left
    vertical edge on a row comes from that row's zebra shading rectangle, which
    exists on alternating rows only. pdfplumber's lattice detector needs a
    CLOSED cell per row, so on the unshaded rows it drops the leftmost cell
    (``rows[r].cells[0] is None``) and the row's first word -- the column-0
    label -- vanishes from the table. The same rect-vs-line x drift invents
    phantom columns. The net effect was ~40% of tables shipping with the
    column-0 label or the whole header missing.

    Every one of those tables is born-digital: the text and its exact position
    are in the PDF's own text layer, and the row/column structure is present as
    deterministic geometry (fill rectangles for row bands, vertical line
    segments and rect edges for columns, whitespace channels between columns).
    So the grid can be rebuilt WITHOUT any model, losslessly, by reading that
    geometry directly instead of relying on lattice's closed-cell requirement.

What this does:
    Given a table REGION (pdfplumber's ``find_tables()`` still locates regions
    well; it only mis-reconstructs the grid inside them) plus the page's words
    and vector objects, it derives column boundaries from vertical edges (or,
    when there are none, from persistent whitespace channels) applied UNIFORMLY
    across the whole region -- so a column boundary recovered from a shaded
    row's rectangle also closes the unshaded rows' leftmost cell. Row boundaries
    come from horizontal rules / fill bands when present (which correctly group
    a wrapped multi-line cell into one row) and from visual word lines
    otherwise. Every cell's text is the PDF's own words placed by geometry, so
    nothing is ever invented -- the same "text from the code" guarantee the
    model paths give, but here the STRUCTURE is deterministic too.

    A confidence score and flags are returned so the caller can decide whether
    to trust the band grid outright (the clean, ruled/shaded common case) or
    ask a model to referee (a lineless, ambiguously-aligned table). This module
    never calls a model and never raises on odd input -- it returns None only
    when the region has no placeable words at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from statistics import median

from .base import Merge, TableData, text_cell

# A vertical edge must run through at least this fraction of a row band to
# count as "separating" the two columns on either side of it there; below it,
# the cell spans the boundary (a merge). Guards against a stray short edge
# fragment reading as a full column rule.
_CROSS_MIN = 0.6
# Column boundaries within this many points are the same boundary (rect edges
# and line segments for the same rule land a fraction of a point apart; see the
# 70.9-vs-72.0 drift that produced phantom columns).
_COL_TOL = 3.0
# Row/horizontal-cut clustering tolerance.
_ROW_TOL = 2.0
# A header line sitting in a shaded band above the region's top ruling line is
# outside find_tables()' bbox; scan up to this far above the region for it.
_HEADER_GAP_MAX = 30.0


@dataclass
class BandTable:
    """A deterministically rebuilt table.

    bbox
        The region in PDF points, possibly extended upward from the input
        region to include a recovered header band (see `_detect_header`).
    confidence
        0..1 estimate of how well the geometry pinned the grid -- high for a
        ruled/shaded table whose words partition cleanly, lower when columns
        had to be guessed from whitespace or a word straddled a boundary. The
        caller uses this to decide whether a model referee is worth invoking.
    flags
        Human-readable notes on what was uncertain ("cols-from-gaps",
        "word-straddle", ...), carried through to the IR for transparency.
    """

    data: TableData
    bbox: tuple[float, float, float, float]
    confidence: float
    flags: list[str] = field(default_factory=list)


def build_band_table(page, region_bbox: tuple[float, float, float, float],
                     words: list[dict]) -> BandTable | None:
    """Rebuild the grid of the table in `region_bbox` (PDF points, top-based)
    from page geometry and `words` (pdfplumber ``extract_words`` dicts).
    Returns None only if the region holds no placeable words.
    """
    rx0, rtop, rx1, rbot = region_bbox
    region_words = [w for w in words
                    if w["text"].strip()
                    and _center_in(w, rx0, rtop, rx1, rbot)]
    if not region_words:
        return None

    v_edges = _vertical_edges(page, rx0, rtop, rx1, rbot)
    h_edges = _horizontal_edges(page, rx0, rtop, rx1, rbot)
    fills = _fill_rects(page, rx0, rtop, rx1, rbot)

    # -- header pull-up: a shaded-band header sits above the lattice bbox ----
    flags: list[str] = []
    header_words, header_top = _detect_header(words, region_bbox, region_words)
    if header_words:
        region_words = header_words + region_words
        rtop = header_top
        flags.append("header-recovered")
        v_edges = _vertical_edges(page, rx0, rtop, rx1, rbot)
        h_edges = _horizontal_edges(page, rx0, rtop, rx1, rbot)
        fills = _fill_rects(page, rx0, rtop, rx1, rbot)

    # -- rows ----------------------------------------------------------------
    lines = _group_lines(region_words)
    row_bounds, row_conf, row_flags = _row_boundaries(lines, h_edges, fills,
                                                       rtop, rbot)
    flags += row_flags

    # -- columns -------------------------------------------------------------
    col_bounds, col_source, col_conf = _column_boundaries(
        region_words, lines, v_edges, rx0, rx1)
    if col_source == "gaps":
        flags.append("cols-from-gaps")
    elif col_source == "single":
        flags.append("single-column")

    # -- placement + merges --------------------------------------------------
    data, place_clean, place_flags = _fill_grid(
        region_words, row_bounds, col_bounds, v_edges, col_source)
    flags += place_flags
    if "header-recovered" in flags:
        # The shaded-band line pulled in above the lattice IS the header row
        # (that's the only thing _detect_header accepts), so row 0's role is
        # known here, not inferred.
        data.header_rows = 1

    confidence = round(0.4 * col_conf + 0.35 * row_conf + 0.25 * place_clean, 3)
    return BandTable(data=data, bbox=(rx0, rtop, rx1, rbot),
                     confidence=confidence, flags=flags)


# ---------------------------------------------------------------------------
# Geometry collection
# ---------------------------------------------------------------------------


def _center_in(w: dict, x0: float, top: float, x1: float, bot: float) -> bool:
    cx = (w["x0"] + w["x1"]) / 2.0
    cy = (w["top"] + w["bottom"]) / 2.0
    return x0 <= cx <= x1 and top <= cy <= bot


def _overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _vertical_edges(page, rx0, rtop, rx1, rbot) -> list[tuple[float, float, float]]:
    """(x, top, bottom) of vertical edges (line segments AND rect edges) whose
    span overlaps the region's y-range and whose x lies within it. pdfplumber's
    ``page.edges`` already decomposes both lines and rectangles into oriented
    edges, so this one source covers ruled columns and shaded-cell edges."""
    out = []
    for e in page.edges:
        if e.get("orientation") != "v":
            continue
        x = float(e["x0"])
        if not (rx0 - _COL_TOL <= x <= rx1 + _COL_TOL):
            continue
        top, bot = float(e["top"]), float(e["bottom"])
        if _overlap_len(top, bot, rtop, rbot) <= 0:
            continue
        out.append((x, top, bot))
    return out


def _horizontal_edges(page, rx0, rtop, rx1, rbot) -> list[float]:
    """Y positions of row-delimiting rules. A rule is frequently drawn as one
    segment PER COLUMN (ACME tables do this), so no single segment spans the
    table; collinear segments at the same y must be unioned before judging
    coverage, or every rule is wrongly discarded as "too short". Returns the y
    of each rule whose unioned segments cover at least half the region width."""
    width = rx1 - rx0
    segs: list[tuple[float, float, float]] = []
    for e in page.edges:
        if e.get("orientation") != "h":
            continue
        y = float(e["top"])
        if not (rtop - _ROW_TOL <= y <= rbot + _ROW_TOL):
            continue
        a, b = max(float(e["x0"]), rx0), min(float(e["x1"]), rx1)
        if b - a <= 0:
            continue
        segs.append((y, a, b))
    segs.sort()
    out: list[float] = []
    groups: list[list[tuple[float, float, float]]] = []
    for s in segs:
        if groups and s[0] - groups[-1][0][0] <= _ROW_TOL:
            groups[-1].append(s)
        else:
            groups.append([s])
    for g in groups:
        cov = _union_len([(a, b) for _, a, b in g])
        if width <= 0 or cov >= 0.5 * width:
            out.append(sum(s[0] for s in g) / len(g))
    return out


def _union_len(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by a set of (start, end) intervals (overlaps
    counted once)."""
    total = 0.0
    cur_a = cur_b = None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _fill_rects(page, rx0, rtop, rx1, rbot) -> list[tuple[float, float]]:
    """(top, bottom) of filled rectangles overlapping the region -- the zebra
    shading bands that delimit rows when there are no per-row rules."""
    out = []
    width = rx1 - rx0
    for r in page.rects:
        if not r.get("fill"):
            continue
        if width > 0 and _overlap_len(float(r["x0"]), float(r["x1"]),
                                      rx0, rx1) < 0.4 * width:
            continue
        top, bot = float(r["top"]), float(r["bottom"])
        if _overlap_len(top, bot, rtop, rbot) <= 0:
            continue
        out.append((top, bot))
    return out


# ---------------------------------------------------------------------------
# Lines / clustering
# ---------------------------------------------------------------------------


# Maximum horizontal gap between two adjacent word boxes on the same line,
# as a fraction of the (smaller) font size, for them to still be ONE printed
# word. extract_words(extra_attrs=[...]) breaks a word at every attribute
# change -- 'support@' (regular) + 'example.com' (bold) come back as two
# "words" with a ~0 gap -- and joining such halves with " " injects a space
# the page never printed (observed in the wild as
# "support@ example.com"). A real inter-word space is ~0.25 em; kerning-scale
# splits sit well under 0.15 em.
_TIGHT_JOIN_FRAC = 0.15


def word_sep(prev: dict, w: dict) -> str:
    """Separator to place between two consecutive word boxes when joining
    them into text: "" when they sit on the same line and (near-)touch --
    i.e. they are two halves of one printed word split at an attribute
    change -- else a single space. Shared by the line builder (pdf_parser)
    and the cell filler below, so both fix that defect the same way."""
    same_line = (min(prev["bottom"], w["bottom"])
                 - max(prev["top"], w["top"])) > 0
    if not same_line:
        return " "
    size = min(float(prev.get("size") or 10.0), float(w.get("size") or 10.0))
    return "" if (w["x0"] - prev["x1"]) < _TIGHT_JOIN_FRAC * size else " "


def join_words(words: list[dict]) -> str:
    """`words` (already in reading order) joined into one string using
    `word_sep` between each adjacent pair."""
    parts: list[str] = []
    prev: dict | None = None
    for w in words:
        parts.append((word_sep(prev, w) if prev is not None else "") + w["text"])
        prev = w
    return "".join(parts)


def _group_lines(words: list[dict]) -> list[list[dict]]:
    """Words grouped into visual lines by `top` proximity, each sorted by x0.
    Self-contained (no import from pdf_parser, which imports this module)."""
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        size = float(w.get("size") or 10.0)
        if cur and w["top"] - cur[0]["top"] > max(2.0, 0.5 * size):
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    for g in lines:
        g.sort(key=lambda w: w["x0"])
    return lines


def _cluster(vals: list[float], tol: float) -> list[float]:
    """Collapse near-equal values into their group means (sorted)."""
    if not vals:
        return []
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


# ---------------------------------------------------------------------------
# Row boundaries
# ---------------------------------------------------------------------------


def _row_boundaries(lines: list[list[dict]],
                    h_edges: list[float],
                    fills: list[tuple[float, float]],
                    rtop: float, rbot: float
                    ) -> tuple[list[float], float, list[str]]:
    """Row boundary y's (len = n_rows + 1), a confidence, and flags.

    Prefer geometric cuts (horizontal rules + fill-band edges) when they exist
    AND every visual line falls entirely inside one band: that geometry groups
    a wrapped multi-line cell into a single row, which pure line-clustering
    would wrongly split. Otherwise fall back to midpoints between visual lines.
    """
    flags: list[str] = []
    line_spans = [(min(w["top"] for w in ln), max(w["bottom"] for w in ln))
                  for ln in lines]

    cut_vals = list(h_edges)
    for top, bot in fills:
        cut_vals += [top, bot]
    cuts = _cluster([c for c in cut_vals if rtop - _ROW_TOL <= c <= rbot + _ROW_TOL],
                    _ROW_TOL)

    if len(cuts) >= 3:
        bounds = cuts[:]
        if bounds[0] > rtop + _ROW_TOL:
            bounds.insert(0, rtop)
        if bounds[-1] < rbot - _ROW_TOL:
            bounds.append(rbot)
        if _lines_fit_bands(line_spans, bounds):
            return bounds, 1.0, flags
        flags.append("rows-cuts-inconsistent")

    # Fall back to visual lines: boundary = midpoint between adjacent lines.
    if not line_spans:
        return [rtop, rbot], 0.3, ["rows-no-lines"]
    bounds = [rtop]
    for (_, prev_bot), (next_top, _) in pairwise(line_spans):
        bounds.append((prev_bot + next_top) / 2.0)
    bounds.append(rbot)
    conf = 0.7 if _regular_spacing(line_spans) else 0.45
    flags.append("rows-from-words")
    return bounds, conf, flags


def _lines_fit_bands(line_spans: list[tuple[float, float]],
                     bounds: list[float]) -> bool:
    """Every visual line lies within a single (bounds[i], bounds[i+1]) band,
    and no band is empty -- i.e. the geometric cuts and the text agree."""
    if len(bounds) < 2:
        return False
    used = set()
    for top, bot in line_spans:
        mid = (top + bot) / 2.0
        band = next((i for i in range(len(bounds) - 1)
                     if bounds[i] - _ROW_TOL <= mid <= bounds[i + 1] + _ROW_TOL), None)
        if band is None:
            return False
        # the line must not straddle this band's own boundaries
        if top < bounds[band] - _ROW_TOL or bot > bounds[band + 1] + _ROW_TOL:
            return False
        used.add(band)
    return len(used) == len(bounds) - 1


def _regular_spacing(line_spans: list[tuple[float, float]]) -> bool:
    if len(line_spans) < 3:
        return True
    centers = [(t + b) / 2.0 for t, b in line_spans]
    gaps = [b - a for a, b in pairwise(centers)]
    m = median(gaps)
    return m > 0 and all(abs(g - m) <= 0.5 * m for g in gaps)


# ---------------------------------------------------------------------------
# Column boundaries
# ---------------------------------------------------------------------------


def _column_boundaries(region_words: list[dict], lines: list[list[dict]],
                       v_edges: list[tuple[float, float, float]],
                       rx0: float, rx1: float
                       ) -> tuple[list[float], str, float]:
    """Column boundary x's (len = n_cols + 1), the source used, a confidence.

    Vertical edges are authoritative when present: clustered across the WHOLE
    region, a boundary recovered from any one row's shaded-cell edge closes
    that column on every row -- the fix for lattice dropping unshaded rows'
    column-0 cell. Their partition is accepted only if words don't straddle the
    interior boundaries; otherwise (or when there are no interior edges) fall
    back to persistent whitespace channels between the words.
    """
    edge_xs = _cluster([x for x, _, _ in v_edges], _COL_TOL)
    edge_bounds = _coalesce_empty_columns(
        _bounds_from_interior(edge_xs, rx0, rx1), region_words)
    if len(edge_bounds) >= 3 and _partitions_cleanly(region_words, edge_bounds):
        return edge_bounds, "edges", 1.0

    gap_bounds = _coalesce_empty_columns(_gap_boundaries(lines, rx0, rx1),
                                         region_words)
    if len(gap_bounds) >= 3:
        conf = 0.6 if len(edge_bounds) < 3 else 0.5
        return gap_bounds, "gaps", conf

    # Nothing separated the words -- a single-column strip (or a table we can't
    # resolve). One column, low confidence, let a referee/flag handle it.
    return [rx0, rx1], "single", 0.2


def _coalesce_empty_columns(bounds: list[float], region_words: list[dict]
                            ) -> list[float]:
    """Fold away columns that hold NO word in any row. The ACME template draws
    each shaded cell as its own padded rectangle, so between two data columns
    sit one or two hair-thin strips bounded by rect edges but containing no
    text; left in, they explode a 5-column table into 15 (the phantom-column
    bug). A column with no word centre in it carries nothing, so merging it into
    its neighbour is lossless. Genuine dash cells ("-"/"―") are real words and
    keep their column."""
    if len(bounds) <= 2:
        return bounds
    n = len(bounds) - 1
    occ = [False] * n
    for w in region_words:
        cx = (w["x0"] + w["x1"]) / 2.0
        occ[_col_of(cx, bounds)] = True
    new = [bounds[0]]
    for c in range(n):
        if occ[c]:
            new.append(bounds[c + 1])
    if len(new) == 1:  # nothing occupied (shouldn't happen) -> leave as-is
        return bounds
    new[-1] = bounds[-1]  # snap final boundary to region right (absorb trailing empties)
    return new


def _bounds_from_interior(interior_xs: list[float], rx0: float, rx1: float
                          ) -> list[float]:
    """Turn interior boundary x's into a full boundary list by adding the
    region's left/right, dropping any interior x within tolerance of an edge."""
    interior = [x for x in interior_xs if rx0 + _COL_TOL < x < rx1 - _COL_TOL]
    return _cluster([rx0] + interior + [rx1], _COL_TOL)


def _partitions_cleanly(region_words: list[dict], bounds: list[float]) -> bool:
    """No word's x-extent straddles an interior boundary by more than tol."""
    interior = bounds[1:-1]
    for w in region_words:
        for b in interior:
            if w["x0"] + _COL_TOL < b < w["x1"] - _COL_TOL:
                return False
    return True


def _gap_boundaries(lines: list[list[dict]], rx0: float, rx1: float,
                    nbins: int = 400) -> list[float]:
    """Interior column boundaries from vertical whitespace channels that are
    empty across (almost) all rows. A gap present in only one row -- a short
    cell's trailing space -- is not a column; requiring it across most rows
    filters those out. Boundary = channel centre."""
    width = rx1 - rx0
    if width <= 0 or not lines:
        return [rx0, rx1] if width > 0 else []
    counts = [0] * nbins
    for ln in lines:
        covered = [False] * nbins
        for w in ln:
            b0 = _clip(int((w["x0"] - rx0) / width * nbins), nbins)
            b1 = _clip(int((w["x1"] - rx0) / width * nbins), nbins)
            for b in range(b0, b1 + 1):
                covered[b] = True
        for b in range(nbins):
            if covered[b]:
                counts[b] += 1
    nrows = len(lines)
    occupied_floor = max(0, round(0.1 * nrows))  # a channel is empty in ~all rows
    min_run = max(3, int(0.012 * nbins))
    is_gap = [counts[b] <= occupied_floor for b in range(nbins)]

    bounds = [rx0]
    run_start = None
    for b in range(nbins):
        if is_gap[b]:
            if run_start is None:
                run_start = b
        else:
            if run_start is not None:
                _emit_channel(bounds, run_start, b - 1, run_start, b - 1,
                              min_run, rx0, width, nbins)
                run_start = None
    # trailing run ignored (region's right edge, not an interior channel)
    bounds.append(rx1)
    return _cluster(bounds, _COL_TOL)


def _emit_channel(bounds, start, end, _rs, _re, min_run, rx0, width, nbins):
    # A leading gap run (touching bin 0) is the region's left margin, not an
    # interior separator; only interior runs become boundaries.
    if start == 0 or end >= nbins - 1:
        return
    if end - start + 1 < min_run:
        return
    bounds.append(rx0 + ((start + end + 1) / 2.0) / nbins * width)


def _clip(b: int, nbins: int) -> int:
    return max(0, min(nbins - 1, b))


# ---------------------------------------------------------------------------
# Placement + merges
# ---------------------------------------------------------------------------


def _col_of(cx: float, bounds: list[float]) -> int:
    for c in range(len(bounds) - 1):
        if bounds[c] <= cx < bounds[c + 1]:
            return c
    return len(bounds) - 2  # right edge inclusive


def _fill_grid(region_words: list[dict], row_bounds: list[float],
               col_bounds: list[float],
               v_edges: list[tuple[float, float, float]], col_source: str
               ) -> tuple[TableData, float, list[str]]:
    """Assign each word to (row, col) by centre, merge columns a row's ruling
    doesn't separate (a full-width title band), and build the TableData. Text
    is only ever the PDF's own words. Returns (data, place_clean, flags)."""
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1
    flags: list[str] = []

    # word index buckets per (row, col)
    buckets: dict[tuple[int, int], list[dict]] = {}
    straddle = 0
    interior = col_bounds[1:-1]
    for w in region_words:
        cy = (w["top"] + w["bottom"]) / 2.0
        cx = (w["x0"] + w["x1"]) / 2.0
        r = _row_of(cy, row_bounds)
        c = _col_of(cx, col_bounds)
        buckets.setdefault((r, c), []).append(w)
        if any(w["x0"] + _COL_TOL < b < w["x1"] - _COL_TOL for b in interior):
            straddle += 1

    cells: list[list] = [[None] * n_cols for _ in range(n_rows)]
    merges: list[Merge] = []
    covered: set[tuple[int, int]] = set()

    for r in range(n_rows):
        occupied = sorted(c for c in range(n_cols) if buckets.get((r, c)))
        if not occupied:
            continue
        # group occupied columns into spans separated by a crossing vertical
        # rule; columns with no rule between them (and no other column between)
        # belong to one spanning cell -- the deterministic merge signal.
        spans = _column_spans(r, occupied, col_bounds, row_bounds, v_edges,
                              col_source, buckets)
        for cs, ce in spans:
            wds = [w for c in range(cs, ce + 1) for w in buckets.get((r, c), [])]
            # reading order within the cell: top-to-bottom, then left-to-right,
            # so a cell whose text wrapped onto several lines reads in order
            # (sorting by x alone would interleave the wrapped lines).
            wds.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
            cells[r][cs] = text_cell(join_words(wds))
            if ce > cs:
                merges.append(Merge(row=r, col=cs, rowspan=1, colspan=ce - cs + 1))
                for c in range(cs + 1, ce + 1):
                    covered.add((r, c))

    for r in range(n_rows):
        for c in range(n_cols):
            if cells[r][c] is None and (r, c) not in covered:
                cells[r][c] = text_cell("")

    content = [w for w in region_words if any(ch.isalnum() for ch in w["text"])]
    place_clean = 1.0 if not content else 1.0 - straddle / len(content)
    if straddle:
        flags.append("word-straddle")

    data = TableData(n_rows=n_rows, n_cols=n_cols, cells=cells, merges=merges)
    return data, place_clean, flags


def _row_of(cy: float, bounds: list[float]) -> int:
    for r in range(len(bounds) - 1):
        if bounds[r] <= cy < bounds[r + 1]:
            return r
    return len(bounds) - 2


def _column_spans(r: int, occupied: list[int], col_bounds: list[float],
                  row_bounds: list[float],
                  v_edges: list[tuple[float, float, float]], col_source: str,
                  buckets: dict[tuple[int, int], list[dict]]
                  ) -> list[tuple[int, int]]:
    """Group this row's occupied columns into cell spans. Two adjacent occupied
    columns merge into one cell only with TWO pieces of evidence at once: no
    vertical rule crosses this row's band at the boundary between them, AND
    their text actually runs together (a small whitespace gap). The rule test
    alone is not enough -- a header row set in a shaded band above the ruled
    area has no vertical rule crossing it, yet its labels ("Specification",
    "Min", ...) are distinct cells separated by wide gaps; requiring contiguous
    text keeps those apart while still merging a genuine full-width title whose
    words run continuously across the columns. Only attempted when columns came
    from real edges -- with no ruling there is no evidence a span exists."""
    if col_source != "edges" or len(occupied) < 2:
        return [(c, c) for c in occupied]

    y0, y1 = row_bounds[r], row_bounds[r + 1]
    spans: list[tuple[int, int]] = []
    cs = prev = occupied[0]
    for c in occupied[1:]:
        adjacent = (c == prev + 1)  # an empty column between -> never a merge
        ruled = _edge_crosses(col_bounds[c], y0, y1, v_edges)
        contiguous = _text_contiguous(buckets.get((r, prev), []),
                                      buckets.get((r, c), []))
        if adjacent and not ruled and contiguous:
            prev = c
            continue  # extend the current span across the un-ruled boundary
        spans.append((cs, prev))
        cs = prev = c
    spans.append((cs, prev))
    return spans


def _wsize(w: dict) -> float:
    return float(w.get("size") or 10.0)


def _text_contiguous(left: list[dict], right: list[dict]) -> bool:
    """Do the rightmost word of `left` and leftmost of `right` sit close enough
    to be one running phrase (a title spanning columns) rather than two
    column labels separated by table whitespace? Scaled by font size so it
    holds across body/heading text sizes."""
    if not left or not right:
        return False
    gap = min(w["x0"] for w in right) - max(w["x1"] for w in left)
    size = max(_wsize(w) for w in left + right)
    return gap <= 1.5 * size


def _edge_crosses(x: float, y0: float, y1: float,
                  v_edges: list[tuple[float, float, float]]) -> bool:
    band = y1 - y0
    if band <= 0:
        return True
    for ex, etop, ebot in v_edges:
        if abs(ex - x) <= _COL_TOL and _overlap_len(etop, ebot, y0, y1) >= _CROSS_MIN * band:
            return True
    return False


# ---------------------------------------------------------------------------
# Header recovery
# ---------------------------------------------------------------------------


def _detect_header(words: list[dict],
                   region_bbox: tuple[float, float, float, float],
                   region_words: list[dict]
                   ) -> tuple[list[dict], float]:
    """A header row set in a shaded band with no ruling line above it sits just
    OUTSIDE find_tables()' bbox (the row's words are then dropped from the table
    and resurface as a stray heading). Look immediately above the region for the
    nearest visual line whose words span the region's width in more than one
    piece -- a plausible column-label row -- and pull it in as row 0.

    Returns (header_words, new_region_top) or ([], region_top).
    """
    rx0, rtop, rx1, _ = region_bbox
    above = [w for w in words
             if w["text"].strip()
             and rtop - _HEADER_GAP_MAX <= (w["top"] + w["bottom"]) / 2.0 < rtop
             and w["x1"] > rx0 and w["x0"] < rx1]
    if not above:
        return [], rtop
    lines = _group_lines(above)
    if not lines:
        return [], rtop
    # nearest line to the region first
    lines.sort(key=lambda ln: -min(w["top"] for w in ln))
    cand = lines[0]
    if len(cand) < 2:
        return [], rtop  # a single token above is a caption/label, not a header
    # it must sit within the region's horizontal span (a full-width paragraph
    # above the table would spill past it)
    if min(w["x0"] for w in cand) < rx0 - _COL_TOL or \
       max(w["x1"] for w in cand) > rx1 + _COL_TOL:
        return [], rtop
    return cand, min(w["top"] for w in cand)
