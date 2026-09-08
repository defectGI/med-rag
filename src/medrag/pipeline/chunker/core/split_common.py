"""Shared span-packing loop for the splitters.

The table splitter and the fallback splitters (paragraph/list/code) solve
the same problem: N ordered units (rows/sentences/items/code lines),
consecutive parts sharing k units, each part grows to fit the budget.
The loop's subtleties (progress guarantee, overlap clipped to part size)
live here in one place — the splitters only answer "does this candidate
fit?".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple


class Span(NamedTuple):
    """A part's unit range: [start, end) units in the part,
    [new_start, end) units appearing for the FIRST time in this part
    (the rest is overlap from the previous part)."""

    start: int
    end: int
    new_start: int

    @property
    def overlap(self) -> int:
        """Units shared with the previous part (0 for the first part)."""
        return self.new_start - self.start


def pack_spans(
    n_units: int,
    overlap: int,
    fits: Callable[[Span], bool],
) -> list[Span]:
    """Split [0, n_units) into the largest parts that `fits` accepts.

    `fits(span)` says whether the candidate fits the budget (the splitter
    renders and counts tokens for it). For `n_units == 0` returns an
    empty list.
    """
    spans: list[Span] = []
    prev_start: int | None = None
    prev_end = 0
    while prev_end < n_units:
        start = 0 if prev_start is None else max(prev_end - overlap, prev_start + 1)
        new_start = prev_end
        end = new_start + 1  # at least 1 new unit — progress guarantee
        while end < n_units and fits(Span(start, end + 1, new_start)):
            end += 1
        spans.append(Span(start, end, new_start))
        prev_start, prev_end = start, end
    return spans