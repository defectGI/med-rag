"""HeadingStack breadcrumb correctness: `enter()` must return the heading's
ANCESTORS only -- closed same-or-deeper headings (siblings, previous sections)
are popped BEFORE the snapshot, never carried into the new heading's
`heading_path`. Regression tests for the pre-pop-snapshot bug where a second
level-1 heading inherited the first as its "ancestor"."""

from __future__ import annotations

from medrag.pipeline.parser.parsers.base import HeadingStack


def test_second_top_level_heading_has_no_ancestors():
    s = HeadingStack()
    assert s.enter(1, "1. Description") is None
    assert s.enter(1, "2. Hardware Overview") is None  # was ["1. Description"]


def test_sibling_subheading_is_not_an_ancestor():
    s = HeadingStack()
    s.enter(1, "1")
    assert s.enter(2, "1.1") == ["1"]
    assert s.enter(2, "1.2") == ["1"]  # was ["1", "1.1"]


def test_new_section_closes_the_whole_deeper_chain():
    s = HeadingStack()
    s.enter(1, "1")
    s.enter(2, "1.1")
    s.enter(3, "1.1.1")
    assert s.enter(1, "2") is None     # was ["1", "1.1", "1.1.1"]
    assert s.enter(2, "2.1") == ["2"]


def test_path_for_body_blocks_includes_the_open_heading():
    # `path()` (used for non-heading blocks) is the full open breadcrumb,
    # including the heading the block sits under -- unchanged by the fix.
    s = HeadingStack()
    s.enter(1, "1")
    s.enter(2, "1.2")
    assert s.path() == ["1", "1.2"]
