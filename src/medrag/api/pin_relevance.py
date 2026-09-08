"""Shared decision logic for the fallback branch taken in
`comparison.py`/`doc_download.py`/`recommendation.py`'s `check_pin` when
`_continuation_checker` is NEVER injected (dev/test).

The previous form was "True until proven otherwise": `normalized not in
_CANCEL_WORDS` -- i.e. any message NOT on the cancel-word list (including an
unrelated topic change) kept the pin alive. This module INVERTS that: the pin
stays alive only if (a) the message is non-empty, (b) it is not a cancel word,
AND (c) it CARRIES A SIGNAL matching the slot the flow currently expects --
otherwise the pin DROPS.

Deliberately, it does NOT compute the "signal" itself -- every flow already
has its own deterministic parser (`comparison.py::parse_comparison_query`,
`doc_download.py::parse_doc_download_query`/`extract_doc_type`,
`recommendation.py::parse_family_query` + the numeric-answer regex); instead
of REINVENTING that logic here, the caller pre-computes the signal and passes
it as `has_signal`. It follows continuation.py's "three flows + one shared
service" pattern, but this service is LLM-free/pure -- when
`_continuation_checker` IS injected, this module is never reached (primary
path behavior is unchanged)."""

from __future__ import annotations

from collections.abc import Iterable


def looks_relevant_to_pin(query: str, *, cancel_words: Iterable[str], has_signal: bool) -> bool:
    """An empty message or a cancel word is always `False` -- beyond those, the
    decision rests entirely on the caller's pre-computed `has_signal` (does
    the message contain something matching the slot the pin is waiting for)."""
    normalized = query.strip().lower()
    if not normalized:
        return False
    if normalized in cancel_words:
        return False
    return has_signal


__all__ = ["looks_relevant_to_pin"]
