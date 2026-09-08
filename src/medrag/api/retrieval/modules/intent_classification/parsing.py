"""Defensive parsing of the model output into an intent label + confidence.

Both parsers are lenient by design: a prompt-based classifier over a remote
endpoint can return stray whitespace, quotes, casing, or extra words, and the
endpoint may or may not return logprobs. Nothing here raises on malformed input
— an unparseable label falls back to ``out_of_scope`` and missing logprobs give
``confidence = None``.
"""

from __future__ import annotations

import math
import re
from typing import Any

from medrag.api.retrieval.core import IntentLabel

_VALID = {label.value for label in IntentLabel}
_FALLBACK = IntentLabel.OUT_OF_SCOPE


def parse_label(raw: str) -> IntentLabel:
    """Map a raw model response to an :class:`IntentLabel`.

    Strategy: normalise (lowercase, strip quotes/punctuation), try exact match,
    then look for any valid label token appearing in the text. Fall back to
    ``out_of_scope`` if nothing matches.
    """
    label, _ = parse_label_verbose(raw)
    return label


def parse_label_verbose(raw: str) -> tuple[IntentLabel, bool]:
    """Same as :func:`parse_label` but also reports whether this was a genuine
    fallback: a caller (e.g. for tracing/observability) needs to tell "the model
    actually said out_of_scope" apart from "the model said something unparseable
    and it silently defaulted to out_of_scope" -- the plain ``bool`` return keeps
    :func:`parse_label` itself unchanged for callers that don't need the
    distinction."""
    if not raw:
        return _FALLBACK, True
    text = raw.strip().strip("\"'`.").lower()

    if text in _VALID:
        return IntentLabel(text), False

    # Token/substring scan: pick the first valid label that appears. Sorted by
    # length desc so e.g. "doc_question" wins over a shorter accidental match.
    for candidate in sorted(_VALID, key=len, reverse=True):
        if re.search(rf"\b{re.escape(candidate)}\b", text):
            return IntentLabel(candidate), False

    return _FALLBACK, True


def confidence_from_logprobs(response_metadata: dict[str, Any] | None) -> float | None:
    """Derive a rough confidence from OpenAI-style logprobs, or ``None``.

    Uses the probability the model assigned to the tokens it actually emitted:
    ``exp(sum(token_logprob))``, clamped to [0, 1]. This is approximate (not a
    normalised class posterior) but a useful signal. Returns ``None`` whenever
    logprobs are absent or malformed — never raises. When/if the endpoint's
    logprobs support is confirmed on the inference machine, this can be upgraded
    to renormalise over the label alternatives in ``top_logprobs``.
    """
    if not response_metadata:
        return None
    logprobs = response_metadata.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    try:
        total = 0.0
        seen = False
        for tok in content:
            lp = tok.get("logprob") if isinstance(tok, dict) else None
            if lp is None:
                continue
            total += float(lp)
            seen = True
        if not seen:
            return None
        return max(0.0, min(1.0, math.exp(total)))
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = ["confidence_from_logprobs", "parse_label", "parse_label_verbose"]
