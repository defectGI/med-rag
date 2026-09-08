"""Judges whether a VLM's raw image transcription is meaningful (real
content, not noise from a logo/icon/decorative graphic), rates how confident
it is that the transcription is accurate, and cleans up spelling/formatting
mistakes in the transcription before it's kept as an ImageBlock's `ocr_text`.

Image OCR is inherently unverified (there is no text layer to align against,
unlike a born-digital PDF page), so the confidence returned here is what
`images/image_handler.py` gates on: an env-configurable threshold decides
whether the text is trustworthy enough to surface as `ocr_text`.
"""

from __future__ import annotations

import json
import re

from medrag.pipeline.parser.llm import LLMClient

_SYSTEM = (
    "You review a raw text transcription of an image. Decide whether the "
    "text is meaningful content (e.g. a sentence, a label, a caption, data) "
    "as opposed to noise from a decorative graphic, logo or icon with no "
    "real textual content. If meaningful, also fix any obvious OCR/spelling "
    "or formatting mistakes without changing the meaning or adding anything "
    "not present in the original. Also rate your confidence, from 0.0 to 1.0, "
    "that the (corrected) transcription accurately reflects the text in the "
    "image -- low when the source looks like a blurry photo, is garbled, or "
    "you are guessing; high when the text is crisp and unambiguous.\n\n"
    "Return ONLY a JSON object, no other text: "
    '{"meaningful": true/false, "cleaned_text": "<corrected text, or "" if not meaningful>", '
    '"confidence": <number between 0.0 and 1.0>}.'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _parse_confidence(value: object) -> float | None:
    """Coerce the model's `confidence` to a float in [0.0, 1.0], or None when
    it's missing/unparseable so the caller can tell "no confidence reported"
    apart from a genuine low score."""
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, conf))


def check_ocr_text(client: LLMClient, raw_text: str) -> tuple[bool, str, float | None]:
    """Return (meaningful, cleaned_text, confidence).

    `confidence` is a float in [0.0, 1.0] rating how accurate the transcription
    is, or None when the model didn't report one (or the reply couldn't be
    parsed). The accept/withhold threshold itself lives in image_handler, not
    here -- this function only reports the verdict.

    On any model/parse failure, treat the text as not meaningful rather than
    risk indexing noise -- the blob and the raw ImageBlock record are kept
    either way (see images/README.md); only `ocr_text` is withheld.
    """
    user = f"Raw transcription:\n{raw_text}\n\nReturn your JSON verdict."
    raw = client.complete(system=_SYSTEM, user=user, max_tokens=500)
    match = _JSON_OBJECT.search(raw)
    if not match:
        return False, "", None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, "", None
    meaningful = bool(parsed.get("meaningful"))
    cleaned = str(parsed.get("cleaned_text") or "").strip()
    confidence = _parse_confidence(parsed.get("confidence"))
    if meaningful and not cleaned:
        return False, "", confidence
    return meaningful, cleaned, confidence
