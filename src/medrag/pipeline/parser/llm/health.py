"""Known-answer health probes for the configured LLM/VLM endpoints.

A model endpoint can be "up" (HTTP 200) yet unusable — a reasoning model
burning its whole token budget on hidden thinking, a truncated multimodal
prompt, a wrong model name silently served by a router. Transport checks
can't see any of that; only a round-trip with a KNOWN answer can. These
probes cost one tiny request each, so a pipeline can afford to run them at
startup, between phases, and whenever mid-run failures start clustering.

Both return (ok, detail): `detail` is the reply excerpt on failure (or the
transport error), so the caller can print WHAT the endpoint actually said
instead of a bare boolean.
"""

from __future__ import annotations

import io

from .base import LLMClient, LLMError, VLMClient

# The VLM canary transcribes this string off a synthetic image. Digits are
# the checked part: they survive case/spacing variations in the reply.
_CANARY_TEXT = "DOCRAG 4137"
_CANARY_DIGITS = "4137"

_LLM_SYSTEM = "You are a health check. Follow the instruction exactly."
_LLM_USER = "Reply with exactly: PONG"

_VLM_SYSTEM = "You transcribe text from images exactly."
_VLM_USER = "Transcribe the text in the attached image. Reply with the text only."


def probe_llm(client: LLMClient) -> tuple[bool, str]:
    """One tiny known-answer round trip against a text model."""
    try:
        reply = client.complete(system=_LLM_SYSTEM, user=_LLM_USER,
                                max_tokens=64)
    except LLMError as exc:
        return False, f"transport: {exc}"
    if reply and "pong" in reply.lower():
        return True, ""
    return False, f"unusable reply: {reply.strip()[:120]!r}"


def probe_vlm(client: VLMClient) -> tuple[bool, str]:
    """One tiny known-answer round trip against a vision model."""
    try:
        png = _canary_png()
    except Exception as exc:  # noqa: BLE001 -- Pillow missing/broken: not a model error; the reason reaches the caller as a "probe skipped ..." message
        return True, f"probe skipped (cannot build canary image: {exc})"
    try:
        reply = client.complete_vision(
            system=_VLM_SYSTEM, user=_VLM_USER,
            images=[("image/png", png)], max_tokens=64)
    except LLMError as exc:
        return False, f"transport: {exc}"
    if reply and _CANARY_DIGITS in reply:
        return True, ""
    return False, f"unusable reply: {reply.strip()[:120]!r}"


def _canary_png() -> bytes:
    """A small, high-contrast image of `_CANARY_TEXT` any working VLM reads.

    Drawn with PIL's built-in bitmap font then upscaled with NEAREST, so no
    font files are needed and the glyphs stay crisp."""
    from PIL import Image, ImageDraw

    small = Image.new("L", (80, 16), color=255)
    ImageDraw.Draw(small).text((4, 2), _CANARY_TEXT, fill=0)
    big = small.resize((320, 64), Image.NEAREST)
    buf = io.BytesIO()
    big.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
