"""The VLM-VISION strategy: describe a visual region from its pixels plus the
document text around it.

The strategy of last resort, for the types that have neither cell data (tables)
nor their own structured XML (native charts, SmartArt): block diagrams,
technical drawings, flowcharts, and rasterized charts arriving as crops from
PDF pages. They are exactly as worth describing as a table -- a block diagram
carries the architecture of a product -- there is just nothing deterministic to
build the description from, so a model has to look at it.

Two things make this more than "classify, but wordier":
  * Context. Classification (`images/visual_classify.py`) shows the model one
    isolated crop; it cannot see the sentence that introduces the diagram or
    the heading naming the part. This pass runs over the FINISHED document, so
    it passes the same surrounding text a table's description gets
    (`describe/context.py`).
  * Type awareness. The label is already known, so the prompt asks for what
    that type actually carries (a flowchart's steps, a drawing's dimensions)
    instead of a generic "describe this image".

`description_source` is always "vlm-vision" here -- the weakest of the three
sources, and marked as such: unlike a chart's XML or a table's cells, there is
no independent ground truth to check the words against. That is also why this
strategy produces no `facts` (see `describe/table.py`): a fact sentence nobody
can verify is a liability, not an asset. What it CAN do (opt-in, `[describe]
visual_check`) is an adversarial verify+retry loop like the table strategy's,
except the checker re-reads the same pixels instead of real cells -- weaker
than a cell check, but enough to catch a description asserting something the
image plainly does not show. If the quality still doesn't hold up on a real
corpus, `[describe] types` drops the type and the content falls back to a
placeholder -- no code change, no re-parse.

Caching: keyed by sha256(crop bytes + context + type + prompt version), under
`storage/labels/` next to the classification cache and for the same reason --
an IR_VERSION or PARSER_VERSION bump re-parses the whole corpus for unrelated
fixes and must not re-pay for a VLM call on an identical crop. The context and
type are part of the key because they are part of the input: the same crop in a
different section deserves a different description. A miss with no VLM
available returns None WITHOUT writing the cache, so a later run with a VLM
still gets a real answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from medrag.pipeline.parser.config import Describe
from medrag.pipeline.parser.llm import LLMClient, LLMError, VLMClient
from medrag.pipeline.parser.storage_paths import labels_dir

__all__ = ["VisualDescription", "describe_crop"]

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Bump when the prompt below changes in a way that should invalidate cached
# descriptions (the cache key includes it -- see module docstring).
# v2: anti-hallucination rewrite -- the type is presented as a GUESS to confirm
# rather than a fact to elaborate, and every type carries an explicit "if you
# can't see it, say so" escape hatch.
_PROMPT_VERSION = "2"

_TYPE_GUIDANCE = {
    "chart": "State the chart type, what is plotted, and the overall trend or "
             "the most notable values. Never invent exact numbers you cannot "
             "read.",
    "block_diagram": "Name the components/blocks shown and how they are "
                     "connected (signal or data flow), so a reader who cannot "
                     "see the diagram understands the structure.",
    "technical_drawing": "State what part or assembly is drawn, the views "
                         "shown, and any labeled dimensions or callouts you "
                         "can read.",
    "flowchart": "State what process is shown and walk through its steps and "
                 "decision points in order.",
    "table": "This is a table shown only as an image (it has no extractable "
             "cells -- a real cell-based table is described elsewhere). State "
             "what the table is about and what its rows and columns represent. "
             "Do NOT reproduce specific cell values, counts or numbers unless "
             "they are clearly legible, and never invent rows, columns or "
             "entries to fill it out.",
    "product_photo": "State what physical object is shown and any visible "
                     "labels, ports or markings.",
    "decorative": "State briefly what the graphic depicts.",
    "unknown": "State what the region shows as plainly as you can. If you "
               "genuinely cannot tell, say so rather than guessing.",
}

_SYSTEM_TEMPLATE = (
    "A region cropped from a technical document has been tentatively "
    "classified as a {label}. Describe what is ACTUALLY in the image, so that "
    "someone who cannot see it can still find and understand it via search.\n"
    "The label is only a guess: if the image is not really a {label} (for "
    "example it is a logo, an icon, a photo, or just a piece of text), say "
    "what you actually see instead -- do not force the description to fit the "
    "label.\n"
    "If it IS a {label}: {guidance}\n"
    "Rules:\n"
    "- Plain text only: no markdown, no bullet points, no preamble or quotes.\n"
    "- Write in the same language as the surrounding context.\n"
    "- 1 to 4 sentences, at most about 80 words.\n"
    "- Describe ONLY what you can actually see. Never invent values, labels, "
    "component names, views or conclusions. If a detail is unreadable, omit it "
    "rather than guessing.\n"
    "- If the image is too small, blurry or empty to make out, do not "
    "manufacture a plausible-sounding description of what such a {label} "
    "usually contains. Say plainly that the content cannot be made out, in one "
    "short sentence, and stop.\n"
    "- Use the surrounding context only to identify the subject; do not repeat "
    "the context back as if it were the content, and never state something is "
    "present just because the context mentions it."
)

# The adversarial verify pass (opt-in, `[describe] visual_check`). A SECOND
# vision call that sees the same image plus the candidate description and looks
# for anything the description asserts that the image does not actually show --
# the visual analogue of describe/table.py's checker, except a table's checker
# reads real cells and this one re-reads the pixels. Deliberately biased toward
# rejection: the whole point is to catch confident fabrication, so an assertion
# that cannot be confirmed from the image is treated as unfaithful.
_VERIFY_SYSTEM = (
    "You verify a candidate description of an image region taken from a "
    "technical document. You can see the image. Judge ONE thing: is every "
    "statement in the description actually supported by what is visible in the "
    "image?\n"
    "Reject (faithful=false) if the description asserts any object, label, "
    "value, connector name, view, component or conclusion that you cannot "
    "confirm from the image itself, or if it describes the wrong kind of thing "
    "(e.g. calls a logo a block diagram). A description that honestly says the "
    "content cannot be made out is faithful. Be strict: if in doubt, reject.\n"
    'Return ONLY a JSON object, no other text: {"faithful": true/false, '
    '"reason": "<short reason if false>"}.'
)


@dataclass
class VisualDescription:
    """Outcome of describing one crop: the text plus the verify+retry bookkeeping
    the shared describe fields expect (`describe_status`/`describe_attempts` in
    parsers/base.py). `status` is None when the verify pass was off (matching a
    table described with `llm_check=False`), else "ok" | "flagged"; "empty" when
    nothing describable was produced."""

    description: str | None
    status: str | None = None
    attempts: int | None = None


def _cache_key(data: bytes, context: str, visual_type: str) -> str:
    h = hashlib.sha256()
    h.update(data)
    # Length-prefixed so ("ab", "c") and ("a", "bc") can't collide.
    for part in (context, visual_type, _PROMPT_VERSION):
        raw = part.encode("utf-8")
        h.update(str(len(raw)).encode("ascii"))
        h.update(b"\0")
        h.update(raw)
    return h.hexdigest()


def _cache_path(sha: str) -> Path:
    return labels_dir() / f"desc-{sha}.json"


def _cache_read(sha: str) -> VisualDescription | None:
    path = _cache_path(sha)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    description = data.get("description")
    if not (isinstance(description, str) and description):
        return None
    return VisualDescription(description=description,
                             status=data.get("status"),
                             attempts=data.get("attempts"))


def _cache_write(sha: str, result: VisualDescription,
                 model: str | None = None) -> None:
    root = labels_dir()
    root.mkdir(parents=True, exist_ok=True)
    payload = {"description": result.description, "source": "vlm-vision",
               "model": model, "status": result.status,
               "attempts": result.attempts}
    tmp = root / f"desc-{sha}.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _cache_path(sha))


def _generate(vlm: VLMClient | LLMClient, system: str, mime: str, data: bytes,
              context: str, hint: str, max_tokens: int) -> str:
    user = ""
    if context:
        user += f"Context around it in the document:\n{context}\n\n"
    user += "Write the description now."
    if hint:
        user += (f"\n\nA previous attempt was rejected for this reason: {hint}\n"
                 "Write a new description that fixes it. If the reason is that "
                 "something is not actually visible, remove that claim rather "
                 "than rewording it.")
    return (vlm.complete_vision(system=system, user=user, images=[(mime, data)],
                                max_tokens=max_tokens) or "").strip()


def _verify(vlm: VLMClient | LLMClient, mime: str, data: bytes,
            description: str, max_tokens: int) -> tuple[bool, str]:
    """Adversarial faithfulness check against the image. Returns (faithful,
    reason). A malformed/failed verdict is treated as NOT faithful (fail toward
    rejection) so a broken checker never rubber-stamps a fabrication."""
    user = (f"Candidate description:\n{description}\n\n"
            "Look at the image and return your JSON verdict.")
    try:
        raw = vlm.complete_vision(system=_VERIFY_SYSTEM, user=user,
                                  images=[(mime, data)], max_tokens=max_tokens)
    except LLMError:
        return False, "verify call failed"
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return False, "checker returned no JSON"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, "checker returned invalid JSON"
    return bool(parsed.get("faithful")), str(parsed.get("reason") or "").strip()


def describe_crop(data: bytes, mime: str, visual_type: str, context: str,
                  vlm: VLMClient | LLMClient | None,
                  cfg: Describe) -> VisualDescription:
    """Describe one visual region from its pixels + `context`, with an optional
    adversarial verify+retry loop (`[describe] visual_check`).

    Returns a `VisualDescription`. `description` is None when nothing could be
    produced -- no VLM configured/reachable (fail-open: the block keeps its type
    and placeholder, exactly as before this pass existed) or an empty reply. A
    cache hit returns the stored result (including its verify status) without a
    model call.

    The verify pass is the visual analogue of describe/table.py's checker: a
    second vision call re-reads the image and flags any statement the pixels do
    not support, and the rejection reason is fed back into the next attempt.
    Unlike a table's checker there is no ground-truth cell data, so the checker
    is the same modality (vision) and is deliberately biased toward rejection.
    When `visual_check` is off, behavior is identical to before: one call,
    `status`/`attempts` left None (matching a table with `llm_check=False`).
    """
    sha = _cache_key(data, context, visual_type)
    cached = _cache_read(sha)
    if cached is not None:
        return cached
    if vlm is None:
        return VisualDescription(description=None)

    label = visual_type.replace("_", " ")
    system = _SYSTEM_TEMPLATE.format(
        label=label,
        guidance=_TYPE_GUIDANCE.get(visual_type, _TYPE_GUIDANCE["unknown"]))

    use_check = cfg.visual_check
    max_attempts = max(1, cfg.visual_check_retries) if use_check else 1
    hint = ""
    description = ""
    status: str | None = None
    attempts = 0
    passed = False
    while attempts < max_attempts:
        attempts += 1
        try:
            description = _generate(vlm, system, mime, data, context, hint,
                                    cfg.max_tokens)
        except LLMError:
            # No usable pixels read this attempt -- don't cache, so a later run
            # with a working VLM still gets a real answer (module docstring).
            return VisualDescription(description=None)
        if not description:
            break
        if not use_check:
            break
        passed, reason = _verify(vlm, mime, data, description, cfg.max_tokens)
        if passed:
            status = "ok"
            break
        status = "flagged"
        hint = reason

    if not description:
        # Empty reply. NOT cached (same as before this loop existed): an empty
        # answer may be transient, so a later run should get to re-ask rather
        # than be pinned to "empty" forever. `status="empty"` is still reported
        # for this run so a consumer/telemetry sees it.
        return VisualDescription(description=None,
                                 status="empty" if use_check else None,
                                 attempts=attempts if use_check else None)

    result = VisualDescription(description=description, status=status,
                               attempts=attempts if use_check else None)
    _cache_write(sha, result)
    return result
