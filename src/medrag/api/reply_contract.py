"""OUTPUT CONTRACT for the answering role -- the root cause behind a set of
live-test failures.

**Why it exists:** the output of every LLM role in this repo has been made
independent of what the model writes around it -- `reconciler` takes what's
between the first ``{`` and the last ``}``, `sqlretrieve` escalates on broken
JSON. The ONE exception was the answering role: `answering_model.py` returned
the raw ``message.content`` directly, and that string went to THREE places at
once -- the user, `ConversationMemory`, and from there (via `orchestrator`'s
`history`) into `reconciler`'s prompt. A single broken turn therefore became a
session-long failure: the model's English internal reasoning was written
verbatim into ``reply``, and in the NEXT two turns the bot denied the full SQL
rows it held.

**Why the root cause is fixed here:** request-side thinking suppression
(`thinking.py`, native ``think:false``) is a REQUEST, not a guarantee -- the
model may still produce reasoning tokens, and if Ollama's template parser
can't separate them into ``message.thinking`` the text stays in ``content``.
So the real guarantee must be on the output side: the model returns a
structured envelope (``{"reply": ..., "cited": [...]}``) and everything
OUTSIDE the envelope -- reasoning, drafts, harmony ``<|channel|>`` markers --
is dropped STRUCTURALLY during parsing. No marker list to chase.

**Layers (defense-in-depth, tried in order):**

1. ``extract_json_object`` -- find the envelope. On success leakage is
   structurally impossible (the EXACT pattern `reconciler` has used since its
   inception; that function now calls THIS -- no two copies).
2. ``strip_reasoning`` -- no envelope, last resort: if a known "the real
   answer starts here" marker exists (``</think>``,
   ``<|channel|>final<|message|>``, ...), take everything AFTER it. This
   layer is heuristic, hence NOT the primary guarantee.
3. ``is_degenerate`` -- even valid-looking text may be degenerate
   (repetition-loop): a correct ``COUNT=0`` answered with hundreds of
   repetitions. This is a RETRY signal; it never silently reaches the user.

This module is DELIBERATELY pure stdlib and imports nothing from `chatbot` --
`reconciler` already imports from `answering_model` (see that file's
``from medrag.api.answering_model import ProviderError, _post_json`` line), so a
reverse import would create a cycle. Both importing this shared module makes
the cycle impossible from the start.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# --- layer 1: structured envelope -------------------------------------------

# Note telling the model the envelope's shape. `_build_messages` appends this
# as a SEPARATE system message AFTER the user turn -- the same placement
# principle proven for `_LANGUAGE_REMINDER` against embedded-instruction
# weakness. The guarantee that `_LANGUAGE_REMINDER` is the LAST message is NOT
# broken: this note comes before it.
REPLY_CONTRACT_NOTE = (
    "OUTPUT FORMAT -- this is a hard requirement, not a preference.\n"
    "Return ONLY a single JSON object, nothing before or after it:\n"
    '{"reply": "<your full answer to the user>", "cited": ["<evidence id>", ...]}\n'
    "- `reply` is the ONLY text the user ever sees. Write it exactly as you "
    "would have written the answer normally -- same language, same markdown "
    "(tables/bullets) as instructed above. Escape newlines as \\n and any "
    'double quotes as \\" so the JSON stays valid.\n'
    "- `cited` lists the evidence ids you actually relied on, copied verbatim "
    "from the [SQL][<id>] / [DOC][<id>] tags in the sources above. If you "
    "relied on no source, use an empty list. NEVER invent an id that is not "
    "in the sources.\n"
    "- Put NO reasoning, drafts, self-corrections, or planning notes anywhere "
    "in the output -- not inside `reply`, not outside the JSON object. Think "
    "silently, then emit only the JSON."
)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def extract_json_object(text: str) -> dict | None:
    """Extracts and parses the first ``{...}`` block from the model response;
    None if absent.

    Drops the markdown fence, then tries between the first ``{`` and the last
    ``}`` -- everything BEFORE the block (reasoning/drafts) and AFTER it is
    dropped. Structural immunity to leakage comes exactly from here: no need
    to guess a marker list to cut on.

    `reconciler._extract_json` used this logic already; once the answering
    role got the same guarantee too, the function moved here (one
    implementation, two consumers)."""
    if not text:
        return None
    stripped = _FENCE_RE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# --- layer 2: last-resort marker cleanup ------------------------------------

# Markers meaning "the real answer starts AFTER HERE". Everything after the
# LAST occurrence is taken (the model may produce several reasoning blocks).
# This list is HEURISTIC and a new model may bring a new format -- which is
# why the primary guarantee is layer 1 (the JSON envelope); this only kicks in
# as a fallback when no envelope arrives at all.
_FINAL_MARKERS = (
    "<|channel|>final<|message|>",  # harmony (gpt-oss) -- the real-answer channel
    "<|end_of_thought|>",
    "</think>",
    "</thinking>",
    "</reasoning>",
    "<channel|>",  # broken/partial form seen in live testing
)

# Leftover harmony/special tokens (e.g. `<|start|>`, `<|message|>`) -- they can
# remain in the tail even after a marker is found.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>\n]{0,40}\|>")


# For leak DETECTION (not cleanup): merely PRESENT in the text is enough for a
# reasoning/channel marker. `preflight.py` uses this -- it used to search only
# for the literal `<think` and could NOT see the `<|channel|>`/harmony forms
# seen in live testing (root-cause finding: preflight printed `OK (thinking
# off)` while real turns leaked).
_LEAK_MARKERS = _FINAL_MARKERS + (
    "<think",
    "<thinking",
    "<reasoning",
    "<|channel|>",
    "<|start|>",
    "<|message|>",
    "<|end_of_thought|>",
)


def has_reasoning_marker(text: str) -> str | None:
    """Returns the reasoning/channel marker if one occurs in the text, else
    None.

    A DIFFERENT job from `strip_reasoning`: that one answers "where does the
    real answer start" (only closing/transition markers), this one answers "is
    there any leak at all" (opening markers included). Preflight's detect-only
    roles (linking/sql) and the verification of checkable roles use this."""
    if not text:
        return None
    lowered = text.lower()
    for marker in _LEAK_MARKERS:
        if marker.lower() in lowered:
            return marker
    return None


def strip_reasoning(text: str) -> tuple[str, str | None]:
    """Returns (cleaned text, marker used).

    If a known "real answer starts here" marker exists, takes everything after
    its LAST occurrence; if no marker at all, the text is UNTOUCHED
    (``marker=None``) -- we can't know where the reasoning ends, and guessing
    by cutting the user's real answer would be worse than a leak. In both
    cases leftover special tokens are cleaned."""
    if not text:
        return "", None
    best_end = -1
    best_marker: str | None = None
    for marker in _FINAL_MARKERS:
        idx = text.rfind(marker)
        if idx >= 0 and idx + len(marker) > best_end:
            best_end = idx + len(marker)
            best_marker = marker
    body = text[best_end:] if best_marker is not None else text
    return _SPECIAL_TOKEN_RE.sub("", body).strip(), best_marker


# --- layer 3: degenerate output detection ------------------------------------

# A 3-60 char unit repeated 20+ times back to back. The threshold is
# deliberately HIGH: the goal is not to misjudge a legitimate table (e.g. all
# 12 rows saying "not specified") as degenerate. The observed real case was
# hundreds of repetitions -- far above the 20 threshold.
_REPEAT_RE = re.compile(r"(.{3,60}?)\1{19,}", re.DOTALL)
_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
_MIN_DEGENERATE_LEN = 200


def is_degenerate(text: str) -> bool:
    """Has the text fallen into a repetition loop?

    Two independent checks: (a) a back-to-back repeated unit -- CONDITIONED on
    the repeated unit containing at least one letter/digit (markdown separator
    rows like `| --- |` consist only of dashes/spaces, so no false positives);
    (b) extremely low word-level diversity. Both are applied only to
    sufficiently LONG text -- repetition in a short answer can be normal."""
    if not text or len(text) < _MIN_DEGENERATE_LEN:
        return False
    match = _REPEAT_RE.search(text)
    if match is not None and _ALNUM_RE.search(match.group(1)):
        return True
    words = text.split()
    if len(words) >= 60:
        unique = len({w.lower() for w in words})
        if unique <= max(4, len(words) // 25):
            return True
    return False


# --- contract result ---------------------------------------------------------


@dataclass(frozen=True)
class ReplyContract:
    """Raw model output after passing through the contract.

    ``status``:
      - ``"structured"`` -- envelope found and ``reply`` non-empty (the wanted
        path).
      - ``"sanitized"``  -- no envelope, but everything after a known marker
        was salvaged (layer 2 engaged).
      - ``"raw"``        -- neither envelope nor marker; text as-is. Leakage
        MAY have passed through in this case, hence `ok` is False and the
        caller retries.
    """

    reply: str
    cited: tuple[str, ...] = ()
    status: str = "structured"
    # Members of `cited` NOT FOUND in the source list -- an invented-citation
    # signal. Today only LOGGED; it never drops the answer.
    unknown_cited: tuple[str, ...] = ()
    degenerate: bool = False
    marker: str | None = None

    @property
    def ok(self) -> bool:
        """Can it go to the user as-is (no retry needed)."""
        return self.status == "structured" and not self.degenerate and bool(self.reply.strip())

    def log_note(self) -> str:
        """One-line diagnostic summary (for the logging_setup file log)."""
        bits = [f"status={self.status}"]
        if self.marker:
            bits.append(f"marker={self.marker!r}")
        if self.degenerate:
            bits.append("degenerate=True")
        if self.cited:
            bits.append(f"cited={len(self.cited)}")
        if self.unknown_cited:
            bits.append(f"unknown_cited={list(self.unknown_cited)}")
        return ", ".join(bits)


def parse_reply(raw: str, *, evidence_ids: tuple[str, ...] | list[str] = ()) -> ReplyContract:
    """Passes raw ``message.content`` through the contract. NEVER raises --
    worst case it carries the raw text with ``status="raw"`` (an answering
    problem must not drop the turn; SAME as `reconciler`'s "no error ever
    drops a turn" principle).

    If `evidence_ids` is given (the ids of `FlowContext.results`), ``cited``
    members are validated against them; entries not on the list fall into
    `unknown_cited`. This is a MECHANICAL signal for invented citations --
    observational only today; it does not reject the answer."""
    data = extract_json_object(raw)
    if isinstance(data, dict) and isinstance(data.get("reply"), str) and data["reply"].strip():
        # Defense-in-depth: the model may have put reasoning INSIDE the
        # envelope -- apply layer 2 on the enveloped path too (untouched if no
        # marker).
        reply, marker = strip_reasoning(data["reply"])
        reply = reply.strip() or data["reply"].strip()
        raw_cited = data.get("cited")
        cited = (
            tuple(str(c).strip() for c in raw_cited if str(c).strip())
            if isinstance(raw_cited, list)
            else ()
        )
        known = {str(i) for i in evidence_ids}
        unknown = tuple(c for c in cited if c not in known) if known else ()
        return ReplyContract(
            reply=reply, cited=cited, status="structured",
            unknown_cited=unknown, degenerate=is_degenerate(reply), marker=marker,
        )

    cleaned, marker = strip_reasoning(raw or "")
    if cleaned:
        return ReplyContract(
            reply=cleaned,
            status="sanitized" if marker is not None else "raw",
            degenerate=is_degenerate(cleaned), marker=marker,
        )
    return ReplyContract(reply=(raw or "").strip(), status="raw",
                         degenerate=is_degenerate(raw or ""))


__all__ = [
    "REPLY_CONTRACT_NOTE",
    "ReplyContract",
    "extract_json_object",
    "has_reasoning_marker",
    "is_degenerate",
    "parse_reply",
    "strip_reasoning",
]
