"""The real `AnsweringModel`: a remote, OpenAI-compatible chat-completions
client -- the last missing piece in the design.

Same principles as the OpenAICompatEmbedder in
`retrieval.modules.top_n.embedder` (and in chunker/vectorize): single client
(ollama/openai/openrouter/local), `anthropic` rejected at that layer, stdlib
`urllib` (no extra HTTP dependency). The code is MIRRORed, not imported --
this component depends on retrieval, but this file uses none of retrieval's
modules (its own role, `CHATBOT_LLM_*`, see .env.example).

NOT RUN on this machine (GPU rule) -- it only sends requests to a remote
endpoint; real verification happens where that endpoint is up (where the user
fills in their own `.env`).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence

from medrag.api.conversation_log import CHANNEL_WEB, CHANNEL_WHATSAPP
from medrag.api.flows.base import FlowContext
from medrag.api.memory import Message
from medrag.api.reply_contract import REPLY_CONTRACT_NOTE, ReplyContract, parse_reply
from medrag.api.retrieval.core import RetrievalResult
from medrag.core.llm.errors import ProviderError
from medrag.core.llm.http import post_json as _post_json
from medrag.core.llm.providers import ANTHROPIC_DEFAULT_URL as _ANTHROPIC_DEFAULT_URL
from medrag.core.llm.providers import COMPAT_DEFAULT_URL as _COMPAT_DEFAULT_URL
from medrag.core.llm.providers import COMPAT_PROVIDERS as _COMPAT_PROVIDERS
from medrag.core.llm.providers import DEFAULT_OLLAMA_NUM_CTX as _DEFAULT_OLLAMA_NUM_CTX

logger = logging.getLogger(__name__)


def _resolve_num_ctx(env: Mapping[str, str], prefix: str, provider: str) -> int | None:
    """Resolves `{prefix}_NUM_CTX` -- if empty AND `provider == "ollama"`, falls
    back to `_DEFAULT_OLLAMA_NUM_CTX` and writes it back to
    `os.environ[f"{prefix}_NUM_CTX"]` (for observability, same pattern as
    parser/facts -- see `pipeline/parser/llm/__init__.py::_build_client`). An
    invalid (non-int) value raises `ProviderError` (not silently swallowed)."""
    raw = (env.get(f"{prefix}_NUM_CTX") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise ProviderError(f"{prefix}_NUM_CTX must be an integer, got {raw!r}")
    if provider == "ollama":
        num_ctx = _DEFAULT_OLLAMA_NUM_CTX
        os.environ[f"{prefix}_NUM_CTX"] = str(num_ctx)
        return num_ctx
    return None


def _image_visible(image: Mapping[str, object], exclude_types: Sequence[str]) -> bool:
    """Is a single image entry (an element of the `images` metadata field, see
    pipeline/vectorize `cli.py::_node_payload`) visible according to
    `exclude_types`? SAME principle as the chunker/parser's
    `[visual] exclude_types` (see the chunker config's `[visual]` / `Visual`):
    an entry is excluded ONLY if the `visual_type` field EXISTS and is in the
    `exclude_types` list -- if `visual_type` is absent (today's vectorize
    payload shape carries only `image_id`) the DEFAULT IS VISIBLE (include
    when in doubt); empty `exclude_types` = no type is excluded."""
    visual_type = image.get("visual_type")
    if not visual_type or not exclude_types:
        return True
    return visual_type not in exclude_types


def _visible_images(metadata: Mapping[str, object], exclude_types: Sequence[str]) -> list[dict]:
    """Returns a filtered copy of `metadata["images"]` (if present) per
    `exclude_types` -- empty list if absent/empty. Raw metadata is NOT
    MUTATED (each element comes back as its own copy)."""
    images = metadata.get("images")
    if not images:
        return []
    return [
        dict(img) for img in images
        if isinstance(img, Mapping) and _image_visible(img, exclude_types)
    ]


def extract_visible_images(
    results: Sequence[RetrievalResult], exclude_types: Sequence[str] = ()
) -> list[dict]:
    """Filters the `images` field in EVERY result's metadata inside `results`
    (e.g. `FlowContext.results` or `SessionState.last_results` carried for the
    next turn) per `exclude_types` and returns ONE flat list -- it NEVER
    touches the message text sent to the answering model (see
    `_source_suffix`/`_render_context`, which only add a NUMBER); this
    function's output is a SEPARATE, structured channel BESIDE the answer
    (see webapp.py attaching it as the `images` field of the
    `/api/chat`+`/api/chat/stream` responses). Each image gets
    `source_node_id` added for traceability to its chunk (the source dict is
    NOT MUTATED; the addition is on a copy)."""
    out: list[dict] = []
    for r in results:
        for img in _visible_images(r.metadata, exclude_types):
            img.setdefault("source_node_id", r.id)
            out.append(img)
    return out


def extract_visible_documents(results: Sequence[RetrievalResult]) -> list[dict]:
    """From the rows produced by the `doc_download` flow (see
    flows/doc_download.py) inside `results` (e.g. `FlowContext.results` or
    `SessionState.last_results`), FILTERS only `requested=True` ones and
    returns them as ONE flat list -- the SAME principle as
    `extract_visible_images`: never touches the message text sent to the
    answering model; it's a SEPARATE structured channel BESIDE the answer
    (see webapp.py attaching it as the `documents` field of `/api/chat` +
    `/api/chat/stream` responses).

    Every returned dict carries only `doc_id`/`model`/`doc_type`/`file_name`
    -- `source_path` is DELIBERATELY NOT ADDED (same principle as
    `_source_suffix` refusing to read `source_path`): these fields never exist
    in the `doc_download` flow's `_to_result` anyway, so "forgetting" them
    here is impossible too, but only the four allowed fields are explicitly
    copied (the full raw `metadata` is never returned as-is). `requested=False`
    (the same product's other files, the follow-up ground, see
    session_state.py::available_document_followups) NEVER enters this channel
    -- only files the user explicitly asked for / were sent this turn are
    offered as downloadable links."""
    out: list[dict] = []
    for r in results:
        metadata = r.metadata
        doc_id = metadata.get("doc_id")
        file_name = metadata.get("file_name")
        if not doc_id or not file_name:
            continue
        if not metadata.get("requested", False):
            continue
        out.append({
            "doc_id": doc_id,
            "model": metadata.get("model"),
            "doc_type": metadata.get("doc_type"),
            "file_name": file_name,
        })
    return out


def _evidence_file_name(metadata: Mapping[str, object]) -> str | None:
    """The FILE NAME the evidence came from (not the full path).

    On [SQL] rows: `source_file_name` (schema.yaml: "SAFE to show as a
    citation"). On [DOC] chunks: only the BASENAME of `source_path` --
    `source_path` itself never goes out to any user/model channel (see the
    `_source_suffix` docstring); the only reason it's read here is to derive
    the file NAME, and the full path is NEVER returned. The file name is
    already the field `extract_visible_documents` shows the user, so the
    forbidden part is location, not name.

    `None` if neither exists -- the caller hides the field."""
    sql_name = metadata.get("source_file_name")
    if sql_name:
        return str(sql_name)
    source_path = metadata.get("source_path")
    if source_path:
        # Without splitting on `PurePosixPath`/`PureWindowsPath`: records may
        # carry both separators (parser ran on Windows, server runs on Linux).
        return str(source_path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or None
    return None


def extract_evidence_chunks(
    results: Sequence[RetrievalResult], *, snippet_len: int = 280
) -> list[dict]:
    """Condenses EVERY result in `results` (e.g. `FlowContext.results` or
    `SessionState.last_results`) into a small dict that can be shown to the
    user as the evidence the answer is BASED on -- the SAME principle as
    `extract_visible_images`/`extract_visible_documents`: it never touches the
    message text sent to the answering model; it's a SEPARATE structured
    channel BESIDE the answer (see webapp.py attaching it as the `chunks`
    field of `/api/chat` + `/api/chat/stream` responses). Reuses the
    [SQL]/[DOC] tag `_render_context` sends to the model (`_source_kind`) and
    the source info (same fields as `_source_suffix`: doc_id/heading_path/
    page), but `source_path` is DELIBERATELY NOT ADDED (same principle as the
    `_source_suffix` docstring). Text is clipped to `snippet_len` -- the same
    clipping principle as `results_to_trace` in the trace panel (see
    trace.py): not the FULL raw chunk."""
    out: list[dict] = []
    for r in results:
        metadata = r.metadata
        text = r.text
        if len(text) > snippet_len:
            text = text[:snippet_len] + "…"
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        page = None
        if page_start is not None:
            page = str(page_start) if page_end in (None, page_start) else f"{page_start}-{page_end}"
        heading_path = metadata.get("heading_path")
        out.append({
            "id": r.id,
            "score": round(r.score, 4),
            "source_kind": _source_kind(metadata),
            "doc_id": metadata.get("doc_id"),
            "section": " > ".join(heading_path) if heading_path else None,
            "page": page,
            # WHICH FILE it came from (user requirement: the panel should also
            # say which file the chunk came from). The two sources have two
            # separate fields, both are FILE NAMES -- never full paths:
            #   [SQL] `source_file_name` -- facts/db/schema.yaml explicitly
            #         marks this "SAFE to show as a citation" (its sibling
            #         `source_doc_id` says "never show it"). Present on a row
            #         only if the generated SQL SELECTs it -- hence
            #         citation columns are added by code (`sql_citation.py`);
            #         still absent -> stays `None`, no crash.
            #   [DOC] only the BASENAME of `source_path` in the payload (see
            #         pipeline/vectorize `cli.py::_node_payload`). `source_path`
            #         ITSELF goes to no channel -- but the file NAME is already
            #         shown to the user by `extract_visible_documents` too, so
            #         the forbidden part is the full path, not the name.
            "file_name": _evidence_file_name(metadata),
            # WHICH ATTRIBUTE this is evidence for: `spec_value.key` is one
            # single feature of one product (e.g. "operating_pressure") -- a
            # SQL row is exactly the evidence of one attribute. [DOC] chunks
            # have NO such field (free text, not tied to one feature) ->
            # `None`; there the closest equivalent is already `section`
            # (heading_path).
            "attribute": metadata.get("key"),
            "product_code": metadata.get("product_code"),
            # `sql_tables` is a metadata field `SqlTopNFlow._run_sql` adds
            # deterministically from its linking stage (already computed, not a
            # new LLM call) -- found only on SQL rows, never on DOC chunks
            # (see sql_topn.py).
            "table": metadata.get("sql_tables"),
            # `chunk_schema_version`/`chunker_version` (user requirement: the
            # panel should also show which chunker version produced the
            # evidence chunks) -- metadata fields coming from
            # pipeline/vectorize `cli.py::_node_payload`, filled only on [DOC]
            # (vectorize-sourced) rows; on [SQL] (facts/specs.db-sourced) rows
            # both are None (a SQL row has no separate provenance of its own).
            "chunk_schema_version": metadata.get("chunk_schema_version"),
            "chunker_version": metadata.get("chunker_version"),
            "snippet": text,
        })
    return out


def _source_suffix(metadata: Mapping[str, object], exclude_types: Sequence[str] = ()) -> str:
    """Produces a readable suffix from `doc_id`/`heading_path`/`page_start`/
    `page_end` (part of the top_n payload, see pipeline/vectorize
    `cli.py::_node_payload`) if present -- SQL rows lack these fields, so an
    empty string is returned for them (so the model can see the chunk's
    source).

    Deliberately does NOT read `source_path` -- a filesystem path is never
    leaked to the user/model even if present in metadata; `doc_id` is already
    the human-readable identifier and suffices to cite the source without
    revealing the real file location. Do NOT ADD `metadata.get("source_path")`
    to "enrich" this line later -- it's a deliberate choice, not a gap. If
    `source_path` ever becomes necessary, that must be an explicit new
    decision, not a silent widening.

    `images` (see `extract_visible_images`): if present in metadata, only the
    VISIBLE (filtered by exclude_types) COUNT is appended -- raw data like
    `image_id`/file paths is DELIBERATELY not leaked (same principle as the
    `source_path` rule above): the model should learn that an image EXISTS
    (so it can say "I can show a diagram") but never see the raw metadata as
    unstructured prose -- the real structured image list travels on a SEPARATE
    channel via `extract_visible_images` (see webapp.py)."""
    bits = []
    doc_id = metadata.get("doc_id")
    if doc_id:
        bits.append(f"doc={doc_id}")
    heading_path = metadata.get("heading_path")
    if heading_path:
        bits.append(f"section={' > '.join(heading_path)}")
    page_start = metadata.get("page_start")
    if page_start is not None:
        page_end = metadata.get("page_end")
        page = f"page={page_start}"
        if page_end is not None and page_end != page_start:
            page += f"-{page_end}"
        bits.append(page)
    visible_images = _visible_images(metadata, exclude_types)
    if visible_images:
        bits.append(f"images={len(visible_images)}")
    return f", {', '.join(bits)}" if bits else ""


def _source_kind(metadata: Mapping[str, object]) -> str:
    """Returns "DOC" (top_n/RAG) if `doc_id`/`heading_path`/`page_start` (the
    top_n payload's signature, see `_source_suffix`) are present, else "SQL"
    (structured data) -- the answer prompt now tells the model EXPLICITLY
    whether the evidence is SQL or top_n via a visible tag, instead of only
    indirectly through the presence of metadata fields (SQL is the
    definitive/authoritative source, see strategies/product_fact.md +
    aggregation.md)."""
    if metadata.get("doc_id") or metadata.get("heading_path") or metadata.get("page_start") is not None:
        return "DOC"
    return "SQL"


def _result_signal_note(context: FlowContext) -> str:
    """`FlowContext.result_shape`/`residual`/`parse_note` -- closed-set,
    STRUCTURED signals derived from the flow's RESULT, sent to the model as a
    SHORT note BESIDE `results` + the strategy prompt. `context.scratch` (the
    flow's internal/raw workspace) is NOT READ here and never will be -- these
    three behave like a closed dict explicitly defined on the flow's own
    FlowContext (a new Flow cannot LEAK arbitrary/free-text intermediate
    output through this path; it can only fill these three fields).

    Returns an empty string when all three are empty (the default -- including
    ALL old behavior for flows that never fill these fields today); no system
    piece is appended (same pattern as `followups` when empty)."""
    bits: list[str] = []
    if context.result_shape:
        bits.append(f"result_shape={context.result_shape}")
    if context.residual:
        bits.append(f'unconsumed part of the query: "{context.residual}"')
    if context.parse_note:
        bits.append(f"parse_note={context.parse_note}")
    if not bits:
        return ""
    return (
        "Internal result signal (for your own reasoning about how to phrase "
        "the reply -- e.g. whether to ask a follow-up, or note that "
        "something didn't look like a valid product code -- never quote "
        "this note verbatim to the user): " + "; ".join(bits)
    )


def _render_context(context: FlowContext, exclude_types: Sequence[str] = ()) -> str:
    if not context.results:
        return "(no relevant source found)"
    parts = []
    for r in context.results:
        suffix = _source_suffix(r.metadata, exclude_types)
        kind = _source_kind(r.metadata)
        parts.append(f"[{kind}][{r.id}] (score={r.score:.3f}{suffix})\n{r.text}")
    return "\n\n".join(parts)


_HEDGE_NOTE = (
    "The user's ongoing goal is unclear -- if needed, briefly ask for "
    "confirmation in your reply (e.g. whether they're still asking about the "
    "same comparison/task)."
)

# A single language sentence buried in the middle of `_BASE_PERSONA` is not
# enough for small/local models -- the rest of the context window (persona +
# strategy + source text) is almost entirely English, so the model can drift to
# the DOMINANT language of the context (buried-instruction weakness, a flavor of
# the "lost in the middle" phenomenon). The fix is not prompt wording but
# PLACEMENT: this instruction is appended as a SEPARATE, LAST message -- AFTER
# the user's turn (see `_build_messages`), the place closest to generation (and
# thus with the highest attention weight) for the model (recency effect; a
# sentence buried INSIDE the system message, before the user turn, does NOT
# achieve this -- it still lands between the conversation history and the real
# question in the token flow). The sentence in `_BASE_PERSONA` was deliberately
# NOT removed (defense-in-depth; the two don't conflict) but the real guarantee
# is now here -- it references the user's own MESSAGE TEXT, not a fixed language
# name (e.g. "Turkish"): no new language-detection dependency/LLM call is
# needed, and it automatically adapts to languages other than English/Turkish.
# The intermediate layers (persona/strategy/source text) deliberately stay in
# English -- the small model's reasoning quality comes from there; this block
# only changes the language of text that SURFACES, not the reasoning.
_LANGUAGE_REMINDER = (
    "Reply in the SAME language as the user's message above -- not the "
    "language of the sources/instructions earlier in this prompt. If the "
    "user wrote in Turkish, reply in Turkish; if English, reply in English; "
    "match whatever language they actually used, even if it changed from "
    "earlier in the conversation."
)

# ALWAYS appended, whether the strategy is filled or empty -- an incident where
# the system message was reduced to just the "Relevant sources: (none found)"
# line (empty strategy + empty sources, e.g. an out_of_scope turn like "selam")
# left the model with NO identity/task framing and a small model produced
# irrelevant meta-output in that void. This base guarantees a floor no matter
# which strategy file is empty; a filled strategy ADDS extra instructions ON TOP
# rather than replacing it.)
_BASE_PERSONA = (
    "You are a personal medical reference assistant for a physician. You "
    "answer ONLY from the user's own uploaded documents (textbooks, "
    "guidelines, lecture notes, personal notes) that are given to you as "
    "sources. This is an EVIDENCE-FIRST role: every factual claim in your "
    "reply -- a disease fact, a mechanism, a dose, a dosage interval, a "
    "contraindication -- MUST carry an inline citation to the document it "
    "came from, in the form (Kaynak: <file name>, s. <page>) written in the "
    "user's language (e.g. '(Source: Nelson.pdf, p. 412)'). A claim WITHOUT "
    "a citation must not be written. If the sources do not contain the "
    "answer, say plainly that you could not find it in the uploaded "
    "documents -- NEVER fill gaps from your own knowledge, NEVER invent "
    "doses, drug names, or facts. If two sources give DIFFERENT values for "
    "the same fact (e.g. two different doses), present BOTH values with "
    "their citations and say the sources differ -- never silently pick one. "
    "If the message is a basic conversational courtesy/greeting ('hi', "
    "'hello', 'thanks'), greet back naturally and briefly; but if a "
    "genuinely unrelated topic comes up (small talk, code, sports, "
    "shopping), briefly state that you only answer from the uploaded "
    "medical documents. When the answer involves drug dosing, treatment "
    "decisions, or diagnosis, end the reply with this one short line in "
    "the user's language: 'Klinik kararlarda güncel kaynakları ve hastanın "
    "bireysel durumunu değerlendiriniz.' (or 'Please verify against current "
    "guidelines and the individual patient context.' in English). Always "
    "reply in the same language the user wrote in. Speak briefly, clearly "
    "and in a natural tone -- when something is unclear or no evidence was "
    "found, say so plainly. Evidence tags such as [SQL]/[DOC], score=, and "
    "doc= appearing in the sources below are for your OWN reasoning only -- "
    "never write them back into the reply text shown to the user."
)

# The two channels go to two different render engines -- the webapp has its own
# hand-written markdown subset (`webapp.py::renderMarkdown`, incl. GFM tables),
# WhatsApp has NONE (plain text + `*bold*`/`_italic_`/```code``` only -- it
# CANNOT render real tables/pipes/dash separators). Decision: the distinction is
# resolved at the SOURCE (this prompt) -- having WhatsApp produce web format
# (GFM tables) first and post-processing into plain text was a structural text
# transform adding a parsing-error-prone layer; if the model produces the right
# format at the source, that layer is never needed ("post-processing increases
# error risk").
_FORMATTING_WEB = (
    "FORMATTING (the frontend renders a small, fixed markdown subset -- "
    "stick to exactly this, nothing fancier):\n"
    "- Comparing 2+ products, or listing 2+ attributes for one product: "
    "use a GitHub-flavored markdown table (header row, then a dash-only "
    "separator row, then one data row per attribute/product). Do NOT lay "
    "out a comparison as a bullet list of 'attribute: * X * Y' -- that's "
    "what a table is for.\n"
    "- A short list of items with no natural row/column shape: use `- ` "
    "(dash + space) bullets, one per line. Never use `*` for bullets "
    "(it's reserved for **bold** emphasis here).\n"
    "- Keep tables narrow: short cell text, no nested tables/lists inside "
    "a cell."
)

_FORMATTING_WHATSAPP = (
    "FORMATTING (this reply is sent as a plain WhatsApp message -- WhatsApp "
    "has NO real table renderer, so a markdown/GFM table would show up as "
    "broken raw text with literal '|' and '---' characters; NEVER produce "
    "one, not even as a fallback):\n"
    "- Comparing 2+ products, or listing 2+ attributes for one product: "
    "write one short paragraph per item -- a bold line with the item's name "
    "(e.g. product name, or a single attribute if there's only one row), "
    "then one indented bullet per attribute/value directly under it "
    "('  - attribute: value'). Leave a blank line between items. If it's "
    "just one attribute/value pair, a single line 'attribute: value' is "
    "enough -- no need for a bullet.\n"
    "- A short list of items with no natural row/column shape: use `- ` "
    "(dash + space) bullets, one per line.\n"
    "- Bold uses a SINGLE asterisk, e.g. *label* -- this is WhatsApp's own "
    "bold syntax, not markdown's **label**. Never use double asterisks.\n"
    "- No headings (#), no nested lists, no code fences -- keep it plain, "
    "short lines a phone screen can show without horizontal scrolling."
)


def _formatting_for_channel(channel: str) -> str:
    """When `channel` is an unknown/empty value (e.g. an old call site, a test),
    err on the SAFE side: web format -- webapp is already the existing
    behavior, and WhatsApp accidentally receiving web instructions (table
    production) is a more predictable failure mode than a raw KeyError."""
    if channel == CHANNEL_WHATSAPP:
        return _FORMATTING_WHATSAPP
    return _FORMATTING_WEB


def _build_messages(
    query: str,
    context: FlowContext,
    strategy_prompt: str,
    history: Sequence[Message],
    *,
    session_goal: str = "",
    hedge: bool = False,
    followups: str = "",
    image_exclude_types: Sequence[str] = (),
    contract: bool = True,
    correction: str = "",
    channel: str = CHANNEL_WEB,
) -> list[dict[str, str]]:
    """If `session_goal` is given (the ongoing-goal framing from L0, see
    session_state.py::SessionState.as_goal_prompt), it is appended to the
    system prompt -- the model does its single-turn job without losing
    context. If `hedge` (low reconciler confidence) is True, the model is asked
    to seek confirmation. `followups` (see SessionState.followups_prompt) is a
    suggestion note derived from the previous turn's REAL spec_keys with no
    off-schema invention risk -- nothing is appended when empty.

    `contract` appends the output-contract note (see reply_contract.py);
    `False` is used only when the contract is fully disabled
    (`OpenAICompatAnsweringModel(contract=False)`, the emergency escape hatch).
    When `correction` is non-empty (retry after a contract violation) the model
    is told WHAT went wrong -- the broken output ITSELF is never sent back, as
    that would re-prime the model on the same pattern.

    `channel` selects which FORMATTING block is appended
    (`_formatting_for_channel`) -- `CHANNEL_WEB`/`CHANNEL_WHATSAPP` (see
    conversation_log.py), coming from `Orchestrator._channel`."""
    system_parts = [_BASE_PERSONA, _formatting_for_channel(channel)]
    if strategy_prompt:
        system_parts.append(strategy_prompt)
    if session_goal:
        system_parts.append(session_goal)
    if hedge:
        system_parts.append(_HEDGE_NOTE)
    if followups:
        system_parts.append(followups)
    signal_note = _result_signal_note(context)
    if signal_note:
        system_parts.append(signal_note)
    system_parts.append(f"Relevant sources:\n\n{_render_context(context, image_exclude_types)}")

    messages: list[dict[str, str]] = [{"role": "system", "content": "\n\n---\n\n".join(system_parts)}]
    for m in history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": query})
    # Output contract (see reply_contract.py) -- appended as a separate
    # message AFTER the user turn, with the same placement principle proven for
    # `_LANGUAGE_REMINDER` (a format instruction buried inside the system
    # message gets skipped by small models). The guarantee that
    # `_LANGUAGE_REMINDER` is the LAST message is deliberately preserved -- the
    # contract note comes BEFORE it.
    if contract:
        messages.append({"role": "system", "content": REPLY_CONTRACT_NOTE})
    if correction:
        messages.append({"role": "system", "content": correction})
    # Language instruction deliberately HERE, after the user turn -- see the
    # `_LANGUAGE_REMINDER` note (recency/placement fix).
    messages.append({"role": "system", "content": _LANGUAGE_REMINDER})
    return messages


def _rejected_extra_body(exc: ProviderError, extra_body: Mapping[str, object]) -> bool:
    """True if the server rejected a field in extra_body (the 400 mentions the
    field name) -- a model without thinking modes doesn't know the suppression
    field, so instead of hard-failing the call is retried without a body (same
    pattern as parser's `llm/openai_compat.py::_rejected_extra_body`)."""
    detail = str(exc).lower()
    return any(str(key).lower() in detail for key in extra_body)


# The text the user gets when EVERY attempt returns degenerate/empty output.
# PRINTING the raw degenerate text (hundreds of repeated nonsense sentences) to
# the user and writing it to memory is exactly what triggered the domino
# failure -- so the last resort is not silent feeding but a short, honest note
# (`wa_session._TTL_RESET_NOTE` is the same pattern). Complies with the
# `_BASE_PERSONA` rule "never use technical language like 'an error occurred'".
_CONTRACT_FALLBACK_REPLY = (
    "Bu soruya düzgün bir cevap oluşturamadım. Sorunuzu biraz farklı "
    "ifade ederek tekrar sorar mısınız?"
)


def _contract_correction(parsed: ReplyContract) -> str:
    """The correction note appended to the retry after a contract violation.

    The broken output ITSELF is never sent back to the model (see the
    `_build_messages` docstring) -- putting a degenerate/reasoning-laden text
    into the prompt would re-prime the model on the same pattern. Only WHAT
    went wrong is said."""
    if parsed.degenerate:
        return (
            "Your previous attempt collapsed into a repeating loop of the same "
            "text. Start over: write ONE short, complete answer, then stop. "
            "Return only the JSON object described above."
        )
    return (
        "Your previous attempt was NOT a single valid JSON object (reasoning, "
        "channel markers or prose appeared outside it). Return ONLY the JSON "
        'object: {"reply": "...", "cited": [...]} -- no text before or after.'
    )


def _best_effort(attempts: Sequence[tuple[str, ReplyContract]]) -> str:
    """What reaches the user when no attempt hits the contract.

    An answering problem NEVER drops the turn (`reconciler`'s same principle)
    -- but leaked/degenerate text isn't passed through as-is either. Order:
    the first non-degenerate filled text (layer 2 may have salvaged it) -> the
    fallback note."""
    for _raw, parsed in attempts:
        if parsed.reply.strip() and not parsed.degenerate:
            logger.warning(
                "cevap sözleşmesi tutmadı, kurtarılan metin kullanılıyor: %s",
                parsed.log_note(),
            )
            return parsed.reply
    logger.error(
        "cevap sözleşmesi tüm denemelerde tutmadı, kanı cevaba düşülüyor "
        "(denemeler: %s)",
        [p.log_note() for _r, p in attempts],
    )
    return _CONTRACT_FALLBACK_REPLY


class OpenAICompatAnsweringModel:
    """Implements the `orchestrator.AnsweringModel` protocol.

    ``extra_body`` is merged into every request -- e.g. ``{"reasoning_effort":
    "none"}`` to turn thinking off (see thinking.py). If the server rejects the
    field (the model has no thinking modes), the call is retried once without a
    body, so it doesn't explode even on a "no thinking" model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        provider: str = "openai",
        num_ctx: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        image_exclude_types: Sequence[str] = (),
        contract: bool = True,
        contract_retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.provider = provider
        # When `provider == "ollama" and num_ctx` are set, the request goes to
        # native `/api/chat` (see the `_call_native` in `answer`) -- because
        # Ollama's `/v1` endpoint can't take num_ctx per-request (module
        # docstring).
        self.num_ctx = num_ctx
        self.extra_body = dict(extra_body or {})
        # Which `visual_type`s are excluded from the image-visibility summary
        # (see `_source_suffix`) -- comes from `config/default.toml [visual]
        # exclude_types` (see `answering_model_from_env`).
        self.image_exclude_types = tuple(image_exclude_types)
        # Output contract (see reply_contract.py). `contract=False` turns the
        # contract FULLY off (raw content returned) -- an emergency escape
        # hatch, not the default; managed via `config/default.toml
        # [answering] contract`.
        self.contract = contract
        self.contract_retries = max(0, int(contract_retries))

    async def answer(
        self,
        query: str,
        context: FlowContext,
        strategy_prompt: str,
        history: Sequence[Message] = (),
        *,
        session_goal: str = "",
        hedge: bool = False,
        followups: str = "",
        channel: str = CHANNEL_WEB,
    ) -> str:
        evidence_ids = tuple(r.id for r in context.results)

        def _call(correction: str = "") -> str:
            messages = _build_messages(
                query, context, strategy_prompt, history,
                session_goal=session_goal, hedge=hedge, followups=followups,
                image_exclude_types=self.image_exclude_types,
                contract=self.contract, correction=correction, channel=channel,
            )
            if self.provider == "anthropic":
                return self._call_anthropic(messages)
            if self.provider == "ollama" and self.num_ctx:
                return self._call_native(messages)
            return self._call_openai_compat(messages)

        def _run() -> str:
            # When the contract is off (emergency escape hatch) behavior is
            # IDENTICAL to before this feature: raw content is returned.
            if not self.contract:
                return _call()
            attempts: list[tuple[str, ReplyContract]] = []
            correction = ""
            for attempt in range(self.contract_retries + 1):
                raw = _call(correction)
                parsed = parse_reply(raw, evidence_ids=evidence_ids)
                attempts.append((raw, parsed))
                if parsed.ok:
                    # An invented citation does NOT drop the answer but
                    # doesn't stay silent either -- WARNING, because it's a
                    # quality signal that must be watched.
                    if parsed.unknown_cited:
                        logger.warning("cevap sözleşmesi: %s", parsed.log_note())
                    elif attempt:
                        logger.warning(
                            "cevap sözleşmesi %d. denemede tuttu: %s",
                            attempt + 1, parsed.log_note(),
                        )
                    else:
                        logger.debug("cevap sözleşmesi: %s", parsed.log_note())
                    return parsed.reply
                # Violation: log the raw text IN FULL (same principle as
                # sql_error -> ERROR + full raw_text -- the real diagnosis is
                # here, never clipped at any level).
                logger.warning(
                    "cevap sözleşmesi ihlali (deneme %d/%d): %s -- ham çıktı: %s",
                    attempt + 1, self.contract_retries + 1, parsed.log_note(), raw,
                )
                correction = _contract_correction(parsed)
            return _best_effort(attempts)

        return await asyncio.to_thread(_run)

    def _call_anthropic(self, messages: list[dict[str, str]]) -> str:
        """Called when `self.provider == "anthropic"` -- see the docstring of
        `core/llm/anthropic.py::call_anthropic` (three differences from the
        OpenAI-compatible `/v1/chat/completions` contract are translated
        there)."""
        from medrag.core.llm.anthropic import call_anthropic

        return call_anthropic(
            messages, model=self.model, base_url=self.base_url,
            api_key=self.api_key, timeout=self.timeout,
        )

    def _call_openai_compat(self, messages: list[dict[str, str]]) -> str:
        base_payload = {"model": self.model, "messages": messages}
        url = f"{self.base_url}/chat/completions"
        try:
            response = _post_json(
                url, {**base_payload, **self.extra_body},
                api_key=self.api_key, timeout=self.timeout,
            )
        except ProviderError as exc:
            if self.extra_body and _rejected_extra_body(exc, self.extra_body):
                response = _post_json(
                    url, base_payload, api_key=self.api_key, timeout=self.timeout
                )
            else:
                raise
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"unexpected chat completion response shape: {str(response)[:500]}"
            ) from exc

    def _call_native(self, messages: list[dict[str, str]]) -> str:
        """Only called when `provider == "ollama" and num_ctx` are set --
        Ollama's native `/api/chat`, which (unlike `/v1`) accepts
        `options.num_ctx` per request (module docstring). The message list
        produced by `_build_messages` is used UNCHANGED; only the POST
        target/body/response-parsing differ."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self.num_ctx},
        }
        # The `reasoning_effort` key in `extra_body` signals that thinking was
        # EXPLICITLY disabled (see thinking.py::thinking_off_body) -- on the
        # native path this is translated directly to `think:false` instead of
        # the `/v1`-specific field. Same mapping as the text2sql note elsewhere
        # in this codebase. With no such signal the `think` field is NOT added
        # at all (neutral; the server's own default applies).
        if "reasoning_effort" in self.extra_body:
            payload["think"] = False
        url = f"{self.base_url.removesuffix('/v1')}/api/chat"
        response = _post_json(url, payload, api_key=self.api_key, timeout=self.timeout)
        try:
            return str(response["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                f"unexpected chat completion response shape: {str(response)[:500]}"
            ) from exc


def _common(env: Mapping[str, str], prefix: str) -> tuple[str, str, str, str | None]:
    provider = (env.get(f"{prefix}_PROVIDER") or "").strip().lower()
    if not provider:
        raise ProviderError(f"{prefix}_PROVIDER tanımsız (bkz. .env.example)")
    # Anthropic is now natively supported (see `core/llm/anthropic.py`) -- it
    # did NOT join the `_COMPAT_PROVIDERS` (OpenAI-compatible
    # `/v1/chat/completions`) list; it is handled separately (own default root
    # `ANTHROPIC_DEFAULT_URL`, routed to `_call_anthropic` instead of
    # `_call_openai_compat`).
    if provider != "anthropic" and provider not in _COMPAT_PROVIDERS:
        raise ProviderError(f"bilinmeyen {prefix}_PROVIDER: {provider!r}")
    model = (env.get(f"{prefix}_MODEL") or "").strip()
    if not model:
        raise ProviderError(f"{prefix}_MODEL tanımsız")
    default_url = _ANTHROPIC_DEFAULT_URL if provider == "anthropic" else _COMPAT_DEFAULT_URL.get(
        provider, ""
    )
    base_url = (env.get(f"{prefix}_BASE_URL") or "").strip() or default_url
    if not base_url:
        raise ProviderError(f"{prefix}_BASE_URL gerekli ({provider} için varsayılan kök yok)")
    return provider, model, base_url, (env.get(f"{prefix}_API_KEY") or None)


def answering_model_from_env(
    env: Mapping[str, str] = os.environ,
    *,
    timeout: float | None = None,
    image_exclude_types: Sequence[str] = (),
    contract: bool = True,
    contract_retries: int = 1,
) -> OpenAICompatAnsweringModel:
    """Builds from `CHATBOT_LLM_*` (this component's own role, NOT retrieval's
    `LLM_*` -- see .env.example). If `CHATBOT_LLM_THINKING` is off, the thinking
    suppression body (see thinking.py) is injected.

    `image_exclude_types` comes NOT from ENV but from `config/default.toml
    [visual] exclude_types` (tuning, root CONFIG.md taxonomy); the caller
    (`webapp.py::create_app`) passes `load_config().visual.exclude_types` here.

    `contract`/`contract_retries` follow the SAME principle -- tuning from
    `config/default.toml [answering]`, not env; they're configuration, not
    connection/order."""
    from medrag.api.thinking import thinking_off_body

    provider, model, base_url, api_key = _common(env, "CHATBOT_LLM")
    resolved_timeout = timeout if timeout is not None else float(env.get("CHATBOT_LLM_TIMEOUT") or 60)
    num_ctx = _resolve_num_ctx(env, "CHATBOT_LLM", provider)
    return OpenAICompatAnsweringModel(
        base_url=base_url, model=model, api_key=api_key, provider=provider,
        num_ctx=num_ctx, timeout=resolved_timeout,
        extra_body=thinking_off_body(env, "CHATBOT_LLM"),
        image_exclude_types=image_exclude_types,
        contract=contract, contract_retries=contract_retries,
    )


__all__ = [
    "OpenAICompatAnsweringModel",
    "ProviderError",
    "answering_model_from_env",
    "extract_evidence_chunks",
    "extract_visible_documents",
    "extract_visible_images",
]
