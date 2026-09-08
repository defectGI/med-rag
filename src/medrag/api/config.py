"""Config loading and validation: `config/default.toml` -> `ChatbotConfig`.

Same pattern as the root CONFIG.md (identical loader as chunker/vectorize):
none of the tuning values are hardcoded in code, they MUST all come from this
file; pydantic fields have no defaults, unknown keys are rejected
(`extra="forbid"`), missing keys fail loudly at load time.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from medrag.core.config.loader import deep_merge as _deep_merge
from medrag.core.config.loader import read_toml as _read_toml

# This file lives under src/medrag/api/ now; config/ moved WITH the package
# (same pattern as the other components) -- it sits right next to this file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.toml"


class Routing(BaseModel):
    """Intent -> flow mapping.

    Deterministic -- the answering model never picks its own strategy at
    runtime. Any intent not listed in `intents` falls back to `default_flow`.
    Keys are `retrieval.core.IntentLabel` values (strings); values are flow
    names registered in the `flows` package.
    """

    model_config = ConfigDict(extra="forbid")

    default_flow: str
    intents: dict[str, str] = Field(default_factory=dict)


class FlowTuning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Number of requested results for every top_n / db_query call.
    top_n_k: int = Field(ge=1)
    sql_k: int = Field(ge=1)
    # Separate cap for the `sql_topn` flow's (product_fact/aggregation/
    # visual_request) OWN top_n calls (first top_n + keyword top_n) --
    # SEPARATE from `sql_k`: so that growing `sql_k` doesn't unintentionally
    # grow the top_n context size / token cost.
    sql_topn_k: int = Field(ge=1)


class SessionTuning(BaseModel):
    """L0 session task-state tuning (see session_state.py / reconciler.py)."""

    model_config = ConfigDict(extra="forbid")

    # How many turns a subject goes unmentioned before it drops out of the
    # active focus (recency-decay).
    decay_window: int = Field(ge=1)
    # Below this reconciler confidence the answering model is steered to ask
    # for confirmation.
    hedge_confidence_threshold: float = Field(ge=0.0, le=1.0)
    # Should L1 (reconcile) + L2 (intent classify) run in ONE small-model
    # call? True: the reconciler also produces the intent and the separate
    # classify call is skipped (one fewer round-trip per turn). False: old
    # behavior (two separate calls). A toggle for A/B measurement on the GPU
    # box -- whether the small model reliably does both jobs (state + 9-label
    # classification) in a single call is experimental; set false if quality
    # drops.
    fuse_intent: bool
    # Semaphore counter that serializes the GPU-heavy SQL chain
    # (linking->generation) process-wide (see gpu_gate.py). 1 = the two large
    # models never race (protection against thrashing on a single 32GB VRAM);
    # if both fit, 2 can be set to try real batch parallelism.
    sql_concurrency: int = Field(ge=1)
    # If a single SQL retrieval call exceeds this time, `SqlGateTimeout` is
    # raised and the semaphore is released immediately (the service must not
    # stop on SQL errors -- see gpu_gate.py::SerializedRetriever). Since a
    # legit worst-case single request (linking 180s + generation
    # retry×3×300s) can take ~1080s, this value must be NOTICEABLY above
    # that -- the goal is not to cut normal slowness, but to prevent a
    # error/hang class from locking out ALL other users indefinitely.
    sql_gate_timeout_seconds: float = Field(gt=0)


class ResultShapeTuning(BaseModel):
    """Threshold for `FlowContext.result_shape` -- 0 results -> "zero", 1
    result -> "one", 2..few_max results -> "few", above few_max -> "many".
    A closed-set signal that only shapes the answer FORM; it doesn't affect
    flow logic (see flows/base.py::compute_result_shape)."""

    model_config = ConfigDict(extra="forbid")

    few_max: int = Field(ge=1)


class VisualTuning(BaseModel):
    """Visual visibility filter -- same principle as the chunker/parser's
    `[visual] exclude_types` (see the chunker's config `[visual]` /
    `Visual`): `visual_type` values listed here are NEVER shown to the user
    from the `images` field in top_n metadata (see
    answering_model.py::extract_visible_images). MUST NOT BE CONFUSED with the
    chunker's filter -- that one filters entry into chunk TEXT (at parse/chunk
    time); this is a separate layer: whether a visual is shown to the USER on
    the structured channel is decided here, regardless of whether it entered
    the chunk text. Empty list = no type is excluded (default: everything is
    shown) -- consistent with the chunker's "include when in doubt"
    principle."""

    model_config = ConfigDict(extra="forbid")

    # MASTER SWITCH: when False the visual channel is FULLY closed --
    # `extract_visible_images` returns an empty list and the UI never prints
    # images. NOT TO BE CONFUSED with `exclude_types`: that one is
    # type-based and works with a "SHOW if the type is unknown/missing" rule,
    # so typeless visuals still leak even if you list every type. Use case: if
    # the crop blob store is absent in this deployment (parser's
    # `STORAGE_IMAGES_DIR` moved/deleted) the UI prints broken image icons;
    # leaving `CHATBOT_IMAGE_STORAGE_DIR` empty makes the endpoint return 503
    # but the FRONTEND still keeps trying -- the real silencing point is
    # here.
    enabled: bool = True
    exclude_types: list[str] = Field(default_factory=list)


class LoggingTuning(BaseModel):
    """Persistent file logging (see logging_setup.py). Level/directory/rotation
    are tuning -> here; the operator can use the CHATBOT_LOG_LEVEL /
    CHATBOT_LOG_DIR env vars for quick overrides (webapp.main applies
    them)."""

    model_config = ConfigDict(extra="forbid")

    # Root + chatbot.* + Flask/werkzeug/urllib level. Set to WARNING to hide
    # query text (INFO) while SQL ERRORS (ERROR) stay visible.
    level: str
    # SEPARATE level for text2sql (SQL linking/generation) -- typically DEBUG
    # since SQL is the priority, the rest INFO.
    text2sql_level: str
    # Rotating-file limits.
    max_bytes: int = Field(ge=10_000)
    backup_count: int = Field(ge=0)
    # Also write to stderr in addition to the file (live watching in the
    # terminal).
    console: bool


class ConversationLogTuning(BaseModel):
    """Detailed per-conversation logging (see conversation_log.py) -- a
    channel SEPARATE from the diagnostic logging in `logging`. `dir` is
    relative to the component root (`chatbot/`); underneath it opens
    `<channel>/<conversation_id>/` folders (on WhatsApp `<conversation_id>`
    is the user's phone number)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    dir: str
    # True: every background line flowing through `logging` during a turn
    # (chatbot.* + text2sql's SQL linking/generation detail) is ALSO routed
    # into the conversation's OWN `conversation.log`. False: only
    # message/answer + the structured `turns.jsonl` are kept.
    capture_background: bool


class AnsweringTuning(BaseModel):
    """OUTPUT CONTRACT for the answering role (see reply_contract.py). The
    root cause of a set of live-test failures was that the answering role's
    output was never validated -- the model now returns a structured envelope
    (``{"reply": ..., "cited": [...]}``) and reasoning/draft text outside the
    envelope is dropped structurally."""

    model_config = ConfigDict(extra="forbid")

    # False = contract FULLY off, raw `message.content` is returned (the old
    # behavior). EMERGENCY escape hatch -- can be temporarily disabled if the
    # small model never hits the envelope format; on by default.
    contract: bool
    # How many EXTRA attempts on a contract violation (no envelope /
    # degenerate output). 0 = no retry (whatever the first output is goes
    # through the contract and is used). Every attempt is a full LLM
    # round-trip -- deliberately kept low.
    retries: int = Field(ge=0, le=3)


class PreambleTuning(BaseModel):
    """Pre-info bubbles (see preamble.py) -- the user shouldn't stare at an
    empty screen while waiting for a long turn. Fully deterministic/templated:
    there is NO EXTRA LLM CALL (the single GPU runs serially, see gpu_gate.py
    -- a model call just for the preamble would delay the very answer we're
    trying to speed up)."""

    model_config = ConfigDict(extra="forbid")

    # false = no bubble is produced at all (the Orchestrator gets
    # `preamble=None`), behavior identical to before this feature.
    enabled: bool
    # Channels the bubble is printed to (`conversation_log.CHANNEL_WEB`/
    # `CHANNEL_WHATSAPP`). The real need is WhatsApp (no live trace panel
    # there); on the webapp it's also shown as a chat bubble.
    channels: list[str] = Field(default_factory=list)
    # If the turn exceeds this many seconds, a SECOND "still working on it"
    # bubble is printed -- chosen from a more fitting pool since the flow in
    # use is known by then. 0 (or negative) = watchdog off, only the early
    # bubble is printed.
    watchdog_seconds: float
    # If we already KNOW at t≈0 that the turn goes to one of these flows
    # (only known on pinned/handoff turns), no bubble is printed -- saying
    # "looking it up" for a 1-2 second turn is noise ("skip it on flows we
    # expect to be short").
    fast_flows: list[str] = Field(default_factory=list)


class QueueTuning(BaseModel):
    """Redis + RQ queue -- the settings used by `wa_bot.py`'s `/mesaj` route
    for idempotency + accepted-work-never-lost. The connection address
    (`REDIS_URL`) is DELIBERATELY not here -- root CONFIG.md pattern: tuning
    here, connection in `.env`."""

    model_config = ConfigDict(extra="forbid")

    # RQ queue name -- the worker (`worker/main.py`) listens for the SAME
    # name.
    wa_queue_name: str
    # If a job exceeds this many seconds, RQ force-drops it. Must be
    # NOTICEABLY above the worst-case duration of the SQL chain
    # (session.sql_gate_timeout_seconds, ~1500s) -- otherwise a normally slow
    # turn hits job_timeout and gets cut short.
    job_timeout_seconds: float = Field(gt=0)
    # How many times a job is retried if it dies on an unexpected exception
    # in the worker. `_isle` already catches orchestrator errors and sends the
    # user a reply -- this counter only covers INFRASTRUCTURE errors (Redis
    # disconnect, bundle setup blowup), so it's kept low.
    job_max_retries: int = Field(ge=0, le=10)
    # Backoff between retries (seconds); must cover as many attempts as the RQ
    # `Retry(interval=[...])` list (repeated/clipped in `wa_bot.py` against
    # `job_max_retries`).
    job_retry_interval_seconds: list[int] = Field(default_factory=list)
    # How long (seconds) the `mesaj_id`-based idempotency key stays alive in
    # Redis -- against the wa-gateway RE-SENDING the same message. The
    # wa-gateway's own dedup (`_TEKRAR` in mesaj.py) is process-local and
    # resets on restart; this TTL closes that gap.
    idempotency_ttl_seconds: int = Field(ge=1)


class EvidenceTuning(BaseModel):
    """The collapsible evidence panel under the answer (webapp.py::addEvidence).

    [DOC] rows already carry the chunk text; this table tunes the EXTRA layer
    that resolves the chunk TEXT behind [SQL] rows (see sql_evidence.py)."""

    model_config = ConfigDict(extra="forbid")

    # false = [SQL] rows show only the column/value table like before this
    # feature; specs.db/chunk corpus is NOT read at all. Escape hatch.
    sql_chunks: bool
    # Max evidence chunks listed under one SQL row (AFTER dedup). `evidence`
    # carries dozens of sources in some rows -- showing them all would make
    # the panel unreadable.
    max_chunks_per_row: int = Field(ge=1, le=50)
    # Clip limit (characters) for evidence chunk text. Larger than
    # `extract_evidence_chunks`' own `snippet_len`: there the goal is a
    # PREVIEW, here it's READING.
    snippet_len: int = Field(ge=100, le=20000)


class ChatbotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing: Routing
    flow: FlowTuning
    session: SessionTuning
    logging: LoggingTuning
    visual: VisualTuning
    conversation_log: ConversationLogTuning
    result_shape: ResultShapeTuning
    answering: AnsweringTuning
    preamble: PreambleTuning
    evidence: EvidenceTuning
    queue: QueueTuning


def load_config(override: str | Path | None = None) -> ChatbotConfig:
    """Loads the default config; if `override` is given, the keys in that TOML
    are deep-merged on top of the defaults (a partial file is enough)."""
    data = _read_toml(DEFAULT_CONFIG_PATH)
    if override is not None:
        data = _deep_merge(data, _read_toml(Path(override)))
    return ChatbotConfig.model_validate(data)


__all__ = [
    "ChatbotConfig",
    "ConversationLogTuning",
    "EvidenceTuning",
    "FlowTuning",
    "LoggingTuning",
    "PreambleTuning",
    "QueueTuning",
    "ResultShapeTuning",
    "Routing",
    "SessionTuning",
    "VisualTuning",
    "load_config",
]
