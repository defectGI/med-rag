"""Production wiring: builds the real retrievers + the `Router` from env.

`retrieval.modules.top_n` is now usable (real vectors written to Qdrant) --
this module does NOT read top_n's own `.env`; it reads the `QDRANT_*` owned by
this component and passes them as parameters to retrieval's factory (same
principle as db_query's `CHATBOT_DB_QUERY_DB_PATH` -> `db_path`: this
component decides WHICH Qdrant/DB to connect to, not the retrieval package --
the isolation pattern is PRESERVED; only the naming scheme was unified across
components).

The previous `CHATBOT_QDRANT_*` names were reduced to `QDRANT_*` -- `vectorize`
(the writer) and `retrieval/modules/top_n` (its own process) already used the
unprefixed `QDRANT_URL`/`QDRANT_API_KEY`, so instead of a third naming scheme
the SAME names are now used in this component's OWN `.env` too. This does NOT
break isolation: every component still reads its own process environment; only
the "three components looking at the same Qdrant under three different names"
duplication is removed. No backward-compat aliases.

The embedding model (`EMBEDDING_*`) remains retrieval's SHARED [inference]
env -- common to top_n and raptor, not repeated here; it just has to be defined
in the process environment (this component's own `.env` or retrieval's
`.env`, whichever was loaded).

**text2sql-engine wiring** (a proven working configuration): `config.toml` + `prompts/` are COMMITTED under
`text2sql/` in this component (runtime wiring). `schema.yaml` and `specs.db`
live in the root `facts/` component: `facts/db/schema.yaml` +
`facts/db/specs.db`, both COMMITTED (`.gitignore` exception
`!facts/db/specs.db`) -- the repo works standalone; if
`CHATBOT_DB_QUERY_DB_PATH` is unset, this committed file is used, and if set,
the user's override wins. `facts` CODE is NOT imported, only the file path is
read. text2sql's OWN mandatory env fields (`SCHEMA_PATH`, `LINKING_*`,
`SQL_*`, `OPENAI_API_KEY` -- all unprefixed, fixed names owned by
text2sql-engine, not ours to change) are documented in `.env.example`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from medrag.api.config import ChatbotConfig, load_config
from medrag.api.continuation import build_continuation_checker_from_env
from medrag.api.conversation_log import CHANNEL_WEB
from medrag.api.flows import (
    AggregationFlow,
    ComparisonFlow,
    DefaultTopNFlow,
    DeterministicComparisonSplitter,
    DocDownloadFlow,
    Flow,
    NoRetrievalFlow,
    RecommendationFlow,
    SqliteDocumentLookup,
    SqlTopNFlow,
)
from medrag.api.gpu_gate import SerializedRetriever
from medrag.api.memory import ConversationMemory
from medrag.api.orchestrator import Orchestrator

# The `PreambleSettings` return annotation of
# `preamble_settings_from_config` needs this name; importing only under
# TYPE_CHECKING broke `typing.get_type_hints()` calls with a NameError.
# preamble.py imports no `medrag` modules, so a module-level import isn't
# circular.
from medrag.api.preamble import PreambleSettings
from medrag.api.retrieval.modules.db_query import DbQueryError
from medrag.api.retrieval.modules.db_query import (
    build_default_retriever as build_db_query_retriever,
)
from medrag.api.retrieval.modules.top_n import TopNRetriever
from medrag.api.retrieval.modules.top_n import (
    build_default_retriever as build_top_n_retriever,
)
from medrag.api.router import Router
from medrag.api.session_state import SessionStateStore
from medrag.api.slot_selection import build_slot_selector_from_env
from medrag.api.sql_citation import add_citation_columns
from medrag.core.paths import FACTS_DB_DIR as _CORE_FACTS_DB_DIR
from medrag.core.paths import SPECS_DB_PATH as _CORE_SPECS_DB_PATH

# This file lives under src/medrag/api/ now; `text2sql/` stayed as DATA in the
# old `chatbot/` directory at the repo root during the package restructure.
# parents[3] = repo root.
_TEXT2SQL_DIR = Path(__file__).resolve().parents[3] / "chatbot" / "text2sql"
_BUNDLED_CONFIG_PATH = _TEXT2SQL_DIR / "config.toml"
_BUNDLED_PROMPT_DIR = _TEXT2SQL_DIR / "prompts"
# `schema.yaml` + `specs.db` are no longer in THIS component but in the root
# `facts/` component: their producer is `facts`, not `chatbot`, so the
# artifact lives under its producer. Only the FILE PATH is read from here --
# `facts` CODE is never imported (same pattern as the image_store/
# document_store exceptions).
# The repo-root path computation is not duplicated here either; it hooks into
# FACTS_DB_DIR, which core.paths derives from the SAME root as SPECS_DB_PATH.
_FACTS_DB_DIR = _CORE_FACTS_DB_DIR
_BUNDLED_SCHEMA_PATH = _FACTS_DB_DIR / "schema.yaml"
# The schema that GOES TO THE MODEL -- a trimmed projection of `schema.yaml`:
# no provenance/internal bookkeeping columns the model has no reason to query
# (`evidence`/`confidence`/`extractor`/`source_chunk_id`/`source_doc_id`/
# `source_file_name`) and no `attribute_glossary` block. TWO FILES DELIBERATE:
# `schema.yaml` stays FULL as source-of-truth and this component keeps reading
# it for `attribute_glossary` (see the
# `slot_candidates.load_attribute_glossary` call below); text2sql's
# `SCHEMA_PATH` sees only the trimmed one. Citation columns are now added by
# CODE, without asking the model (`sql_citation.py`). Both files are produced
# by the facts pipeline build.
_BUNDLED_MODEL_SCHEMA_PATH = _FACTS_DB_DIR / "schema_model.yaml"
# specs.db is committed inside the repo (`.gitignore` exception
# `!facts/db/specs.db`). If CHATBOT_DB_QUERY_DB_PATH is unset, this is the
# fallback -- so the repo works standalone. The user's env override always
# wins. (Path resolution lives in core.paths, same computation as
# _FACTS_DB_DIR; the variable stays because schema.yaml/schema_model.yaml use
# it too.)
_BUNDLED_DB_PATH = _CORE_SPECS_DB_PATH


def build_top_n_retriever_from_env() -> TopNRetriever:
    """Builds a real top_n retriever from `QDRANT_URL`/`QDRANT_API_KEY`/
    `QDRANT_COLLECTION` (see `.env.example`) + retrieval's shared
    `EMBEDDING_*`. It must use the SAME `EMBEDDING_*` settings as the embedding
    model that WROTE the vectors (warning in
    retrieval/modules/top_n/README.md) -- otherwise search returns meaningless
    results or Qdrant dies on a dimension mismatch.
    """
    return build_top_n_retriever(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION"),
    )


def resolve_db_path_from_env() -> str:
    """Resolves `CHATBOT_DB_QUERY_DB_PATH` (see .env.example) -- falls back to
    the committed `facts/db/specs.db` if unset. Neither
    `build_db_query_retriever_from_env` (the SQL half of sql_topn) nor
    `build_document_lookup_from_env` (the deterministic `document` table query
    of the `doc_download` flow) nor webapp.py's download endpoint
    (`document_store.py::resolve_document_download`) repeats this resolution
    on THEIR OWN -- single source, against the risk of the same
    env/bundled-fallback logic DRIFTING in three places (same rationale as the
    general principle: "reuse the same storage-access pattern/env, don't
    invent a second one")."""
    db_path = os.getenv("CHATBOT_DB_QUERY_DB_PATH")
    if db_path:
        return db_path
    if _BUNDLED_DB_PATH.exists():
        return str(_BUNDLED_DB_PATH)
    raise DbQueryError(
        "CHATBOT_DB_QUERY_DB_PATH tanımsız ve commit'li "
        f"{_BUNDLED_DB_PATH} da yok -- sql_topn/doc_download/indirme "
        "endpoint'i için hazır bir SQLite dosyası gerekir."
    )


def build_document_lookup_from_env() -> SqliteDocumentLookup:
    """Deterministic access to the `document` table for the `doc_download`
    flow -- built with the SAME env/bundled-fallback pattern of
    `resolve_db_path_from_env` (NO SECOND storage-access env INVENTED; see that
    function's docstring)."""
    return SqliteDocumentLookup(resolve_db_path_from_env())


def _document_lookup_from_env_or_none() -> SqliteDocumentLookup | None:
    """The OPTIONAL instance injected into `Orchestrator._document_hint` --
    exactly `build_document_lookup_from_env`, but when the DB is entirely
    unconfigured / the committed file is also missing (`DbQueryError`) it does
    NOT take down all of `create_app()`: it returns `None` (the hint feature is
    off). DELIBERATELY different from the UNCONDITIONAL call
    `build_flows_from_env` makes for the `doc_download` flow: that one is built
    on the assumption "no ready DB -> the app shouldn't boot at all" (in
    practice `resolve_db_path_from_env`'s committed `facts/db/specs.db`
    fallback always satisfies it) -- while the `/api/documents/<doc_id>`
    endpoint (`webapp.py::get_document`) catches the SAME error as a 503 AT
    REQUEST TIME (graceful-degrade, see that endpoint's docstring); this hint
    feature is architecturally OPTIONAL too -- it must NEVER sacrifice boot to
    the same class of error."""
    try:
        return build_document_lookup_from_env()
    except DbQueryError:
        return None


def build_db_query_retriever_from_env(*, sql_concurrency: int = 1, sql_gate_timeout_seconds: float | None = None):
    """Builds a real db_query retriever from `CHATBOT_DB_QUERY_DB_PATH` (see
    `.env.example`).

    `sql_concurrency` (config `[session]`) serializes the GPU-heavy SQL chain
    (linking->generation) process-wide: the returned retriever is wrapped in a
    `SerializedRetriever` (see gpu_gate.py). 1 = the two large models never
    race on a single 32GB VRAM. Above 1, real batch parallelism can be tried
    (if the models fit together).

    `sql_gate_timeout_seconds` (config `[session]`) goes to the SAME
    `SerializedRetriever` -- if a single SQL call exceeds it, the semaphore is
    released immediately and other users aren't locked out indefinitely (see
    the gpu_gate.py docstring; the root cause was a `max_tokens`
    misconfiguration in `text2sql/config.toml`).

    `SCHEMA_PATH`/`PROMPT_DIR` (text2sql's own mandatory/optional env fields)
    fall back to the committed `facts/db/schema.yaml` (produced by the facts
    component, read from here by PATH only) + this component's
    `text2sql/prompts/` if the user hasn't defined them in their own `.env`
    (`setdefault` semantics -- the user's own override always wins).
    `config.toml` is ALWAYS this component's committed
    `text2sql/config.toml` -- passed directly as a `Path` parameter, never
    dependent on the `DB_QUERY_TEXT2SQL_CONFIG` env var (CWD-independent,
    always finds the right file).

    `text2sql/prompts/linking.txt` + `generation_raw.txt` are copies of the
    package's own templates with ONE difference: a rule suppressing
    thinking/reasoning-trace output has been added (the text2sql-engine 0.2.0
    provider layer never forwards a field like `extra_body`/`reasoning_effort`
    to the OpenAI SDK call, and `StageSettings` in `config.toml` is a closed
    field set -- so the request-body suppression pattern in this component's
    `thinking.py` CANNOT be used against that package; adding the instruction
    to the template via `PROMPT_DIR` is the only seam within the current
    constraints). If `LINKING_MODEL`/`SQL_MODEL` are NOT thinking/reasoning
    models the rule is harmless (no-op); with a reasoning model the guarantee
    is only prompt-compliance -- less firm than `thinking.py`'s request-body
    suppression.
    """
    # Resolution is centralized in `resolve_db_path_from_env` now --
    # `doc_download`/the download endpoint share the SAME function; behavior is
    # identical (env unset -> committed facts/db/specs.db).
    db_path = resolve_db_path_from_env()
    # The TRIMMED schema that goes to the model -- NOT the full `schema.yaml`.
    # Not `setdefault`: if the key EXISTS WITH AN EMPTY VALUE in `.env`
    # (`SCHEMA_PATH=`, left blank instead of deleted as .env.example advises),
    # `setdefault` counts it as "present" and won't overwrite the empty string
    # -- `text2sql_native._require_env` then fails with "missing or empty".
    # Aligned with the `if db_path:` truthy pattern in `resolve_db_path_from_env`.
    if not os.environ.get("SCHEMA_PATH"):
        os.environ["SCHEMA_PATH"] = str(_BUNDLED_MODEL_SCHEMA_PATH)
    if not os.environ.get("PROMPT_DIR"):
        os.environ["PROMPT_DIR"] = str(_BUNDLED_PROMPT_DIR)
    # Citation columns are added by CODE, not by a prompt (the model shouldn't
    # be told about this bookkeeping at all, so it doesn't get confused).
    # `rewriters` is retrieval's existing `Text2SqlGenerator` seam -- before
    # the generated SQL runs, `product_code`/`key`/`source_file_name` are added
    # to `spec_value` projections (see sql_citation.py; DISTINCT/aggregate/
    # UNION queries are deliberately left alone).
    retriever = build_db_query_retriever(
        db_path=db_path,
        config_path=str(_BUNDLED_CONFIG_PATH),
        rewriters=[add_citation_columns],
    )
    return SerializedRetriever(
        retriever, concurrency=sql_concurrency, timeout_seconds=sql_gate_timeout_seconds
    )


def build_flows_from_env(cfg: ChatbotConfig | None = None) -> dict[str, Flow]:
    """Builds the flows referenced by `config/default.toml` [routing] with real
    retrievers: `default_topn`, `sql_topn`, `aggregation`, `comparison`,
    `recommendation`, `doc_download`, `no_retrieval`.

    Both are required: `sql_topn` needs top_n AND db_query (see
    flows/sql_topn.py). A party that only wants to test top_n can use
    `build_top_n_retriever_from_env()` + `DefaultTopNFlow` directly, without
    calling this function at all.

    `comparison` and `aggregation` SHARE the SAME `SqlTopNFlow` instance
    (`sql_topn_flow`) -- both repeatedly call the `SqlTopNFlow.run_pass` core
    (see flows/comparison.py, flows/aggregation.py). That's why no separate
    retriever is built; the single instance is also registered under the three
    flow names.

    `aggregation`'s `splitter` (see
    flows/aggregation.py::AggregationSplitter) stays `None` until retrieval's
    planned query-rewriting module ships -- the raw query runs as a single
    pass (the flow doesn't blow up).

    `comparison`'s `splitter` is `DeterministicComparisonSplitter` --
    retrieval's query rewriting is still absent, but this splitter is
    deterministic/regex-based (requires no LLM/GPU call, in line with the
    standing no-GPU-on-dev-box rule), so unlike `aggregation` it was not left
    `None` -- the real N-product comparison + clarification flow (see
    flows/comparison.py) is live in production from day one.
    """
    cfg = cfg or load_config()
    top_n_retriever = build_top_n_retriever_from_env()
    db_query_retriever = build_db_query_retriever_from_env(
        sql_concurrency=cfg.session.sql_concurrency,
        sql_gate_timeout_seconds=cfg.session.sql_gate_timeout_seconds,
    )

    sql_topn_flow = SqlTopNFlow(
        db_query_retriever, top_n_retriever=top_n_retriever,
        sql_k=cfg.flow.sql_k, topn_k=cfg.flow.sql_topn_k,
        few_max=cfg.result_shape.few_max,
    )
    # The "is this pinned session still relevant" decision now delegates (when
    # possible) to this shared LLM checker instead of a fixed cancel-word list
    # (see the continuation.py module docstring). Returns `None` when the
    # endpoint is unconfigured -- each of the three flows then falls back to
    # its own old word-list FALLBACK (zero behavior change).
    continuation_checker = build_continuation_checker_from_env()
    # Dynamic question pool for `recommendation` -- again the SAME
    # graceful-degrade principle (endpoint unconfigured -> `None`, the flow
    # falls back to its static emergency pool). `db_path`/`schema_yaml_path`
    # come from the ALREADY SHARED bundled-fallback resolvers used by
    # sql_topn/doc_download (`resolve_db_path_from_env`/`_BUNDLED_SCHEMA_PATH`)
    # -- no second storage-access env INVENTED.
    slot_selector = build_slot_selector_from_env()
    db_path = resolve_db_path_from_env()

    return {
        "default_topn": DefaultTopNFlow(top_n_retriever, k=cfg.flow.top_n_k),
        "sql_topn": sql_topn_flow,
        "aggregation": AggregationFlow(sql_topn_flow),
        "comparison": ComparisonFlow(
            sql_topn_flow, splitter=DeterministicComparisonSplitter(),
            continuation_checker=continuation_checker, few_max=cfg.result_shape.few_max,
        ),
        # `sql_topn_flow` shares the SAME `SqlTopNFlow` instance as
        # comparison/aggregation (same rationale) -- once clarification ends it
        # makes a single `run_pass` call.
        "recommendation": RecommendationFlow(
            sql_topn_flow, continuation_checker=continuation_checker,
            slot_selector=slot_selector, db_path=db_path, schema_yaml_path=_BUNDLED_SCHEMA_PATH,
            few_max=cfg.result_shape.few_max,
        ),
        # A deterministic `document` table query that never goes through LLM
        # SQL generation (see flows/doc_download.py) -- does NOT share a
        # retriever with `sql_topn`/`comparison`/...: it has its own
        # `DocumentLookup`.
        "doc_download": DocDownloadFlow(
            build_document_lookup_from_env(), continuation_checker=continuation_checker,
            few_max=cfg.result_shape.few_max,
        ),
        "no_retrieval": NoRetrievalFlow(),
    }


@dataclass
class OrchestratorBundle:
    """The triple returned by `build_orchestrator_from_env` -- webapp.py AND
    `medrag.worker.jobs` (the RQ worker processing WhatsApp messages) SHARE the
    SAME wiring (no code duplication). `wa_bot.py` no longer CALLS this
    factory -- the Flask process only does auth/idempotency/enqueue."""

    orchestrator: Orchestrator
    memory: ConversationMemory
    session_state: SessionStateStore


def _build_intent_chat_model(env: os._Environ | dict):
    """Builds the chat model for the intent classifier. DEFAULT: retrieval's
    own `build_default_chat_model` (`LLM_*`, plain `/v1`, no num_ctx) --
    retrieval stays provider-agnostic, behavior UNCHANGED from today.

    When `INTENT_PROVIDER=ollama` is EXPLICITLY set (chatbot/.env -- a
    role-based, configurable key instead of the machine-wide
    `OLLAMA_CONTEXT_LENGTH`), a `NativeOllamaChatModel` is built instead: it
    redirects the `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` that retrieval
    already reads (the target model doesn't change) to native `/api/chat` and
    forces `INTENT_NUM_CTX` (falling back to
    `answering_model._DEFAULT_OLLAMA_NUM_CTX` when empty -- the same
    `_resolve_num_ctx` pattern as `RECONCILER_NUM_CTX`/`CONTINUATION_NUM_CTX`)
    as `options.num_ctx` -- so alignment with the other roles sharing
    gemma4:31b (answering/reconciler/continuation) becomes possible, avoiding
    reload/cold-start thrash."""
    from medrag.api.retrieval.modules.intent_classification import (
        build_default_chat_model,
    )

    provider = (env.get("INTENT_PROVIDER") or "").strip().lower()
    if provider != "ollama":
        return build_default_chat_model()

    from medrag.api.answering_model import ProviderError, _resolve_num_ctx
    from medrag.api.ollama_chat_model import NativeOllamaChatModel

    base_url = (env.get("LLM_BASE_URL") or "").strip()
    model = (env.get("LLM_MODEL") or "").strip()
    if not base_url or not model:
        raise ProviderError(
            "INTENT_PROVIDER=ollama ama LLM_BASE_URL/LLM_MODEL boş "
            "(retrieval/.env -- intent classifier bu değerleri paylaşır)."
        )
    num_ctx = _resolve_num_ctx(env, "INTENT", provider)
    return NativeOllamaChatModel(
        ollama_base_url=base_url,
        ollama_model=model,
        num_ctx=num_ctx,
        timeout=float(env.get("LLM_TIMEOUT") or 60.0),
    )


def preamble_settings_from_config(
    cfg: ChatbotConfig, channel: str
) -> PreambleSettings | None:
    """`cfg.preamble` -> `PreambleSettings` (see preamble.py); `None` if this
    channel is NOT in the list (that service never prints bubbles).

    A separate function because `webapp.py`/`medrag.worker.jobs` call the SAME
    factory with different channels and the "is it on for this channel"
    decision must live in one place -- same channel-parameterized pattern as
    `conversation_logger_from_config`."""
    from medrag.api.preamble import PreambleSettings

    pre = cfg.preamble
    if not pre.enabled or channel not in pre.channels:
        return None
    return PreambleSettings(
        watchdog_seconds=pre.watchdog_seconds,
        fast_flows=frozenset(pre.fast_flows),
    )


def build_orchestrator_from_env(
    cfg: ChatbotConfig | None = None, *, channel: str = CHANNEL_WEB
) -> OrchestratorBundle:
    """The SHARED form of the Orchestrator wiring built by
    webapp.py::create_app() (`medrag.worker.jobs::_get_bundle` uses it too --
    `wa_bot.py` no longer CALLS this, see the `OrchestratorBundle` docstring)
    -- behavior/parameters IDENTICAL.

    `channel` is passed to the `Orchestrator` and from there to every
    `AnsweringModel.answer` call -- it selects which FORMATTING instruction
    (web's GFM tables vs. WhatsApp's plain-text/bold blocks, see
    answering_model.py::_formatting_for_channel) reaches the model.
    `webapp.py` passes `CHANNEL_WEB` (default), `medrag.worker.jobs` passes
    `CHANNEL_WHATSAPP`.

    `LLMIntentClassifier`/`build_default_chat_model` and
    `answering_model_from_env` are DELIBERATELY imported here (inside the
    function), not at MODULE level:
    `tests/test_webapp.py::_patch_wiring` monkeypatches these names through
    the `retrieval.modules.intent_classification`/`medrag.api.answering_model`
    modules' OWN attributes -- a local import reads that attribute FRESH on
    every call, while a module-level import freezes a reference at import time
    and never sees the patch. `_build_intent_chat_model` also local-imports
    `build_default_chat_model` for the SAME reason -- the patched path keeps
    working as long as tests don't set `INTENT_PROVIDER` (default empty !=
    "ollama")."""
    from medrag.api.answering_model import answering_model_from_env
    from medrag.api.retrieval.modules.intent_classification import LLMIntentClassifier

    cfg = cfg or load_config()
    memory = ConversationMemory()
    session_state = SessionStateStore()
    orchestrator = Orchestrator(
        classifier=LLMIntentClassifier(_build_intent_chat_model(os.environ)),
        router=build_router_from_env(cfg),
        # `cfg.visual.exclude_types` is passed from here --
        # answering_model.py doesn't read env; tuning lives in
        # config/default.toml (root CONFIG.md taxonomy).
        answering_model=answering_model_from_env(
            image_exclude_types=cfg.visual.exclude_types,
            # Output contract (see reply_contract.py) -- from config (tuning),
            # not env; same taxonomy as `image_exclude_types`.
            contract=cfg.answering.contract,
            contract_retries=cfg.answering.retries,
        ),
        memory=memory,
        reconciler=build_reconciler_from_env(cfg),
        session_state=session_state,
        hedge_threshold=cfg.session.hedge_confidence_threshold,
        # Handoff approval also uses the SAME shared checker -- functionally
        # identical to the instance `build_flows_from_env` builds itself (reads
        # the same env); constructing a second object makes NO network call
        # (see continuation.py).
        continuation_checker=build_continuation_checker_from_env(),
        channel=channel,
        # The preamble is switched per CHANNEL: a channel not in `channels`
        # gets `None` and the Orchestrator produces no bubbles at all.
        preamble=preamble_settings_from_config(cfg, channel),
        # So that `Orchestrator._document_hint` can produce a suggestion on
        # single-product SQL turns (product_fact/aggregation/visual_request)
        # when a real file exists -- the SAME `build_document_lookup_from_env()`
        # pattern used for the `doc_download` flow in `build_flows_from_env`,
        # NO SECOND env/access path INVENTED. But HERE the error is SWALLOWED
        # (see the `_document_lookup_from_env_or_none` docstring) -- this is an
        # optional hint; a missing DB must NEVER break app boot.
        document_lookup=_document_lookup_from_env_or_none(),
    )
    return OrchestratorBundle(orchestrator=orchestrator, memory=memory, session_state=session_state)


def build_router_from_env(cfg: ChatbotConfig | None = None) -> Router:
    cfg = cfg or load_config()
    return Router(build_flows_from_env(cfg), cfg.routing)


def build_reconciler_from_env(cfg: ChatbotConfig | None = None):
    """Builds the L0+L1 reconciler from env (see reconciler.py). The endpoint
    is the `RECONCILER_*` override or the `LLM_*` fallback (shares the small
    intent model). Thinking is controlled by `RECONCILER_THINKING`.
    `decay_window` comes from config.

    Returns `None` when the endpoint is unconfigured (base_url/model empty) --
    the Orchestrator then skips L0/L1 (the raw query is classified directly).
    """
    from medrag.api.reconciler import (
        LLMReconciler,
        _reconciler_endpoint,
        _reconciler_num_ctx,
        _reconciler_provider,
    )
    from medrag.api.retrieval.core import IntentLabel
    from medrag.api.thinking import thinking_off_body

    cfg = cfg or load_config()
    base_url, model, api_key = _reconciler_endpoint(os.environ)
    if not base_url or not model:
        return None
    # Fusion: if `fuse_intent` is on, the reconciler gets the label set
    # injected -> it also produces the intent and the orchestrator skips the
    # separate classify call. Off -> None: intent isn't asked, old two-call
    # path.
    intent_labels = [label.value for label in IntentLabel] if cfg.session.fuse_intent else None
    # Reload/cold-start fix: because the reconciler shares `LLM_*`, calling
    # that model from text2sql's linking role with native num_ctx collided with
    # the reconciler always calling num_ctx-less `/v1`, causing constant
    # reloads in Ollama. RECONCILER_PROVIDER/NUM_CTX (else LLM_*) are now
    # resolved and the native path is used.
    provider = _reconciler_provider(os.environ)
    num_ctx = _reconciler_num_ctx(os.environ, provider)
    return LLMReconciler(
        base_url=base_url, model=model, api_key=api_key,
        provider=provider, num_ctx=num_ctx,
        extra_body=thinking_off_body(os.environ, "RECONCILER"),
        decay_window=cfg.session.decay_window,
        intent_labels=intent_labels,
    )


__all__ = [
    "OrchestratorBundle",
    "build_db_query_retriever_from_env",
    "build_document_lookup_from_env",
    "build_flows_from_env",
    "build_orchestrator_from_env",
    "build_reconciler_from_env",
    "build_router_from_env",
    "build_top_n_retriever_from_env",
    "preamble_settings_from_config",
    "resolve_db_path_from_env",
]
