# chatbot (code: `src/medrag/api/`)

The orchestration layer that wires the independent retrieval methods exposed
by the `retrieval` (`mkd-retriever`) package into a single conversational
flow: classify intent → reconcile session state → deterministic route →
run flow → call the main conversational model. `retrieval` deliberately does
not orchestrate (retrieval ARCHITECTURE.md #6) — this component is the
"consuming project" it expects.

**Code has moved:** the real code lives under `src/medrag/api/` (the package
is `medrag.api`, not `chatbot`). This root directory (`chatbot/`) hosts only
the `strategies/` directory (intent-guidance prompt drafts, DATA) and
`text2sql/` (text2sql-engine runtime wiring, DATA); it contains no code.
`.env`/`.env.example` and `config/default.toml` also live with the code
under `src/medrag/api/`.

## Overview

One of the independent components of medrag (parser, pipeline,
chatbot-corpus, benchmark, chunker, vectorize, retrieval, facts + this
component); none directly imports another — see the root `README.md`.
`medrag.api` and `medrag.pipeline` do not import each other (the binding is the
DB and the file system, locked by the `import-linter` contract in the root
`pyproject.toml`). Position in the pipeline: `vectorize` produces chunk
vectors in Qdrant and `facts` produces `specs.db` — this component only
**reads** both, as the terminal consumer; there is no step after it (it
returns the answer directly to the user).

```
                          ┌─────────────────────────┐
  user query        ───► │ SessionState reconciliation │  (L0 pin + L1 reconciler,
                          │ (is there a pin?            │   optional, wired via LLM_*)
                          │  is the query elliptical?)  │
                          └────────────┬─────────────┘
                                       ▼
                          ┌─────────────────────────┐
  or raw query       ───► │ intent_classification    │  (retrieval, usable)
                          │ -> IntentResult(label)   │
                          └────────────┬─────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │ Router (deterministic)   │  config/default.toml
                          │ intent -> Flow            │  [routing]
                          └────────────┬─────────────┘
                                       ▼
                    ┌──────────────────────────────────┐
                    │ Flow.run(query) -> FlowContext     │
                    │  - sql_topn: SQL + top_n combined   │
                    │  - default_topn: single top_n call  │
                    │  - doc_download: deterministic      │
                    │    document table query             │
                    │  - aggregation/comparison/           │
                    │    recommendation: sql_topn core    │
                    │    N times / clarification dialog   │
                    │  - no_retrieval: no calls at all     │
                    └────────────────┬─────────────────┘
                                     ▼
                    ┌──────────────────────────────────┐
                    │ Orchestrator.handle                │
                    │  strategies/<key>.md + context      │
                    │  -> single call to main model       │
                    │  -> reply_contract validation        │
                    └──────────────────────────────────┘
```

**Critical rule:** intermediate steps inside a Flow (the SQL string,
keywords, the initial top_n call) never enter the main conversational
model's message history — only `FlowContext.results` + the matching
strategy prompt + the previous turns' actual questions/answers
(`ConversationMemory`) reach it.

**Routing is deterministic:** the main conversational model does not pick
its own strategy at runtime; the intent → flow mapping is defined in
`src/medrag/api/config/default.toml` under `[routing]`. Adding a new path is
one row in that table plus a new file under `src/medrag/api/flows/`; existing
flows are not touched.

## Status

- `retrieval.modules.top_n` **usable** — real vectors are written to Qdrant
  (`vectorize`), and `default_topn`/`sql_topn` work against a real retriever.
- `src/medrag/api/factory.py::build_router_from_env()` builds real `top_n` +
  `db_query` retrievers from env and wires all flows.
- Two implementations exist for the main conversational model: a remote
  OpenAI-compatible client (`answering_model.OpenAICompatAnsweringModel`)
  and `ollama_chat_model.NativeOllamaChatModel` using Ollama's native
  `/api/chat` endpoint (for roles that need a forced `num_ctx` to avoid
  reload thrashing).
- Session-level extras: `SessionStateStore` + `Reconciler` (resolve
  elliptical queries, L0/L1), `ContinuationChecker` (is a pinned
  clarification/handoff still valid), `slot_selection` (dynamic
  clarification-question selection for `recommendation`) — all optional
  small-model calls; when `LLM_*` is unset they gracefully degrade to the
  old (hardcoded/skipped) behavior.
- A web chat UI (`webapp.py`), a WhatsApp bridge (`wa_bot.py`), and a
  read-only conversation observer panel (`panel/`) are included.
- Two separate logging channels: a diagnostic stream (single file) plus a
  per-conversation detailed record (see "Logging" below).
- `retrieval.modules.query_rewriting`, `raptor`, `agentic` are still
  `not started` — see their own `README.md` files
  (`src/medrag/api/retrieval/modules/`).
- No real endpoint has been contacted from this machine (the GPU rule from
  the root: LLM/VLM inference is not run on this machine, only HTTP
  requests are sent to remote Ollama / OpenAI-compatible endpoints) — the
  first end-to-end verification will run on the GPU machine that fills in
  `.env`.

## Setup

This component no longer carries its own `requirements.txt` — the root
`pyproject.toml` `[project.optional-dependencies]` `serve` group (flask,
gunicorn, qdrant-client, langchain-core/openai, sqlglot,
text2sql-engine-native, tiktoken, PyYAML) carries its runtime dependencies.
From the repo root:

```bash
pip install -e ".[serve,dev]"
cd src/medrag/api
cp .env.example .env      # fill in QDRANT_*/CHATBOT_DB_QUERY_DB_PATH/EMBEDDING_*/...
```

## Configuration

- **Tuning** (routing table, flow/session/logging/preamble/evidence
  settings): `src/medrag/api/config/default.toml`. Unknown keys are
  rejected; missing keys fail loudly.
- **Connection/secret** (endpoint, model name, API key, Qdrant/DB path):
  `src/medrag/api/.env` — template: `.env.example` (documented line by line,
  no real value is repeated here). Roughly three groups:
  - `CHATBOT_LLM_*` (the main conversational model),
    `RECONCILER_*`/`CONTINUATION_*`/`SLOT_SELECTOR_*`/`INTENT_PROVIDER`
    (optional small-model roles; if unset, they fall back to `LLM_*` or
    the feature is skipped),
  - `QDRANT_*` + `EMBEDDING_*` (top_n target — must match the
    collection name / model that `vectorize` writes),
  - `CHATBOT_DB_QUERY_DB_PATH` (optional; if empty, the committed
    `facts/db/specs.db` is used) plus text2sql's own required fields
    (`LINKING_*`/`SQL_*`/`OPENAI_API_KEY` — see "text2sql-engine wiring"
    below).
  - `retrieval/.env`'s `LLM_*` lives in a separate file (the
    intent_classification role's own) — this component's `.env` does not
    touch it.

## Usage

### Web chat UI

```bash
python -m medrag.api.webapp   # http://127.0.0.1:8507 (overridable via CHATBOT_WEB_PORT)
```

A single-page Flask app with `templates/index.html` — no build step.
Conversation memory lives only in RAM (`ConversationMemory`; the session
id is carried by a cookie); **history is not persisted as a product
feature** — when the server restarts, all sessions are reset. The
"Reset chat" button only clears the current session's memory. This does
NOT cover the diagnostic/quality LOG (a separate thing): each
conversation is also written under
`src/medrag/api/logs/conversations/web/<session_id>/` (messages + all
background stages; see "Logging" below).

**The right-hand panel streams background stages live** (SSE): which
intent was found, which SQL was produced + the rows returned, the
chunks returned by top_n (with id/score/text), which strategy was used.
`POST /api/chat/stream` returns a Server-Sent Events stream; the frontend
reads it via `fetch()` + manual SSE parsing (EventSource was not used
because it only supports GET). This trace NEVER reaches the main
conversational model — only the browser (see `trace.py` + the
orchestrator's "flow internals do not leak into the model" rule).
`POST /api/chat` (no trace, single-shot answer) is still available for
programmatic / simple use. `GET /api/images/<image_id>` and
`GET /api/documents/<doc_id>` are the image/document download endpoints.

### Test the retrieval layer standalone (without the web UI)

```bash
python -m medrag.api "de1000 fiyati nedir" --intent product_fact   # -> sql_topn (SQL + top_n)
python -m medrag.api "nasil calisir"       --intent doc_question    # -> default_topn (top_n only)
python -m medrag.api "hava durumu"         --intent out_of_scope     # -> no_retrieval (empty context)
```

Without invoking the main model, these show only the context returned by
the retrievers the Router wires up — useful for isolating top_n/db_query.
Both commands actually make embedding/Qdrant/SQL calls (HTTP requests to
remote services — they do not violate the root GPU rule: real inference is
NOT RUN on this machine; the requests already go to a remote machine).

### Connecting to WhatsApp

`src/medrag/api/wa_bot.py` is a separate entry point implementing the
HTTP+JSON contract of the sibling repo `wa-gateway` (INDEPENDENT of
`webapp.py`'s web chat UI — both can run simultaneously). Steps (in
order):

1. Start the bridge:
   ```bash
   python -m medrag.api.wa_bot
   ```
   Default port `WA_BOT_PORT=8003` (see `.env.example`).

2. Add a new `[[uygulama]]` block to `wa-gateway/config.toml` in the
   sibling repo:
   ```toml
   [[uygulama]]
   id          = "medrag"
   ad          = "Ürün Bilgi Asistanı"
   aciklama    = "Ürün/spec soruları"
   adres       = "http://127.0.0.1:8003"
   anahtar_env = "SERVIS_ANAHTARI_MEDRAG"
   erisim      = "ic"
   ```
   **`erisim` decision:** default `"ic"` is chosen — only numbers
   registered in wa-gateway's `veri/kisiler.csv` see this bot. Set to
   `"acik"` to expose it publicly (see wa-gateway README §4).

3. In the wa-gateway repo, generate the key (`bash kur.sh`) and copy the
   printed `SERVIS_ANAHTARI_MEDRAG=...` value.

4. Write that value into THIS component's OWN `.env`:
   `SERVIS_ANAHTARI=<value>`.

5. Validate the contract (from the wa-gateway repo, while wa_bot is
   running):
   ```bash
   bash dogrula_servis.sh http://127.0.0.1:8003 <key from step 3>
   ```

6. Restart both: `python -m medrag.api.wa_bot` + `bash baslat.sh` in the
   wa-gateway repo.

For the character budget (`WA_BOT_CONTEXT_CHAR_BUDGET`) and session TTL
(`WA_BOT_SESSION_TTL_HOURS`), see the `[wa_bot]` section in
`.env.example` — the budget is a rough estimate (not a real tokenizer)
and applies only to this bridge, not the web UI.

## text2sql-engine wiring (the SQL half of sql_topn)

`chatbot/text2sql/` (the root, the directory of this README — DATA, not
code) carries this component's **committed** runtime wiring:
`config.toml` (text2sql behavior settings — `sql_dialect=sqlite`,
`generation.output_mode=raw`) and `prompts/`. `src/medrag/api/factory.py`
locates these via
`Path(__file__).resolve().parents[3] / "chatbot" / "text2sql"`.

**`schema.yaml` / `schema_model.yaml` and `specs.db` live in the root
`facts/` component** — `facts/db/schema.yaml` +
`facts/db/schema_model.yaml` (model-facing, a projection with the
provenance columns trimmed) + `facts/db/specs.db` (an artifact lives
under the component that PRODUCES it). `chatbot` does NOT import the
`facts` code; it only reads these files by PATH via `medrag.core.paths`;
`factory.py` locates them automatically — no manual path setup needed.

`facts/db/specs.db` is COMMITTED in the repo (exception in the root
`.gitignore`) — no need to copy it manually. `CHATBOT_DB_QUERY_DB_PATH`
is filled in only if you want to use a different / newer DB.

text2sql-engine's own required env fields must also be filled in `.env`
(unprefixed — text2sql's own fixed names): `LINKING_PROVIDER` /
`LINKING_MODEL` / `LINKING_BASE_URL`, `SQL_PROVIDER` / `SQL_MODEL` /
`SQL_BASE_URL`, `OPENAI_API_KEY`. `.env.example` ships with proven
values (Ollama running gemma4:31b for linking + qwen2.5-coder:32b for
SQL).

## Logging — two channels

**1) Diagnostic stream (single file, all sessions mixed):**
`src/medrag/api/logs/chatbot.log` (`logging_setup.py`,
RotatingFileHandler + console). Collects the application's own logs,
Flask/werkzeug, AND text2sql's (SQL linking/generation) lines. Level:
`config/default.toml [logging]`; quick override via
`CHATBOT_LOG_LEVEL` / `CHATBOT_TEXT2SQL_LOG_LEVEL`.

**2) Per-conversation detailed record:** each conversation lives in
its OWN folder —

```
src/medrag/api/logs/conversations/
  web/<session_id>/         # browser UI (cookie session)
    meta.json               # channel, id, first/last seen, turn count
    turns.jsonl             # one JSON per turn: message + ALL intermediate steps + answer
    conversation.log        # readable dump of the same turn + all background logs
  whatsapp/<numara>/        # WhatsApp: folder name is the USER NUMBER
    meta.json  turns.jsonl  conversation.log
```

Toggle / redirect: `[conversation_log] enabled/dir/capture_background`
or `CHATBOT_CONVERSATION_LOG_ENABLED` / `_DIR` /
`_CAPTURE_BACKGROUND`. These files contain user messages and phone
numbers — `src/medrag/api/logs/` is in `.gitignore` and is not committed.

## Conversation observer panel (`panel/`)

`src/medrag/api/panel/` is a read-only observation layer that reads the
JSONL/log files above and summarizes them into its own SQLite
(similar to `lineage.db`, not committed to git) — see
`src/medrag/api/panel/README.md`. It contains `ingest.py` + `serve.py`;
for details see that README (it does not duplicate this README's
scope).

## Architecture

```
src/medrag/api/
  config.py               config/default.toml -> ChatbotConfig (routing + flow tuning)
  router.py                Router: intent -> Flow (deterministic, from config)
  orchestrator.py           Orchestrator: SessionState -> classify -> route -> flow -> memory -> answering model
  factory.py                env -> real retrievers + Router (build_router_from_env)
  answering_model.py          real AnsweringModel: remote OpenAI-compatible chat client
  ollama_chat_model.py         NativeOllamaChatModel: Ollama /api/chat, forces num_ctx
  session_state.py             SessionState/PinnedFlow (L0 session task-state)
  reconciler.py                 elliptical-query resolution (L1, optional small model)
  continuation.py                is a pinned clarification/handoff still valid (optional)
  slot_selection.py               recommendation's dynamic clarification-question selection
  memory.py                        ConversationMemory: RAM-only, no persistence/log
  reply_contract.py                 answering model's output envelope ({"reply","cited"})
  sql_citation.py / sql_evidence.py  citation column on [SQL] rows + evidence chunk resolution
  pin_relevance.py / preamble.py      pin relevance check / deterministic "thinking" bubbles
  chunk_store.py / document_store.py / image_store.py   path-read (non-imported) helpers
  gpu_gate.py                       concurrency/timeout gate for the SQL chain
  trace.py                          TraceFn: optional live-progress event (never reaches the model)
  webapp.py                         Flask: web chat UI + /api/chat, /api/chat/stream (SSE), /api/reset, /api/images, /api/documents
  wa_bot.py                          WhatsApp bridge (wa-gateway contract)
  panel/                             read-only conversation observer panel
  __main__.py                       CLI: `python -m medrag.api "question" --intent <label>` (retrieval-only test, no main model)
  flows/
    base.py                       Flow interface + FlowContext, Pin* protocols
    sql_topn.py                    SQL + two top_n combined flow
    default_topn.py                plain top_n flow
    doc_download.py                 deterministic document table query
    aggregation.py                   runs the sql_topn core N times across category/filter dimensions
    comparison.py                     runs the sql_topn core per product (N-product comparison)
    recommendation.py                 clarification dialog + single sql_topn call
    no_retrieval.py                    intents that need no retrieval (out_of_scope)
  retrieval/                        vendored `mkd-retriever` (core/modules/eval) — see retrieval/README.md
  config/default.toml               tuning: routing table + flow/session/logging/preamble/evidence settings
  .env / .env.example               connection/secret
  tests/                            offline, with fake Retriever/IntentClassifier/AnsweringModel

chatbot/                (this README's directory — NO code)
  strategies/            per-intent guidance-prompt draft (this component's
                          OWN copy -- why it differs from retrieval/strategies
                          is explained in strategies/README.md)
  text2sql/              text2sql-engine runtime wiring: config.toml +
                          prompts/ (COMMITTED). schema.yaml + specs.db are
                          NOT here -- they live under the root facts/db/
                          (committed); only read by PATH from here, code
                          is not imported
```

### Why it carries its own `strategies/` instead of `retrieval/strategies/`

`retrieval/strategies/*.md` is not included in the PyPI wheel (it's
absent from `retrieval/MANIFEST.in`) and the label → strategy mapping is
the consuming project's responsibility anyway (retrieval ARCHITECTURE.md
#6/#7). So `chatbot` carries its own copy of `strategies/` — the
retrieval drafts were used only as starting points, they are not
imported. Detail: [`strategies/README.md`](./strategies/README.md).

## Running tests

```bash
cd src/medrag/api
pytest                    # fully offline, with fake retriever/model/classifier
```

`pytest.ini` restricts `testpaths = tests` — running `pytest` with no
args only collects `src/medrag/api/tests/` and does NOT include the
vendored `retrieval/tests/` (deliberate: the two `tests/` packages
collide on the same name and cause `ModuleNotFoundError`). To test
`retrieval` modules separately:

```bash
pytest retrieval/tests
```

## Further information

- For component-specific rules (architectural boundaries, LLM-role
  isolation, etc.) see the documentation in the source tree.
