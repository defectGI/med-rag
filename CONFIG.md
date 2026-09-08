# Configuration contract (medrag)

This repo separates config **by kind** — not by tool. Every value goes to exactly
one home based on what it is. The goal: the question "where does this value come
from?" always has a single, predictable answer.

This document is the enforceable reference: when adding a config value, first ask
"which kind is this?", then put it in the home the table assigns to that kind.

## Kinds and homes

| Kind | Example | Home | Versioned? |
|------|---------|------|------------|
| **Secret** | `LLM_API_KEY`, private endpoint | `.env` (gitignored) / secret file | **Never** |
| **Connection identity** | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `*_THINKING_ON`, `*_NUM_CTX` | `.env` | No (machine/deployment-specific) |
| **Tuning / settings** | thresholds (`0.7`), ratios (`table 0.5`), DPI, token budgets, concurrency, feature toggles, weights | `config/default.toml` (per component, committed) | **Yes** (reviewed in PR) |
| **Run input** | which folder, `--limit`, output path | CLI flag (argparse) | No (ephemeral) |
| **Profile** | `src/medrag/pipeline/cli/config/cfg*.env` (scenario bundle) | named `.env`/config file, selected with a single switch | Yes |

**Why does connection identity live in `.env` while tuning lives in TOML?**
Model + provider + endpoint + key is one "backing service" connection (12-factor);
it changes from machine to machine and lives alongside the secret. Model settings
are deliberately kept in one place (`parser/.env`) and read by both the pipeline and
the benchmark. Mathematical/behavioural tuning, in contrast, is part of the code's
logic: it is commentable, reviewable and deterministic — the natural content of TOML.

## Reserved homes for newer settings

These newer settings already have their homes fixed per the taxonomy, so
implementation does not have to re-ask "where does this go?":

| Value | Kind | Home | Note |
|---|---|---|---|
| `[pipeline.versions]` — stage versions (parser/chunker/facts/vectorize) | Tuning | `config/default.toml` | Bumped manually on every change that alters output; not bumped for refactors. |
| Nightly window (start 23:00, ceiling 8 h) | Tuning | `config/default.toml` | The timing itself lives in the deployment scheduler (Coolify scheduled task), but the **ceiling** belongs to the code — the run must be able to cut itself off. |
| Answer-latency ceiling (120 s) | Tuning | `config/default.toml` | Constrains `top_k` and model choice. |
| Report history directory | Run input / path | CLI + `config/default.toml` default | No retention limit — reports are kept indefinitely. |
| Backup target directory | Connection identity | `.env` | Machine/deployment-specific — a volume path in the container, something else locally. |
| `OLLAMA_NUM_PARALLEL` | Connection identity | `.env` | Machine-specific (VRAM). **Chosen together with `num_ctx`** — the product of the two determines VRAM usage. |
| Redis URL, queue name, job timeout, idempotency TTL | Connection identity (URL) + Tuning (durations) | `.env` + `config/default.toml` | The URL is machine-specific; the durations are behavioural. |
| File hosting the `issues` table | Connection identity | `.env` | The location is a deployment decision. |

**No setting changes without a deploy.** Every setting lives in `config.toml` under
git, so the git history shows who changed what and when. The price is that changing
one threshold requires a deploy — the accepted trade-off.

## Precedence (last writer wins)

```
TOML default  →  TOML override (*_CONFIG env → file)  →  env vars (.env)  →  CLI flag
```

This chain lets the `src/medrag/pipeline/cli/config/*.env` profiles (which set env
vars) override the parser's TOML defaults — the profile mechanism keeps working
while the defaults live in readable TOML.

## Loader pattern (per component, independent)

Reference: `src/medrag/pipeline/chunker/config.py` +
`src/medrag/pipeline/chunker/config/default.toml`.

- Each component carries its own `config/default.toml` (tuning) + `config.py`
  (pydantic loader). **No shared config package** — components deliberately do not
  import each other (see the layer rule in `README.md`).
- `config.py`: a single `load_config(override=None)`; reads `default.toml`,
  deep-merges an optional partial override TOML, validates with `model_validate`.
  `ConfigDict(extra="forbid")` (unknown keys rejected), **no** Python defaults on
  fields (a missing key fails loudly), `Field(ge/le)` + cross-field validators.
- **Injection**: the entry point (CLI/runner) reads env + TOML, builds `cfg` and
  passes it down; library code never touches `os.environ` directly. (Pragmatic
  exception in the parser: a cached-singleton `get_config()` — instead of an
  argument-explosion.)
- Secrets never enter TOML. A new env variable is also recorded in the relevant
  `.env.example`.

## Component status

| Component | Tuning | Connection/secret | Input | Status |
|-----------|--------|-------------------|-------|--------|
| **chunker** | `config/default.toml` | `.env` (`LLM_*`/`EMBEDDING_*`) | env + `--limit` | ✅ reference |
| **benchmark** | `config/default.toml` | `.env` (`VLM_*`/`LLM_*`), layered² | CLI flags + `BENCHMARK_*` env³ | ✅ |
| **chatbot-corpus** | `document_info/config/default.toml` | — (no secrets) | `.env` paths | ✅ |
| **parser** | `config/default.toml` | `.env` (`LLM_*`/`VLM_*`) | `.env` paths | ✅ |
| **pipeline** | `pipeline_config.toml`¹ | `.env` + profile | `.env` + `--profile`/flags | ✅ |
| **vectorize** | `config/default.toml` | `.env` (`EMBEDDING_*`/`QDRANT_*`) | env (`VECTORIZE_INPUT_DIR`) + `--limit`/`--force` | ✅ |
| **facts** | `config/default.toml`⁴ | `.env` (`CHATBOT_CORPUS_DIR`, `FACTS_LLM_*`) | env paths | ✅ |

¹ The pipeline loader is deliberately named **`pipeline_config`**, not `config`:
`run_parse_pipeline` prepends the parser to `sys.path` and imports parser modules;
a `config` module/directory would shadow `parser/config.py`.

⁴ The facts tuning lives in `src/medrag/pipeline/facts/config/default.toml`; its
loader is named `facts_config.py` for the same shadowing reason as note ¹ — the
facts entry points add `facts/` to `sys.path`, where a top-level `config` module
would shadow the parser's `config.py`.

² The benchmark loads its `.env` **layered** (highest precedence first:
`--env-file` → `benchmark/.env` → `./.env` → `parser/.env`); each file only fills
what the higher ones leave unset (the real environment overrides everything). This
way `benchmark/.env` can define its own judge model (different from the parser's)
while unset fields still come from `parser/.env`. This is a deliberate relaxation of
the "model settings live in `parser/.env`" rule: the benchmark's judge model must be
selectable independently of the parser's model. `benchmark/.env` is not tracked in
git; its template is `benchmark/.env.example`.

³ The benchmark's run inputs are still primarily CLI flags (`-f/-s/-o`), but each
falls back to an optional `.env` default (`BENCHMARK_FOLDER`/`SCOPE`/`OUT`); an
explicit flag overrides env. This matches the parser/pipeline pattern of "input
paths come from `.env`" (rows above).

## Env name schemes

Two decisions are pinned here as an **enforceable reference** so future changes do
not re-litigate "which name, which polarity".

**Qdrant connection — ONE scheme.** `QDRANT_URL` / `QDRANT_API_KEY` /
`QDRANT_COLLECTION` (no prefix) is the home for the writer (`vectorize`, which
produces the collection), `retrieval/modules/top_n` (its own process) and the
chatbot (`src/medrag/api/.env`). Earlier prefixed variants (`TOP_N_QDRANT_*`,
`CHATBOT_QDRANT_*`) were consolidated into this one scheme; there are **no**
back-compat aliases. The chatbot's isolation pattern — "do not read retrieval's
env; read your own env and pass values as **parameters** to
`build_default_retriever()`" — remains untouched (deliberate, see the docstring of
`src/medrag/api/factory.py`; the chatbot must be able to feed multiple retrieval
modules with different configs). Only the env **variable names** read by the three
components were aligned; the code structure is unchanged.

**"Thinking" toggles — TWO separate constants, deliberately not merged.** The
similar names are misleading; the polarities are OPPOSITE:

| Pattern | Scheme | Unset behaviour | Used by |
|---|---|---|---|
| A | `*_THINKING_ON` (truthy: `1`/`true`/`yes`/`on`) | **OFF** (suppressed) | `parser` (`LLM_/VLM_/VLM2_/VLM_CLASSIFY_THINKING_ON`), `benchmark`, `src/medrag/pipeline/cli/config/cfg_*.env` |
| B | `<PREFIX>_THINKING` (opt-in-off: only `="off"` suppresses) | **ON** (neutral) | `retrieval` (`LLM_THINKING`), `chatbot` (`CHATBOT_LLM_THINKING`, `RECONCILER_THINKING`, `CONTINUATION_THINKING`, `SLOT_SELECTOR_THINKING`) |

They were deliberately **not** forced into one name/one polarity. Pattern A's
"default closed" rationale (hybrid-reasoning models spending the whole token budget
on hidden thinking in short tasks and returning empty answers, observed in
`parser`/`benchmark`) and Pattern B's "default open/neutral" rationale (an injected
field risking a 400 on some servers, see the docstring of
`src/medrag/api/thinking.py`) solve different problems; collapsing to a single
polarity would silently invert one of the two defaults. Regression locks:
`src/medrag/pipeline/parser/tests/test_llm_client.py::test_d52_thinking_on_polarity_is_opposite_of_retrieval_llm_thinking`
and `src/medrag/api/retrieval/tests/modules/intent_classification/test_factory_thinking.py::test_d52_pre_unification_behavior_preserved`.

`*_THINKING_OFF_BODY` (Pattern B's adjunct, a JSON body override) carries no
polarity of its own; it only selects the body injected when Pattern B is "off" —
its default in both patterns is `{"reasoning_effort": "none"}`.

Note: `parser/llm/__init__.py::_env(prefix, name, fallback)` builds variable names
dynamically with a fallback chain (unset `VLM_*` → falls back to `LLM_*`, unset
`VLM_CLASSIFY_*` → falls back to `VLM_*`); grepping for literal env names will not
find these.
