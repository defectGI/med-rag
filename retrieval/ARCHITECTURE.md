# Architecture

## Why this document exists

This repository hosts several retrieval methods (top-N, query rewriting, intent
classification, RAPTOR, agentic — LangChain/LangGraph/LlamaIndex —, database
query retrieval, and others added over time) under one roof but **without
dependencies between them**.

Anti-goal: a Jenga tower. If methods are built on top of each other, with
dependencies between them, then changing or removing (or simply not using) one
piece brings the rest down. As this repo grows, preventing that is the main
job.

## Rules (hard rules)

1. **A module never imports another module.**
   Nothing inside `src/retrieval/modules/raptor/` may reference anything inside
   `src/retrieval/modules/agentic/`, and the reverse is also forbidden. If
   inter-module communication is needed, that is the responsibility of the
   consuming project (its orchestration layer) — this repo provides
   independent building blocks, not inter-module orchestration.

2. **Anything shared lives in `core/`, and `core/` stays small.**
   `core/` contains only: (a) the minimal interfaces (protocol/interface) that
   modules must conform to, and (b) common data schemas shared across modules
   (e.g. a `Document`, `RetrievalResult` type). `core/` never depends on a
   module's implementation detail — the dependency arrow always goes
   module → core, never core → module and never module → module.

3. **Dependencies are isolated per module.**
   Each module has its own optional-dependency group in `pyproject.toml`
   (`top_n`, `raptor`, `agentic`, ...). A library a module needs (e.g.
   `langgraph`) does not leak into another module's environment. Only
   consumers who use that module install that dependency.

4. **Every module must be understandable and testable on its own.**
   You should not have to read another module's code to understand one
   module. Tests live under `tests/modules/<name>/`, mapping one-to-one to
   the module.

5. **This repo does not depend on, and never imports, any specific product
   project (e.g. medrag).** Integration always happens from outside,
   through this repo's public API — the consuming project's code is never
   pulled into this repo.

6. **"Which methods to use, and how to integrate them" is out of scope for
   this repo.** This repo offers the building blocks (top-N, rewriting,
   RAPTOR, agentic, db query, ...); the consuming project decides how to
   combine them. Even if we wanted to define a "default pipeline" inside
   this repo, it should live as a separate, **optional** orchestration
   layer, distinct from the modules themselves.

7. **Intent classification returns only a label; it does not route.**
   The `intent_classification` module says "this query is `product_fact`"
   and stops. Embedding "if product_fact then go to db_query" into the
   module would make the intent module depend on other modules — that
   mapping is the consuming project's job. The same principle applies
   generally: if one module's output needs to know another module's name
   or type, the boundary is drawn wrong.

8. **LangChain rule: `langchain-core` is a dependency only where it is
   actually imported.** Common abstractions (Embeddings, VectorStore, chat
   model interfaces) come from `langchain-core` — a thin interface package.
   But a dependency only moves into base `dependencies` when `core/` (or
   the `eval/` surface every install pulls in) **actually `import`s** it.
   Today only `intent_classification` imports it → `langchain-core` stays
   in that module's extra. The day `core/` starts importing it for the
   shared embedding/vectorstore abstraction, it moves to the base. Heavy
   integration packages (`langchain-openai`, `langgraph`, `llama-index`,
   store clients, etc.) always stay in the optional-dependency group of
   the module that needs them; they never rise to base.

9. **Ingestion contract: the output formats of external projects are
   defined as schemas; no code dependency is established.** The
   `*.chunks.json`, `product_nodes.json`, `document_nodes.json` formats
   produced by medrag are defined as pydantic models inside
   `core/`; file paths come from config. medrag (or any other
   producer project) is never imported — any project that produces the
   same shape of data can feed this layer.

10. **Nothing that requires a GPU is run on this development machine.**
    LLM/VLM/embedding inference runs on a separate machine (behind an
    OpenAI-compatible endpoint); this repo only contains code that makes
    requests to remote endpoints. Tests run with fakes/mocks and never
    download a real model or start local inference.

## Dependency direction

```
modules/top_n  ─┐
modules/raptor  ─┤
modules/agentic ─┼──▶  core (interfaces + schemas)
modules/db_query─┤
...             ─┘

(no arrows between modules — they are never dependent on each other)
```

## Checklist for adding a new module

- [ ] Own folder under `src/retrieval/modules/<name>/`
- [ ] Own optional-dependency group in `pyproject.toml`
- [ ] Own section in `.env.example` (if needed)
- [ ] Own tests under `tests/modules/<name>/`
- [ ] Depends only on the interfaces in `retrieval.core`, not on another
      module