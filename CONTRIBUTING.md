# Contributing

Thanks for your interest in med-rag. This file gives the ground rules for a
contribution that will actually land.

## Scope

- The project is a **medical document RAG assistant**. The non-negotiable
  product rule is **mandatory source citation** for every factual claim — a
  contribution must never weaken that.
- The **medical disclaimer** and the safety line in answers are intentional;
  do not remove them.

## Developer setup

```bash
uv sync --extra parse --extra serve --extra dev
pytest src/ tools/ -q
```

The test suite is **offline** (no network, model or API key). Some dormant-path
tests (specs.db / text2sql / nightly-report status strings) carry pre-existing
failures on a clean checkout; see [`PLAN.md`](PLAN.md). The quality gate that
must stay green is:

```bash
ruff check src tools
python -m pytest src/medrag/tests/test_import_contracts.py -q
```

## Conventions

- **All documentation, comments and docstrings are in English.** User-facing
  strings may be Turkish/English bilingual (see the language policy).
- **Comments are for "why", not "what".** Prefer explaining a decision and its
  trade-off over restating the code.
- **Directory statements of commitment** are in the root `README.md`,
  `CONFIG.md`, `DEPLOY.md`; update them if you change behavior.

## Layer rule

- `medrag.api` and `medrag.pipeline` must never import each other; the coupling
  is the database and the filesystem (`import-linter` enforces this).
- `medrag.core` sits at the bottom and must not import the layers above it.

## Releases

- Keep the version in `pyproject.toml` and the stage `*_version` constants
  (`[pipeline.versions]`) in sync when a pipeline stage's output changes.
- Run `ruff check src tools` and the layer-contract test before submitting.
