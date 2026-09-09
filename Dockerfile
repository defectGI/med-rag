# syntax=docker/dockerfile:1
# =============================================================================
# ONE Dockerfile, TWO runtime targets:
#   docker build --target runtime  -t <image>:<sha> .   -> web (thin)
#   docker build --target pipeline -t <image>-pipeline:<sha> .  -> nightly run
#
# STRICT RULE: both targets MUST be built from the SAME commit and carry the
# SAME sha tag -- otherwise the version drift this rule exists to prevent comes
# right back. This Dockerfile CANNOT guarantee that on its own (two separate
# `docker build` calls can straddle another commit); the guarantee belongs in
# CI, where both tags are produced from a single build job.
#
# Layer-order rule: pyproject.toml + uv.lock (+ packages/, see the note below)
# FIRST, then src/. Code changes never invalidate the dependency layer; rebuilds
# take seconds. `--frozen` matches the lockfile exactly.
# =============================================================================

# -----------------------------------------------------------------------------
# Builder (web): the SPA (Vite + React + shadcn-style UI) -- med-rag chat UI.
# Output is copied into the runtime images at /app/web/dist (webapp.py's
# `_dist_dir` default).
# -----------------------------------------------------------------------------
FROM node:22-alpine AS builder-web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Builder (serve): the `serve` extra from uv.lock + the project code, into /app/.venv.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder-serve

# uv is a PINNED version -- `latest` is forbidden (same principle as the Qdrant
# pin in compose): uv is the tool the lockfile was produced with; a surprise
# upgrade breaks build reproducibility. Upgrades must be deliberate, one-line changes.
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /usr/local/bin/
# Forbid uv from downloading its own Python into the image: use the
# python:3.12-slim interpreter (small + reproducible).
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1) Dependency layer. CAUTION — deliberate deviation from the original design:
# packages/ (text2sql-native, editable path dep) is copied BEFORE the first sync.
# Reason: `uv sync --no-install-project` skips the root package but still installs
# its `serve` extra's PATH dependency in editable mode (PEP 660); an editable
# install needs the package's source tree — without packages/ the first sync dies
# with "directory not found". packages/ is small pure-Python source; src/ is NOT
# needed in this layer thanks to --no-install-project (on code changes the
# dependency layer comes from cache).
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/
RUN uv sync --frozen --no-install-project --extra serve

# 2) Source layer: src/ enters here; the second sync only adds the root package
# (editable), it does not reinstall the dependencies. The SPA build joins it --
# runtime serves the chat UI from /app/web/dist (single origin, no CORS).
COPY src/ ./src/
COPY --from=builder-web /web/dist ./web/dist
RUN uv sync --frozen --extra serve

# -----------------------------------------------------------------------------
# Builder (pipeline): the `parse` extra ON TOP of the serve builder.
# Resolved from the same lockfile, the venv is a file-level SUPERSET of the serve
# venv -- it can be layered onto runtime (below).
# -----------------------------------------------------------------------------
FROM builder-serve AS builder-pipeline
RUN uv sync --frozen --extra serve --extra parse

# -----------------------------------------------------------------------------
# runtime target: the thin "serve" image — the web (chat UI + API) container.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user is MANDATORY. uid 1000: same as the `specs.db` permission
# requirement of the volume setup (volume files will be chowned to this uid).
RUN useradd -m -u 1000 app

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# venv + source. Editable installs bind via .pth to ABSOLUTE /app/... paths, so
# src/ and packages/ must live at the SAME paths in the image as in the builder
# (different path = broken import).
COPY --from=builder-serve --chown=app:app /app/.venv /app/.venv
COPY --from=builder-serve --chown=app:app /app/src /app/src
COPY --from=builder-serve --chown=app:app /app/packages /app/packages
# SPA (Vite + React) -- builder-serve'de `/app/web/dist`'e derlenmişti ama
# runtime bunu kopyalamıyordu: `_dist_dir()` None döner, `webapp.py` eski
# şablon arayüzüne (ürün asistanı) düşerdi. Med-rag'ın kendi SPA'sı
# (kütüphane + chat + notlar) bununla servis edilir.
COPY --from=builder-serve --chown=app:app /app/web /app/web

# Committed reference files (NOT code, DATA that the runtime reads):
# - facts/db/*.yaml : factory.py's bundled schema fallbacks
#   (_BUNDLED_SCHEMA_PATH / _BUNDLED_MODEL_SCHEMA_PATH). *.db deliberately does
#   NOT enter (.dockerignore) — specs.db comes from the named volume (which copy
#   of the DB ships is a deployment decision).
# - chatbot/text2sql/ : factory.py's bundled config.toml + prompts/ fallback
#   (_BUNDLED_CONFIG_PATH / _BUNDLED_PROMPT_DIR — stays in the old root-level
#   chatbot/ directory, was not moved with the code).
COPY --chown=app:app facts/db/ /app/facts/db/
COPY --chown=app:app chatbot/text2sql/ /app/chatbot/text2sql/
# strateji prompt'ları -- orchestrator STRATEGIES_DIR=/app/chatbot/strategies
# üzerinden okur; kapalıysa her flow strateji metni olmadan (boş string) koşar.
COPY --chown=app:app chatbot/strategies/ /app/chatbot/strategies/

# Writable roots (all owned by the app user):
# - /data, /uploads, /logs: compose volume mount points (sqlite, uploads, logs).
#   An EMPTY named volume is born owned by root while the container runs as
#   `app` (uid 1000) — if the mount point is not opened owned by this uid in the
#   image, CHATBOT_LOG_DIR's mkdir dies with Errno 13 (and pointing logs at the
#   read-only /corpus mount gives Errno 30 instead). Created under this uid in
#   the image, ownership is correct from the FIRST mount.
# - /app/chatbot/logs, /app/reports: logging_setup.py (chatbot/logs/) and
#   core/paths.py REPORTS_DIR (reports/) write relative defaults; the /app root
#   is owned by app too so an unexpected relative subdirectory can be created.
RUN mkdir -p /data /uploads /logs /app/chatbot/logs /app/reports \
    && chown app:app /app /data /uploads /logs /app/chatbot /app/chatbot/logs /app/reports

USER app

# Deliberately NO CMD/ENTRYPOINT: the same image can run under different
# commands — web (`gunicorn "medrag.api.webapp:create_app()"`, compose.yaml) or
# the panel (`python -m medrag.api.panel.app`). Not forcing a default prevents
# running with a wrong one.

# -----------------------------------------------------------------------------
# pipeline target: serve+parse image — `medrag-nightly` nightly runs (Coolify
# scheduled task) and manual pipeline runs. NOT a request-serving service: in
# compose.yaml it exists only as an idle (`sleep infinity`) exec target for
# scheduled runs.
# -----------------------------------------------------------------------------
FROM runtime AS pipeline

USER root

# SYSTEM packages of the parse extra:
# - tesseract-ocr + language data: pytesseract looks up PDF_TESSERACT_LANG keys
#   (e.g. "tur+eng", pdf_parser.py) — eng/osd arrive as Debian dependencies, but
#   listing them EXPLICITLY is safer under --no-install-recommends.
# - libgl1, libglib2.0-0: system requirements of the opencv-python pip wheel
#   (without libGL.so.1 / libglib-2.0.so.0, `import cv2` dies).
# - libgomp1: numpy/torch's OpenMP requirement (docling-ibm-models chain).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-tur \
        tesseract-ocr-osd \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The builder-pipeline venv is layered ON TOP of the serve venv: since both were
# PRODUCED from the same uv.lock via `uv sync --frozen`, the pipeline venv is a
# file-level superset — stacked copies create no skew (uv sync builds the FULL
# environment; no missing or stale files remain). This way the pipeline shares
# ALL of runtime's layers (shared layers in the registry = small storage).
COPY --from=builder-pipeline --chown=app:app /app/.venv /app/.venv

# The CODE of the `scan` stage. run_nightly.py::stage_scan imports it from a
# repo-root-relative path via path-based import (_DOCUMENT_INFO_DIR = parents[4]
# / "chatbot-corpus" / "document_info" -- /app/chatbot-corpus/document_info in
# the image); if the file is missing stage_scan dies with RuntimeError, i.e. the
# nightly run fails at its FIRST stage. `.dockerignore` has an exception for
# these three paths. Why only these three: classify_documents.py + its own
# `config.py` (it imports with a plain `from config import load_config`, so the
# file must sit in the same directory) + `config/default.toml` (scan tuning, no
# default in code -- a missing file makes the load fail loudly).
# The DATA (document_nodes.json, BELGELER/) is deliberately
# NOT here: it comes from the /corpus volume (same code/data split as the
# specs.db pattern). med-rag note: the original project's product catalog
# (product_info/) was removed -- scan runs in catalog-less mode.
COPY --chown=app:app chatbot-corpus/document_info/classify_documents.py chatbot-corpus/document_info/config.py /app/chatbot-corpus/document_info/
COPY --chown=app:app chatbot-corpus/document_info/config/ /app/chatbot-corpus/document_info/config/

# The canonical dictionary of facts (`run_full.py:91`, read at import time).
# The `_FROZEN_SPEC_KEYS` fallback (facts/experiments/...) deliberately does NOT
# enter the image -- that is the benchmark's frozen copy; production uses the
# canonical one.
COPY --chown=app:app tools/spec_schema/spec_keys.yaml /app/tools/spec_schema/spec_keys.yaml

# facts' specs.db lives on the `sqlite` VOLUME (/data/specs.db -- the chatbot's
# CHATBOT_DB_QUERY_DB_PATH points there too), but `facts/discover.py:63` HARDCODES
# the path: `_REPO_ROOT / "facts" / "db" / "specs.db"` -- no env override. If the
# two diverge, the nightly run writes to a DB the chatbot NEVER sees: a silent,
# late-discovered bug. The symlink is a deliberate DISTRIBUTION-layer fix: both
# paths resolve to one file without touching the facts code (frozen, takes no new
# features). TECH DEBT: this line goes away once discover.py gets a FACTS_DB_PATH
# env override.
RUN mkdir -p /app/facts/db && ln -sf /data/specs.db /app/facts/db/specs.db     && chown -h app:app /app/facts/db/specs.db

USER app
