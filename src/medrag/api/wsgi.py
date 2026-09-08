"""Production WSGI entry point.

The chat UI can also be served under gunicorn via this module
(`gunicorn ... medrag.api.wsgi:app`); `python -m medrag.api.webapp` (Flask dev
server) is for local runs. The compose.yaml `web` service uses gunicorn's
factory syntax directly (`gunicorn ... "medrag.api.webapp:create_app()"`),
so this module is an optional, equivalent entry point.

The extra `.env` load is deliberately ABSENT: `create_app` reads env from the
process environment, and inside the container compose's `env_file: .env`
fills it -- the image contains no `.env` file (`.dockerignore`, secret
hygiene). `create_app()` does NOT touch the network at import time, so
gunicorn loads it safely even without preload.
"""

from __future__ import annotations

from medrag.api.webapp import create_app

app = create_app()
