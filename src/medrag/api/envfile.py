"""`.env` load order: this component's own `.env` + retrieval's `.env`.

`retrieval.modules.intent_classification.build_default_chat_model` reads the
plain `LLM_BASE_URL`/`LLM_MODEL` (NOT this component's `CHATBOT_LLM_*`) --
those live in `retrieval/.env` (model settings kept in one place). If
`chatbot`'s own `.env` is loaded only from the process working directory
(`cwd`), `retrieval/.env` is never read and `LLM_*` stays empty.

The pattern matches layered `.env` loading: first this component's own `.env`
(higher priority), then `retrieval/.env` (only fills keys the former left
unset -- `load_dotenv` by default never overwrites an existing `os.environ`
key). `CHATBOT_LLM_*` and `LLM_*` carry different names, so there is no
practical collision.

The shared names are `EMBEDDING_*` and `QDRANT_URL` / `QDRANT_API_KEY` /
`QDRANT_COLLECTION` -- both `.env` files may set them under the SAME name (all
components are aligned on a single Qdrant). Because `api/.env` loads first,
whatever it sets WINS there (the value in `api/retrieval/.env` is silently
ignored) -- intentional, since both are expected to point at the same Qdrant;
if they ever needed to point at different ones, this layered loading can NOT
separate them. WARNING: leaving a key empty in `api/.env` (`QDRANT_URL=`)
loads an "empty string", not "unset" -- `load_dotenv` still counts it as
"existing" and blocks the real value from `retrieval/.env`; do not rely on
emptiness to fall back to retrieval's value. To share a value genuinely,
DELETE the line from `api/.env` entirely.
"""

from __future__ import annotations

from pathlib import Path

# src/medrag/api/envfile.py -> parent = component root (src/medrag/api/);
# retrieval is a SUB-package of api/ now: src/medrag/api/retrieval/.env
_COMPONENT_DIR = Path(__file__).resolve().parent
_DEFAULT_RETRIEVAL_ENV = _COMPONENT_DIR / "retrieval" / ".env"


def load_component_env() -> None:
    """Loads `api/.env` + `api/retrieval/.env` (in that order; the second only
    fills gaps) into the process environment. If `api/retrieval/.env` is
    missing, `load_dotenv` silently no-ops -- it does not raise."""
    from dotenv import load_dotenv

    load_dotenv(_COMPONENT_DIR / ".env")
    load_dotenv(_DEFAULT_RETRIEVAL_ENV)


__all__ = ["load_component_env"]
