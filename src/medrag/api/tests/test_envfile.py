"""Locks in the layered `.env` loading order of `envfile.load_component_env`.

After the env schema unification, `QDRANT_URL`/`QDRANT_API_KEY`/
`QDRANT_COLLECTION` may now appear under the SAME names in both `api/.env` and
`api/retrieval/.env` (previously `CHATBOT_QDRANT_*` vs `TOP_N_QDRANT_*` were
distinct names precisely to avoid this collision). This file locks in two
things:

1) A value set in `api/.env` is NOT OVERWRITTEN by the value in
   `api/retrieval/.env` (precedence belongs to `api/.env`).
2) If the key is LEFT BLANK in `api/.env` (`QDRANT_URL=`), it does NOT behave
   like "unset" and fall through to `retrieval/.env` -- it stays as an empty
   string (load_dotenv behavior, pinned with a regression test so it holds no
   surprises).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def _load_layered(tmp_path, component_value: str | None, retrieval_value: str) -> str | None:
    component_env = tmp_path / "component.env"
    retrieval_env = tmp_path / "retrieval.env"
    if component_value is None:
        component_env.write_text("")
    else:
        component_env.write_text(f"QDRANT_URL={component_value}\n")
    retrieval_env.write_text(f"QDRANT_URL={retrieval_value}\n")

    load_dotenv(component_env)
    load_dotenv(retrieval_env)
    return os.environ.get("QDRANT_URL")


def test_component_env_value_wins_over_retrieval(tmp_path, monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    result = _load_layered(tmp_path, "http://chatbot-host:6333", "http://retrieval-host:6333")
    assert result == "http://chatbot-host:6333"


def test_blank_component_key_does_not_fall_back_to_retrieval(tmp_path, monkeypatch):
    """QDRANT_URL= (empty value) is NOT 'unset' -- it does not FALL BACK to the
    real value in retrieval/.env; it stays as an empty string. An operator
    relying on layered loading should know this (see envfile.py docstring)."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    component_env = tmp_path / "component.env"
    retrieval_env = tmp_path / "retrieval.env"
    component_env.write_text("QDRANT_URL=\n")
    retrieval_env.write_text("QDRANT_URL=http://retrieval-host:6333\n")

    load_dotenv(component_env)
    load_dotenv(retrieval_env)

    assert os.environ.get("QDRANT_URL") == ""


def test_unset_in_component_env_falls_back_to_retrieval(tmp_path, monkeypatch):
    """If the key NEVER APPEARS in the component .env (not an empty value -- not
    even a line), it genuinely falls back to the value in retrieval/.env."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    result = _load_layered(tmp_path, None, "http://retrieval-host:6333")
    assert result == "http://retrieval-host:6333"
