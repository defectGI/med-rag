"""Single-user password login (K3): a simple session using MEDRAG_SIFRE.

Rule: if `MEDRAG_SIFRE` is UNSET/EMPTY authentication is entirely OFF
(backward compatibility + offline tests); in that case it is warned LOUDLY
at startup. If set, `/api/*` (except auth endpoints) returns 401; the SPA
static files stay public -- the login screen sees the 401 and is shown
(single origin, no secret in the static file).

Session: Flask `session` (signed cookie). Key: `MEDRAG_GIZLI_ANAHTAR`
env; if unset it is derived from the password (sufficient for a single-user
home setup; both values are connection/identity class -> `.env`, root
CONFIG.md taxonomy).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

from flask import Flask, jsonify, request, session

logger = logging.getLogger("medrag.api.auth")

_PASSWORD_ENV = "MEDRAG_SIFRE"
_SECRET_ENV = "MEDRAG_GIZLI_ANAHTAR"

#: Authentication EXEMPT paths (prefix matching).
_EXEMPT_PREFIXES = ("/api/auth/",)


def password_configured() -> bool:
    return bool((os.environ.get(_PASSWORD_ENV) or "").strip())


def _secret_key() -> str:
    raw = (os.environ.get(_SECRET_ENV) or "").strip()
    if raw:
        return raw
    sifre = (os.environ.get(_PASSWORD_ENV) or "").strip()
    return hashlib.sha256(f"medrag-oturum:{sifre}".encode()).hexdigest()


def configure_auth(app: Flask) -> None:
    """Called once from inside `create_app`: wires the key + before_request +
    the `/api/auth/*` endpoints."""
    sifre = (os.environ.get(_PASSWORD_ENV) or "").strip()
    if sifre:
        app.secret_key = _secret_key()
        logger.info("kimlik doğrulama AÇIK (%s)", _PASSWORD_ENV)
    else:
        logger.warning(
            "kimlik doğrulama KAPALI: %s tanımsız -- API şifresiz erişilebilir "
            "(ev ağı dışına açmayın)", _PASSWORD_ENV)

    @app.post("/api/auth/login")
    def auth_login():
        data = request.get_json(force=True, silent=True) or {}
        girilen = (data.get("password") or "").strip()
        if not sifre:
            # With auth off, login always "succeeds" -- don't break the SPA's
            # flow; there is no protection anyway.
            session["auth"] = True
            return jsonify({"authenticated": True})
        if not hmac.compare_digest(girilen, sifre):
            return jsonify({"error": "şifre hatalı"}), 401
        session["auth"] = True
        session.permanent = True
        return jsonify({"authenticated": True})

    @app.post("/api/auth/logout")
    def auth_logout():
        session.pop("auth", None)
        return jsonify({"authenticated": False})

    @app.get("/api/auth/session")
    def auth_session():
        return jsonify({"authenticated": _is_authed(), "required": bool(sifre)})

    @app.before_request
    def _guard():
        if not sifre:
            return None
        path = request.path
        if not path.startswith("/api/"):
            return None  # static files + SPA are open
        for prefix in _EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return None
        if _is_authed():
            return None
        return jsonify({"error": "giriş gerekli"}), 401

    def _is_authed() -> bool:
        return bool(not sifre or session.get("auth"))


__all__ = ["configure_auth", "password_configured"]
