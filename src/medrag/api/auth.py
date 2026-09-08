"""Tek-kullanıcı şifre girişi (K3): MEDRAG_SIFRE ile basit oturum.

Kural: `MEDRAG_SIFRE` TANIMSIZ/BOŞSA kimlik doğrulama tamamen KAPALIDIR
(geriye uyumluluk + offline testler); bu durumda başlangıçta YÜKSEK SESLE
uyarılır. Tanımlıysa `/api/*` (auth uçları hariç) 401 döner; SPA statik
dosyaları herkese açık kalır -- giriş ekranı 401'i görüp gösterilir
(tek origin, statik dosyada sır yok).

Oturum: Flask `session` (imzalı çerez). Anahtar: `MEDRAG_GIZLI_ANAHTAR`
env; tanımsızsa şifreden türetilir (tek kullanıcılı ev kurulumu için
yeterli; iki değer de bağlantı/kimlik sınıfıdır -> `.env`, kök CONFIG.md
taksonomisi).
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

#: Kimlik doğrulama MUAF yolları (prefix eşleşmesi).
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
    """`create_app` içinden bir kez çağrılır: anahtar + before_request +
    `/api/auth/*` uçlarını bağlar."""
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
            # Doğrulama kapalıyken giriş her zaman "başarılı" -- SPA'nın
            # akışı bozulmasın; koruma zaten yok.
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
            return None  # statik dosyalar + SPA açık
        for prefix in _EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return None
        if _is_authed():
            return None
        return jsonify({"error": "giriş gerekli"}), 401

    def _is_authed() -> bool:
        return bool(not sifre or session.get("auth"))


__all__ = ["configure_auth", "password_configured"]
