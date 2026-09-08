"""Does the write token behave fail-closed -- panel/auth.py."""

from __future__ import annotations

from medrag.api.panel import auth


def test_write_disabled_when_token_unset(monkeypatch):
    monkeypatch.delenv(auth.ENV_WRITE_TOKEN, raising=False)
    assert auth.write_token_configured() is False
    # A blank token counts as unset -- it cannot be bypassed by a matching
    # empty value (fail-closed).
    assert auth.check_write_token("") is False
    assert auth.check_write_token("herhangi-bir-sey") is False


def test_write_disabled_when_token_blank(monkeypatch):
    monkeypatch.setenv(auth.ENV_WRITE_TOKEN, "   ")
    assert auth.write_token_configured() is False
    assert auth.check_write_token("   ") is False


def test_write_allowed_with_matching_token(monkeypatch):
    monkeypatch.setenv(auth.ENV_WRITE_TOKEN, "gizli-token")
    assert auth.write_token_configured() is True
    assert auth.check_write_token("gizli-token") is True
    assert auth.check_write_token(" gizli-token ") is True  # whitespace is stripped


def test_write_rejected_with_wrong_token(monkeypatch):
    monkeypatch.setenv(auth.ENV_WRITE_TOKEN, "gizli-token")
    assert auth.check_write_token("baska-token") is False
    assert auth.check_write_token(None) is False
