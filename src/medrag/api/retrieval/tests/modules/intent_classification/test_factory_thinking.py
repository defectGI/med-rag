"""Unit tests for the LLM_THINKING -> extra_body logic in the factory.

Suppression is OPT-IN (only when LLM_THINKING is explicitly "off"). Tests the
pure helper `_thinking_extra_body` -- no langchain_openai / network needed."""

from __future__ import annotations

import pytest

from medrag.api.retrieval.modules.intent_classification.factory import (
    _thinking_extra_body,
)


def test_empty_when_unset(monkeypatch):
    # OPT-IN: when unset, nothing is injected (neutral).
    monkeypatch.delenv("LLM_THINKING", raising=False)
    monkeypatch.delenv("LLM_THINKING_OFF_BODY", raising=False)
    assert _thinking_extra_body() == {}


def test_empty_when_thinking_on(monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "on")
    assert _thinking_extra_body() == {}


def test_default_body_when_explicit_off(monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "off")
    monkeypatch.delenv("LLM_THINKING_OFF_BODY", raising=False)
    assert _thinking_extra_body() == {"reasoning_effort": "none"}


def test_override_body(monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "off")
    monkeypatch.setenv("LLM_THINKING_OFF_BODY", '{"think": false}')
    assert _thinking_extra_body() == {"think": False}


def test_non_object_body_raises(monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "off")
    monkeypatch.setenv("LLM_THINKING_OFF_BODY", "[1, 2]")
    with pytest.raises(ValueError):
        _thinking_extra_body()


def test_d52_pre_unification_behavior_preserved(monkeypatch):
    """Polarity guard for the two thinking-flag conventions.

    This `<PREFIX>_THINKING` flag is OPT-IN-OFF: unset means neutral and only an
    explicit "off" suppresses. Elsewhere a truthy `*_THINKING_ON` convention is
    used instead, where unset means "off". The two are deliberately kept apart,
    and this test pins this flag's polarity so a future collapse to a single
    naming scheme cannot flip it toward the other convention's."""
    # unset -> neutral (not suppressed), the opposite pole from the
    # `*_THINKING_ON` convention's "unset = off".
    monkeypatch.delenv("LLM_THINKING", raising=False)
    monkeypatch.delenv("LLM_THINKING_OFF_BODY", raising=False)
    assert _thinking_extra_body() == {}

    # only an explicit "off" suppresses (with the default body).
    monkeypatch.setenv("LLM_THINKING", "off")
    assert _thinking_extra_body() == {"reasoning_effort": "none"}

    # "on" (or any other value) stays neutral -- only "off" is special.
    monkeypatch.setenv("LLM_THINKING", "on")
    assert _thinking_extra_body() == {}
