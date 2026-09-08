import io

import medrag.api.preflight as preflight_mod
from medrag.api.preflight import (
    QdrantProbe,
    RoleProbe,
    probe_qdrant,
    probe_role,
    qdrant_probe_from_env,
    roles_from_env,
    run_startup_preflight,
)


def _pong(reply: str):
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": reply}}]}

    return fake_post


# --- roles_from_env ----------------------------------------------------------


def test_roles_from_env_skips_unconfigured():
    # Only answering is configured; intent/linking/sql have no base_url -> skipped.
    env = {"CHATBOT_LLM_BASE_URL": "http://x/v1", "CHATBOT_LLM_MODEL": "m"}
    roles = roles_from_env(env)
    assert [r.name for r in roles] == ["answering"]


def test_roles_from_env_answering_no_suppression_by_default():
    # OPT-IN: THINKING unset -> no suppression, thinking allowed.
    env = {"CHATBOT_LLM_BASE_URL": "http://x/v1", "CHATBOT_LLM_MODEL": "m"}
    role = roles_from_env(env)[0]
    assert role.controllable is True
    assert role.want_thinking is True  # not suppressing -> leak flag not armed
    assert role.off_body == {}


def test_roles_from_env_answering_suppression_when_off():
    env = {"CHATBOT_LLM_BASE_URL": "http://x/v1", "CHATBOT_LLM_MODEL": "m",
           "CHATBOT_LLM_THINKING": "off"}
    role = roles_from_env(env)[0]
    assert role.want_thinking is False  # suppressing -> leak = failure
    assert role.off_body == {"reasoning_effort": "none"}


def test_roles_from_env_reconciler_falls_back_to_llm():
    # RECONCILER_* absent -> shares the LLM_* endpoint.
    env = {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small"}
    roles = {r.name: r for r in roles_from_env(env)}
    assert "reconciler" in roles
    assert roles["reconciler"].base_url == "http://llm/v1"
    assert roles["reconciler"].controllable is True


def test_roles_from_env_reconciler_override():
    env = {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small",
           "RECONCILER_BASE_URL": "http://rec/v1", "RECONCILER_MODEL": "rec"}
    roles = {r.name: r for r in roles_from_env(env)}
    assert roles["reconciler"].base_url == "http://rec/v1"
    assert roles["reconciler"].model == "rec"


def test_roles_from_env_continuation_falls_back_to_llm():
    # CONTINUATION_* absent -> shares the LLM_* endpoint (SAME pattern as the
    # `reconciler` role).
    env = {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small"}
    roles = {r.name: r for r in roles_from_env(env)}
    assert "continuation" in roles
    assert roles["continuation"].base_url == "http://llm/v1"
    assert roles["continuation"].controllable is True


def test_roles_from_env_continuation_override():
    env = {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small",
           "CONTINUATION_BASE_URL": "http://cont/v1", "CONTINUATION_MODEL": "cont"}
    roles = {r.name: r for r in roles_from_env(env)}
    assert roles["continuation"].base_url == "http://cont/v1"
    assert roles["continuation"].model == "cont"


def test_roles_from_env_answering_native_when_ollama_and_num_ctx():
    env = {"CHATBOT_LLM_BASE_URL": "http://x:11434", "CHATBOT_LLM_MODEL": "m",
           "CHATBOT_LLM_PROVIDER": "ollama", "CHATBOT_LLM_NUM_CTX": "8192"}
    role = roles_from_env(env)[0]
    assert role.provider == "ollama"
    assert role.num_ctx == 8192


def test_roles_from_env_answering_no_provider_stays_v1():
    # no provider given -> defaults to "openai", num_ctx stays None (old /v1
    # behavior unchanged).
    env = {"CHATBOT_LLM_BASE_URL": "http://x/v1", "CHATBOT_LLM_MODEL": "m"}
    role = roles_from_env(env)[0]
    assert role.provider == "openai"
    assert role.num_ctx is None


def test_roles_from_env_reconciler_num_ctx_falls_back_to_llm():
    env = {"LLM_BASE_URL": "http://llm:11434", "LLM_MODEL": "small",
           "LLM_PROVIDER": "ollama", "LLM_NUM_CTX": "16384"}
    roles = {r.name: r for r in roles_from_env(env)}
    assert roles["reconciler"].provider == "ollama"
    assert roles["reconciler"].num_ctx == 16384
    assert roles["continuation"].provider == "ollama"
    assert roles["continuation"].num_ctx == 16384


def test_roles_from_env_intent_never_gets_native_fields():
    # intent (retrieval's own classifier) does not support the native split
    # yet -- provider/num_ctx deliberately stay None.
    env = {"LLM_BASE_URL": "http://llm:11434", "LLM_MODEL": "small",
           "LLM_PROVIDER": "ollama", "LLM_NUM_CTX": "16384"}
    role = {r.name: r for r in roles_from_env(env)}["intent"]
    assert role.provider is None
    assert role.num_ctx is None


def test_roles_from_env_linking_sql_detect_only():
    env = {
        "LINKING_BASE_URL": "http://x/v1", "LINKING_MODEL": "gemma",
        "SQL_BASE_URL": "http://x/v1", "SQL_MODEL": "coder",
    }
    roles = {r.name: r for r in roles_from_env(env)}
    assert roles["linking"].controllable is False
    assert roles["linking"].off_body == {}  # detect-only, no control
    assert roles["sql"].controllable is False


# --- probe_role --------------------------------------------------------------


def _role(**kw):
    base = {"name": "answering", "base_url": "http://x/v1", "model": "m"}
    base.update(kw)
    return RoleProbe(**base)


def test_probe_role_healthy_pong(monkeypatch):
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("PONG"))
    res = probe_role(_role())
    assert res.ok is True
    assert "thinking off" in res.detail


def test_probe_role_native_path_hits_api_chat_with_num_ctx(monkeypatch):
    seen = {}

    def fake_post(url, payload, *, api_key, timeout):
        seen["url"] = url
        seen["payload"] = payload
        return {"message": {"content": "PONG"}}

    monkeypatch.setattr(preflight_mod, "_post_json", fake_post)
    res = probe_role(_role(base_url="http://x:11434", provider="ollama", num_ctx=8192))
    assert res.ok is True
    assert seen["url"] == "http://x:11434/api/chat"
    assert seen["payload"]["options"]["num_ctx"] == 8192


def test_probe_role_native_path_strips_v1_suffix(monkeypatch):
    seen = {}

    def fake_post(url, payload, *, api_key, timeout):
        seen["url"] = url
        return {"message": {"content": "PONG"}}

    monkeypatch.setattr(preflight_mod, "_post_json", fake_post)
    probe_role(_role(base_url="http://x:11434/v1", provider="ollama", num_ctx=8192))
    assert seen["url"] == "http://x:11434/api/chat"


def test_probe_role_native_path_sends_think_false_when_suppressing(monkeypatch):
    seen = {}

    def fake_post(url, payload, *, api_key, timeout):
        seen["payload"] = payload
        return {"message": {"content": "PONG"}}

    monkeypatch.setattr(preflight_mod, "_post_json", fake_post)
    probe_role(_role(provider="ollama", num_ctx=8192, off_body={"reasoning_effort": "none"},
                     want_thinking=False))
    assert seen["payload"]["think"] is False


def test_probe_role_no_num_ctx_still_uses_v1(monkeypatch):
    # provider="ollama" but no num_ctx -> old /v1 behavior (regression lock).
    seen = {}

    def fake_post(url, payload, *, api_key, timeout):
        seen["url"] = url
        return {"choices": [{"message": {"content": "PONG"}}]}

    monkeypatch.setattr(preflight_mod, "_post_json", fake_post)
    probe_role(_role(base_url="http://x/v1", provider="ollama", num_ctx=None))
    assert seen["url"] == "http://x/v1/chat/completions"


def test_probe_role_think_leak_fails(monkeypatch):
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("<think>hmm</think>PONG"))
    res = probe_role(_role())
    assert res.ok is False
    assert "<think>" in res.detail


def test_probe_role_detects_harmony_channel_leak_not_only_think_tag(monkeypatch):
    """Root-cause investigation: this check used to look ONLY for the literal
    `<think` and could NOT see the harmony/`<|channel|>` form actually observed
    in live testing -- preflight printed `OK (thinking off)` while real turns
    leaked. Detection now lives in `reply_contract.has_reasoning_marker` (the
    SAME marker list as the answering contract)."""
    monkeypatch.setattr(
        preflight_mod, "_post_json",
        _pong("analysis: trivial<|channel|>final<|message|>PONG"),
    )
    res = probe_role(_role())
    assert res.ok is False
    assert "muhakeme" in res.detail


def test_probe_role_detects_partial_channel_marker_leak(monkeypatch):
    # The broken/partial form observed in live testing.
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("Wait, let me re-check.<channel|>PONG"))
    res = probe_role(_role())
    assert res.ok is False


def test_probe_role_clean_pong_still_passes(monkeypatch):
    # False-positive protection: a clean answer must still be OK.
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("PONG"))
    res = probe_role(_role())
    assert res.ok is True


def test_probe_role_empty_content_flags_thinking(monkeypatch):
    monkeypatch.setattr(preflight_mod, "_post_json", _pong(""))
    res = probe_role(_role())
    assert res.ok is False
    assert "bastırılamadı" in res.detail


def test_probe_role_want_thinking_allows_no_pong(monkeypatch):
    # If thinking was explicitly requested ON, empty/missing content is expected -> OK.
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("PONG <think>...</think>"))
    res = probe_role(_role(want_thinking=True))
    assert res.ok is True
    assert res.detail == "thinking on"


def test_probe_role_detect_only_clean(monkeypatch):
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("PONG"))
    res = probe_role(_role(name="sql", controllable=False))
    assert res.ok is True
    assert "detect-only" in res.detail


def test_probe_role_retries_without_off_body_on_rejection(monkeypatch):
    from medrag.api.answering_model import ProviderError

    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload)
        if "reasoning_effort" in payload:
            raise ProviderError("HTTP 400: unknown parameter reasoning_effort")
        return {"choices": [{"message": {"content": "PONG"}}]}

    monkeypatch.setattr(preflight_mod, "_post_json", fake_post)
    res = probe_role(_role(off_body={"reasoning_effort": "none"}))
    assert res.ok is True
    assert len(calls) == 2


# --- run_startup_preflight ---------------------------------------------------


def test_run_startup_preflight_all_ok(monkeypatch):
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("PONG"))
    env = {"CHATBOT_LLM_BASE_URL": "http://x/v1", "CHATBOT_LLM_MODEL": "m"}
    out = io.StringIO()
    ok = run_startup_preflight(env, out=out)
    assert ok is True
    text = out.getvalue()
    assert "model preflight:" in text
    assert "[answering] OK" in text


def test_run_startup_preflight_reports_failure_but_returns_false(monkeypatch):
    # Suppression is ON (THINKING=off) but `<think>` leaks into the answer -> FAILED.
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("<think>x</think> PONG"))
    env = {"CHATBOT_LLM_BASE_URL": "http://x/v1", "CHATBOT_LLM_MODEL": "m",
           "CHATBOT_LLM_THINKING": "off"}
    out = io.StringIO()
    ok = run_startup_preflight(env, out=out)
    assert ok is False
    assert "FAILED" in out.getvalue()
    assert "[!]" in out.getvalue()  # warn + continue (not strict)


def test_run_startup_preflight_no_roles(monkeypatch):
    out = io.StringIO()
    ok = run_startup_preflight({}, out=out)
    assert ok is True
    assert "konfigüre edilmiş rol yok" in out.getvalue()


# --- qdrant --------------------------------------------------------------


def test_qdrant_probe_from_env_skips_unconfigured():
    assert qdrant_probe_from_env({}) is None
    assert qdrant_probe_from_env({"QDRANT_URL": "http://q"}) is None


def test_qdrant_probe_from_env_configured():
    env = {"QDRANT_URL": "http://q", "QDRANT_COLLECTION": "medrag_chunks",
           "QDRANT_API_KEY": "secret"}
    probe = qdrant_probe_from_env(env)
    assert probe == QdrantProbe(url="http://q", collection_name="medrag_chunks", api_key="secret")


def test_probe_qdrant_ok(monkeypatch):
    class _Info:
        points_count = 42

    monkeypatch.setattr(preflight_mod, "_qdrant_get_collection", lambda *a, **k: _Info())
    res = probe_qdrant(QdrantProbe(url="http://q", collection_name="c"))
    assert res.ok is True
    assert "collection='c'" in res.detail
    assert "points=42" in res.detail


def test_probe_qdrant_transport_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(preflight_mod, "_qdrant_get_collection", _boom)
    res = probe_qdrant(QdrantProbe(url="http://q", collection_name="c"))
    assert res.ok is False
    assert "connection refused" in res.detail


def test_run_startup_preflight_includes_qdrant(monkeypatch):
    monkeypatch.setattr(preflight_mod, "_post_json", _pong("PONG"))
    monkeypatch.setattr(preflight_mod, "_qdrant_get_collection", lambda *a, **k: object())
    env = {"CHATBOT_LLM_BASE_URL": "http://x/v1", "CHATBOT_LLM_MODEL": "m",
           "QDRANT_URL": "http://q", "QDRANT_COLLECTION": "c"}
    out = io.StringIO()
    ok = run_startup_preflight(env, out=out)
    assert ok is True
    assert "[qdrant" in out.getvalue()
    assert "OK" in out.getvalue()


def test_run_startup_preflight_qdrant_only_skips_no_roles_message(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(preflight_mod, "_qdrant_get_collection", _boom)
    env = {"QDRANT_URL": "http://q", "QDRANT_COLLECTION": "c"}
    out = io.StringIO()
    ok = run_startup_preflight(env, out=out)
    assert ok is False
    assert "konfigüre edilmiş rol yok" not in out.getvalue()
    assert "[qdrant" in out.getvalue()
