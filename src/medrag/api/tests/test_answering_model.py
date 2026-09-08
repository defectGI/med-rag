import asyncio
import json

import pytest

from medrag.api import answering_model as answering_model_mod
from medrag.api.answering_model import (
    _CONTRACT_FALLBACK_REPLY,
    _LANGUAGE_REMINDER,
    OpenAICompatAnsweringModel,
    ProviderError,
    _build_messages,
    _render_context,
    _source_suffix,
    _visible_images,
    answering_model_from_env,
    extract_evidence_chunks,
    extract_visible_documents,
    extract_visible_images,
)
from medrag.api.flows.base import FlowContext
from medrag.api.memory import Message
from medrag.api.reply_contract import REPLY_CONTRACT_NOTE
from medrag.api.retrieval.core import RetrievalResult


def _envelope(reply: str, cited: list[str] | None = None) -> str:
    """A contract-conformant model output (see reply_contract.py) -- the
    real model now returns this envelope instead of raw text, so the fake
    `_post_json`s return it too."""
    return json.dumps({"reply": reply, "cited": cited or []})


def test_build_messages_includes_strategy_context_and_history():
    context = FlowContext(results=[])
    history = [Message(role="user", content="onceki soru"), Message(role="assistant", content="onceki cevap")]

    messages = _build_messages("yeni soru", context, "STRATEJI METNI", history)

    assert messages[0]["role"] == "system"
    assert "STRATEJI METNI" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "onceki soru"}
    assert messages[2] == {"role": "assistant", "content": "onceki cevap"}
    # The user's real question is no longer the LAST message -- a separate
    # language-reminder message comes AFTER it (a recency/position fix, see
    # the _LANGUAGE_REMINDER docstring note). A reply-contract note was also
    # inserted in between (see reply_contract.py), hence the question at -3.
    assert messages[-3] == {"role": "user", "content": "yeni soru"}
    assert messages[-2] == {"role": "system", "content": REPLY_CONTRACT_NOTE}
    assert messages[-1]["role"] == "system"
    assert "same language" in messages[-1]["content"].lower()


def test_build_messages_default_channel_instructs_gfm_table():
    # If no channel is passed (default CHANNEL_WEB) the GFM table instruction
    # rendered by the webapp must go through -- the old behavior.
    messages = _build_messages("soru", FlowContext(results=[]), "", [])
    assert "GitHub-flavored markdown table" in messages[0]["content"]
    assert "WhatsApp" not in messages[0]["content"]


def test_build_messages_whatsapp_channel_forbids_table_uses_bold_bullets():
    # WhatsApp cannot render real tables, so the model must not produce
    # tables AT THE SOURCE -- rather than relying on post-processing (the
    # first, since-removed fix attempt).
    messages = _build_messages(
        "soru", FlowContext(results=[]), "", [], channel="whatsapp"
    )
    content = messages[0]["content"]
    assert "GitHub-flavored markdown table" not in content
    assert "NEVER produce" in content
    assert "SINGLE asterisk" in content


def test_build_messages_always_includes_base_persona_even_with_empty_strategy():
    # With an empty strategy (e.g. the out_of_scope.md stub) plus empty
    # sources, a model without identity produced irrelevant meta-output. The
    # base persona must ALWAYS be in the system message, even when the
    # strategy is empty.
    messages = _build_messages("selam", FlowContext(results=[]), "", [])
    # med-rag persona: tıbbi referans asistanı, kanıt-first.
    assert "medical reference assistant" in messages[0]["content"]
    assert "NEVER" in messages[0]["content"]


def test_build_messages_language_reminder_is_always_last_regardless_of_history():
    """Small/local models could ignore the single sentence buried in the
    middle of _BASE_PERSONA and answer in English instead of the dominant
    language of the user query. Fix: a separate language-reminder message is
    ALWAYS appended as the last message -- however many history turns there
    are, whatever strategy/session_goal/followups/hedge combination is in
    play, it comes even AFTER the user question that follows the conversation
    history (recency: the position closest to the generation, hence the
    highest attention weight)."""
    history = [
        Message(role="user", content="ilk soru"), Message(role="assistant", content="ilk cevap"),
        Message(role="user", content="ikinci soru"), Message(role="assistant", content="ikinci cevap"),
    ]
    messages = _build_messages(
        "son soru", FlowContext(results=[]), "STRATEJI", history,
        session_goal="devam eden amaç", hedge=True, followups="bir öneri",
    )
    # The reply-contract note was added, but the language reminder's
    # last-message guarantee was DELIBERATELY kept (verified by live
    # observation) -- the contract note comes BEFORE it.
    assert messages[-1] == {"role": "system", "content": _LANGUAGE_REMINDER}
    assert messages[-2] == {"role": "system", "content": REPLY_CONTRACT_NOTE}
    assert messages[-3] == {"role": "user", "content": "son soru"}


def test_build_messages_language_reminder_references_users_message_not_a_fixed_language():
    """A fixed language name (e.g. "Turkish") does not classify -- the
    reminder references the user's own message, so it adapts automatically
    to languages beyond Turkish/English (without a new language-detection
    dependency or LLM call)."""
    messages = _build_messages("hello", FlowContext(results=[]), "", [])
    content = messages[-1]["content"]
    assert "the user's message above" in content.lower()


def test_render_context_includes_doc_heading_and_page_range():
    result = RetrievalResult(
        id="c1",
        score=0.9,
        text="metin",
        metadata={
            "doc_id": "PN1022",
            "heading_path": ["Specifications", "Environmental"],
            "page_start": 3,
            "page_end": 4,
        },
    )
    rendered = _render_context(FlowContext(results=[result]))
    assert "doc=PN1022" in rendered
    assert "section=Specifications > Environmental" in rendered
    assert "page=3-4" in rendered


def test_render_context_single_page_has_no_range():
    result = RetrievalResult(
        id="c1", score=0.9, text="metin", metadata={"page_start": 5, "page_end": 5}
    )
    rendered = _render_context(FlowContext(results=[result]))
    assert "page=5" in rendered
    assert "page=5-5" not in rendered


def test_render_context_sql_row_without_source_fields_has_no_suffix():
    # SQL row metadata is column->value; there is no doc_id/heading_path.
    result = RetrievalResult(id="row1", score=1.0, text="ad=PN1015", metadata={"ad": "PN1015"})
    rendered = _render_context(FlowContext(results=[result]))
    assert "[SQL][row1] (score=1.000)\nad=PN1015" in rendered


def test_render_context_tags_topn_result_as_doc():
    result = RetrievalResult(
        id="c1", score=0.9, text="metin", metadata={"doc_id": "PN1022"}
    )
    rendered = _render_context(FlowContext(results=[result]))
    assert rendered.startswith("[DOC][c1]")


def test_render_context_tags_sql_result_as_sql():
    result = RetrievalResult(id="row1", score=1.0, text="ad=PN1015", metadata={"ad": "PN1015"})
    rendered = _render_context(FlowContext(results=[result]))
    assert rendered.startswith("[SQL][row1]")


def test_build_messages_notes_empty_context():
    context = FlowContext(results=[])
    messages = _build_messages("soru", context, "", [])
    assert "no relevant source found" in messages[0]["content"]


# --- followups ----------------------------------------------------------------


def test_build_messages_includes_followups_when_present():
    messages = _build_messages(
        "soru", FlowContext(results=[]), "", [],
        followups="PN1309: operating_temperature",
    )
    assert "PN1309: operating_temperature" in messages[0]["content"]


def test_build_messages_omits_followups_when_empty():
    messages = _build_messages("soru", FlowContext(results=[]), "", [], followups="")
    # when empty, no extra system part must be added -- only base persona +
    # FORMATTING (channel-based, ALWAYS added, see _formatting_for_channel)
    # + sources = 2 separators.
    assert messages[0]["content"].count("---") == 2


def test_answer_posts_messages_and_returns_content(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": {"content": _envelope("merhaba!")}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)

    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m")
    context = FlowContext(results=[])
    reply = asyncio.run(model.answer("soru", context, "", []))

    assert reply == "merhaba!"
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["payload"]["model"] == "m"


def test_answer_merges_extra_body_for_thinking_off(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": _envelope("ok")}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(
        base_url="http://x/v1", model="m", extra_body={"reasoning_effort": "none"}
    )
    asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert captured["payload"]["reasoning_effort"] == "none"


def test_answer_retries_without_extra_body_when_rejected(monkeypatch):
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload)
        if "reasoning_effort" in payload:
            raise ProviderError("HTTP 400: unknown parameter 'reasoning_effort'")
        return {"choices": [{"message": {"content": _envelope("ok")}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(
        base_url="http://x/v1", model="m", extra_body={"reasoning_effort": "none"}
    )
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == "ok"
    # First attempt had the body (rejected), second attempt body-less (succeeded).
    assert len(calls) == 2
    assert "reasoning_effort" in calls[0]
    assert "reasoning_effort" not in calls[1]


def test_answer_unrelated_error_not_retried(monkeypatch):
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload)
        raise ProviderError("cannot reach http://x/v1: connection refused")

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(
        base_url="http://x/v1", model="m", extra_body={"reasoning_effort": "none"}
    )
    with pytest.raises(ProviderError):
        asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert len(calls) == 1  # a transport error is not retried


def test_answer_malformed_response_raises(monkeypatch):
    monkeypatch.setattr(answering_model_mod, "_post_json", lambda *a, **k: {"nope": True})
    model = OpenAICompatAnsweringModel(base_url="http://x", model="m")
    context = FlowContext(results=[])
    with pytest.raises(ProviderError):
        asyncio.run(model.answer("soru", context, "", []))


def test_answering_model_from_env_missing_provider():
    with pytest.raises(ProviderError):
        answering_model_from_env({})


def test_answering_model_from_env_anthropic_uses_default_base_url():
    """anthropic is natively supported now (see core/llm/anthropic.py) --
    it did not join `_COMPAT_PROVIDERS`, but `_common` accepts it and supplies
    its own default root (ANTHROPIC_DEFAULT_URL)."""
    model = answering_model_from_env(
        {"CHATBOT_LLM_PROVIDER": "anthropic", "CHATBOT_LLM_MODEL": "claude-sonnet-4-5"}
    )
    assert model.provider == "anthropic"
    assert model.base_url == "https://api.anthropic.com"


def test_answer_routes_to_call_anthropic_when_provider_is_anthropic(monkeypatch):
    captured = {}

    def fake_call_anthropic(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return _envelope("cevap")

    monkeypatch.setattr("medrag.core.llm.anthropic.call_anthropic", fake_call_anthropic)
    model = OpenAICompatAnsweringModel(
        base_url="https://api.anthropic.com", model="claude-sonnet-4-5",
        provider="anthropic", api_key="sk-ant-x",
    )
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))

    assert reply == "cevap"
    assert captured["kwargs"]["model"] == "claude-sonnet-4-5"
    assert captured["kwargs"]["api_key"] == "sk-ant-x"


def test_answering_model_from_env_ollama_default_base_url():
    model = answering_model_from_env({"CHATBOT_LLM_PROVIDER": "ollama", "CHATBOT_LLM_MODEL": "qwen2.5:32b"})
    assert model.base_url == "http://localhost:11434/v1"
    assert model.model == "qwen2.5:32b"


# --- native num_ctx path ---------------------------------------------------------


def test_answer_uses_native_api_chat_when_ollama_and_num_ctx_set(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": _envelope("native cevap")}}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=32768,
    )
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))

    assert reply == "native cevap"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["options"]["num_ctx"] == 32768
    assert captured["payload"]["stream"] is False
    assert "think" not in captured["payload"]  # no thinking signal -> field not added at all


def test_answer_native_path_maps_thinking_off_to_native_think_false(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["payload"] = payload
        return {"message": {"content": _envelope("ok")}}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=16384,
        extra_body={"reasoning_effort": "none"},
    )
    asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert captured["payload"]["think"] is False
    # extra_body is an /v1-specific field -- it must never leak into the native body.
    assert "reasoning_effort" not in captured["payload"]


def test_answer_without_num_ctx_still_uses_v1_path_even_for_ollama(monkeypatch):
    # Regression: without num_ctx the behavior goes EXACTLY down the old /v1 path.
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        return {"choices": [{"message": {"content": _envelope("eski yol")}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(base_url="http://localhost:11434/v1", model="m", provider="ollama")
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == "eski yol"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"


def test_answer_native_path_malformed_response_raises(monkeypatch):
    monkeypatch.setattr(answering_model_mod, "_post_json", lambda *a, **k: {"nope": True})
    model = OpenAICompatAnsweringModel(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=16384,
    )
    with pytest.raises(ProviderError):
        asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))


def test_answering_model_from_env_ollama_num_ctx_defaults_to_16384(monkeypatch):
    monkeypatch.delenv("CHATBOT_LLM_NUM_CTX", raising=False)
    model = answering_model_from_env(
        {"CHATBOT_LLM_PROVIDER": "ollama", "CHATBOT_LLM_MODEL": "m"}
    )
    assert model.num_ctx == 16384


def test_answering_model_from_env_non_ollama_num_ctx_stays_none():
    model = answering_model_from_env(
        {"CHATBOT_LLM_PROVIDER": "openai", "CHATBOT_LLM_MODEL": "m", "CHATBOT_LLM_BASE_URL": "http://x/v1"}
    )
    assert model.num_ctx is None


def test_answering_model_from_env_invalid_num_ctx_raises():
    with pytest.raises(ProviderError):
        answering_model_from_env(
            {"CHATBOT_LLM_PROVIDER": "ollama", "CHATBOT_LLM_MODEL": "m", "CHATBOT_LLM_NUM_CTX": "abc"}
        )


def test_answering_model_from_env_explicit_num_ctx_respected():
    model = answering_model_from_env(
        {"CHATBOT_LLM_PROVIDER": "ollama", "CHATBOT_LLM_MODEL": "m", "CHATBOT_LLM_NUM_CTX": "262144"}
    )
    assert model.num_ctx == 262144


# --- visual visibility filter -------------------------------------------------------


def test_visible_images_passthrough_when_no_visual_type():
    # Today's vectorize payload shape carries only `image_id` -- without a
    # `visual_type` an image is always visible regardless of exclude_types
    # (include-when-in-doubt policy).
    metadata = {"images": [{"image_id": "sha256:aaa"}]}
    assert _visible_images(metadata, ["chart"]) == [{"image_id": "sha256:aaa"}]


def test_visible_images_filters_excluded_visual_type():
    metadata = {"images": [
        {"image_id": "sha256:aaa", "visual_type": "chart"},
        {"image_id": "sha256:bbb", "visual_type": "product_photo"},
    ]}
    visible = _visible_images(metadata, ["chart"])
    assert visible == [{"image_id": "sha256:bbb", "visual_type": "product_photo"}]


def test_visible_images_empty_exclude_list_shows_everything():
    metadata = {"images": [{"image_id": "sha256:aaa", "visual_type": "chart"}]}
    assert _visible_images(metadata, []) == [{"image_id": "sha256:aaa", "visual_type": "chart"}]


def test_visible_images_no_images_key_returns_empty():
    assert _visible_images({}, ["chart"]) == []


def test_visible_images_does_not_mutate_source_dict():
    original = {"image_id": "sha256:aaa", "visual_type": "chart"}
    metadata = {"images": [original]}
    _visible_images(metadata, [])[0]["extra"] = "added"
    assert "extra" not in original


def test_source_suffix_adds_visible_image_count():
    metadata = {"doc_id": "PN1015", "images": [{"image_id": "sha256:aaa"}, {"image_id": "sha256:bbb"}]}
    assert "images=2" in _source_suffix(metadata, [])


def test_source_suffix_omits_count_when_all_images_excluded():
    metadata = {"doc_id": "PN1015", "images": [{"image_id": "sha256:aaa", "visual_type": "chart"}]}
    suffix = _source_suffix(metadata, ["chart"])
    assert "images=" not in suffix
    assert "doc=PN1015" in suffix


def test_render_context_never_leaks_raw_image_id_into_prose():
    # Rule: raw image_id/file path must NEVER enter the text sent to the
    # real model -- only a count is added.
    result = RetrievalResult(
        id="c1", score=0.9, text="metin",
        metadata={"images": [{"image_id": "sha256:deadbeef"}]},
    )
    rendered = _render_context(FlowContext(results=[result]))
    assert "sha256:deadbeef" not in rendered
    assert "images=1" in rendered


def test_extract_visible_images_aggregates_across_results_with_source_node_id():
    r1 = RetrievalResult(id="c1", score=0.9, text="a", metadata={"images": [{"image_id": "sha256:aaa"}]})
    r2 = RetrievalResult(
        id="c2", score=0.8, text="b",
        metadata={"images": [{"image_id": "sha256:bbb", "visual_type": "chart"}]},
    )
    images = extract_visible_images([r1, r2], exclude_types=["chart"])
    assert images == [{"image_id": "sha256:aaa", "source_node_id": "c1"}]


def test_extract_visible_images_no_results_returns_empty_list():
    assert extract_visible_images([]) == []


def test_extract_visible_images_default_exclude_types_shows_all():
    r1 = RetrievalResult(
        id="c1", score=0.9, text="a",
        metadata={"images": [{"image_id": "sha256:aaa", "visual_type": "chart"}]},
    )
    images = extract_visible_images([r1])
    assert images == [{"image_id": "sha256:aaa", "visual_type": "chart", "source_node_id": "c1"}]


def test_answer_uses_configured_image_exclude_types_in_source_suffix(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": _envelope("ok")}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(
        base_url="http://x/v1", model="m", image_exclude_types=["chart"]
    )
    result = RetrievalResult(
        id="c1", score=0.9, text="metin",
        metadata={"images": [{"image_id": "sha256:aaa", "visual_type": "chart"}]},
    )
    asyncio.run(model.answer("soru", FlowContext(results=[result]), "", []))
    system_content = captured["payload"]["messages"][0]["content"]
    assert "images=" not in system_content  # the only image is an excluded type -> 0 visible


# --- extract_visible_documents --------------------------------------------------------


def test_extract_visible_documents_returns_only_requested_rows():
    requested = RetrievalResult(
        id="doc:d1:PN1309", score=1.0,
        metadata={"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
                   "file_name": "a.pdf", "requested": True},
    )
    other = RetrievalResult(
        id="doc:d2:PN1309", score=0.5,
        metadata={"doc_id": "d2", "model": "PN1309", "doc_type": "BROCHURE",
                   "file_name": "b.pdf", "requested": False},
    )
    docs = extract_visible_documents([requested, other])
    assert docs == [{"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET", "file_name": "a.pdf"}]


def test_extract_visible_documents_never_includes_source_path_even_if_present():
    # Defense in depth: even if `source_path` accidentally sits in metadata,
    # extract_visible_documents never copies it (only the 4 allowed fields
    # are explicitly selected).
    result = RetrievalResult(
        id="doc:d1:PN1309", score=1.0,
        metadata={
            "doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
            "file_name": "a.pdf", "requested": True,
            "source_path": "C:\\gizli\\yol\\a.pdf",
        },
    )
    docs = extract_visible_documents([result])
    assert docs == [{"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET", "file_name": "a.pdf"}]
    assert "source_path" not in docs[0]


def test_extract_visible_documents_ignores_rows_without_doc_id_or_file_name():
    spec_row = RetrievalResult(
        id="r1", score=1.0, metadata={"model": "PN1309", "key": "operating_temperature"},
    )
    assert extract_visible_documents([spec_row]) == []


def test_extract_visible_documents_no_results_returns_empty_list():
    assert extract_visible_documents([]) == []


# --- result signal note ---------------------------------------------------------------


def test_build_messages_omits_result_signal_when_fields_unset():
    # The legacy/default shape of FlowContext (including flows that don't
    # fill result_shape today) -- no extra system part must be added (same
    # principle as the not-added-when-empty `followups`, see
    # test_build_messages_omits_followups_when_empty).
    messages = _build_messages("soru", FlowContext(results=[]), "", [])
    # FORMATTING is always added, so the baseline is 2 separators (see
    # test_build_messages_omits_followups_when_empty).
    assert messages[0]["content"].count("---") == 2


def test_build_messages_includes_result_shape_signal():
    context = FlowContext(results=[], result_shape="zero")
    messages = _build_messages("soru", context, "", [])
    assert "result_shape=zero" in messages[0]["content"]


def test_build_messages_includes_residual_signal():
    context = FlowContext(results=[], residual="ağırlık")
    messages = _build_messages("soru", context, "", [])
    assert "ağırlık" in messages[0]["content"]


def test_build_messages_includes_parse_note_signal():
    context = FlowContext(results=[], parse_note="near_miss_product_code")
    messages = _build_messages("soru", context, "", [])
    assert "parse_note=near_miss_product_code" in messages[0]["content"]


def test_build_messages_result_signal_never_dumps_scratch():
    # Rule: `scratch` NEVER reaches the model -- only
    # result_shape/residual/parse_note may.
    context = FlowContext(results=[], scratch={"secret_internal_state": "asla gorunmemeli"})
    messages = _build_messages("soru", context, "", [])
    assert "secret_internal_state" not in messages[0]["content"]
    assert "asla gorunmemeli" not in messages[0]["content"]


def test_extract_visible_documents_defaults_requested_false_when_missing():
    # Without the `requested` flag (e.g. an unexpected source) stay on the
    # SAFE side -- do NOT show it as downloadable.
    result = RetrievalResult(
        id="doc:d1:PN1309", score=1.0,
        metadata={"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET", "file_name": "a.pdf"},
    )
    assert extract_visible_documents([result]) == []


# --- extract_evidence_chunks --------------------------------------------------


def test_extract_evidence_chunks_labels_doc_source_with_section_and_page():
    result = RetrievalResult(
        id="c1", score=0.876543, text="kaynak metin",
        metadata={
            "doc_id": "d1", "heading_path": ["Bölüm 1", "Alt Bölüm"],
            "page_start": 3, "page_end": 4,
            "chunk_schema_version": 6, "chunker_version": "1.2.3",
        },
    )
    assert extract_evidence_chunks([result]) == [{
        "id": "c1", "score": 0.8765, "source_kind": "DOC", "doc_id": "d1",
        "section": "Bölüm 1 > Alt Bölüm", "page": "3-4", "table": None,
        "file_name": None, "attribute": None, "product_code": None,
        "chunk_schema_version": 6, "chunker_version": "1.2.3",
        "snippet": "kaynak metin",
    }]


def test_extract_evidence_chunks_labels_sql_source_without_doc_fields():
    result = RetrievalResult(id="row:1", score=1.0, text="fiyat: 100 TL", metadata={})
    assert extract_evidence_chunks([result]) == [{
        "id": "row:1", "score": 1.0, "source_kind": "SQL", "doc_id": None,
        "section": None, "page": None, "table": None,
        "file_name": None, "attribute": None, "product_code": None,
        "chunk_schema_version": None, "chunker_version": None,
        "snippet": "fiyat: 100 TL",
    }]


def test_extract_evidence_chunks_labels_sql_source_with_table_name():
    result = RetrievalResult(
        id="row:1", score=1.0, text="product_code=PN1309 | price=100",
        metadata={"sql_tables": "product"},
    )
    chunks = extract_evidence_chunks([result])
    assert chunks[0]["source_kind"] == "SQL"
    assert chunks[0]["table"] == "product"


def test_extract_evidence_chunks_sql_row_carries_file_and_attribute():
    """The evidence line should also state which file it came from and which
    attribute it evidences. `spec_value` already carries both fields -- when
    the SQL SELECTS them they must flow into the evidence row."""
    result = RetrievalResult(
        id="row:1", score=1.0,
        text="product_code=PN1162 | key=compliance_standards",
        metadata={
            "product_code": "PN1162", "key": "compliance_standards",
            "source_file_name": "ACME_PN5043_Datasheet.pdf",
        },
    )
    chunk = extract_evidence_chunks([result])[0]
    assert chunk["attribute"] == "compliance_standards"
    assert chunk["product_code"] == "PN1162"
    assert chunk["file_name"] == "ACME_PN5043_Datasheet.pdf"


def test_extract_evidence_chunks_doc_chunk_file_name_from_source_path_basename():
    """On the [DOC] side the file NAME is derived from the basename of
    `source_path` -- the FULL PATH never comes back. The file name itself is
    already the field `extract_visible_documents` shows to the user."""
    for yol in ("C:" + chr(92) + "depo" + chr(92) + "belgeler" + chr(92) + "ACME_PN5092_Manual.pdf",
                "/srv/depo/belgeler/ACME_PN5092_Manual.pdf"):
        result = RetrievalResult(
            id="c1", score=0.5, text="metin",
            metadata={"doc_id": "d1", "source_path": yol},
        )
        chunk = extract_evidence_chunks([result])[0]
        assert chunk["file_name"] == "ACME_PN5092_Manual.pdf"
        assert yol not in str(chunk)


def test_extract_evidence_chunks_sql_file_name_wins_over_source_path():
    result = RetrievalResult(
        id="row:1", score=1.0, text="x",
        metadata={"source_file_name": "A.pdf", "source_path": "/tmp/B.pdf"},
    )
    assert extract_evidence_chunks([result])[0]["file_name"] == "A.pdf"


def test_extract_evidence_chunks_file_and_attribute_absent_stay_none():
    """If the generated SQL did not select these columns, the fields silently
    stay `None` -- the UI hides them and nothing blows up anywhere."""
    result = RetrievalResult(id="row:1", score=1.0, text="model=PN1148", metadata={})
    chunk = extract_evidence_chunks([result])[0]
    assert chunk["file_name"] is None
    assert chunk["attribute"] is None
    assert chunk["product_code"] is None


def test_extract_evidence_chunks_truncates_long_text_to_snippet_len():
    result = RetrievalResult(id="c1", score=0.5, text="a" * 400, metadata={})
    chunks = extract_evidence_chunks([result], snippet_len=280)
    assert len(chunks[0]["snippet"]) == 281  # 280 + "…"
    assert chunks[0]["snippet"].endswith("…")


def test_extract_evidence_chunks_never_includes_source_path_even_if_present():
    result = RetrievalResult(
        id="c1", score=0.5, text="t", metadata={"source_path": "C:/secret/dosya.pdf"},
    )
    chunk = extract_evidence_chunks([result])[0]
    assert "source_path" not in chunk
    assert "C:/secret/dosya.pdf" not in str(chunk)


def test_extract_evidence_chunks_no_results_returns_empty_list():
    assert extract_evidence_chunks([]) == []


# --- reply contract (see reply_contract.py) ---------------------------------------
#
# Root cause of the live-test findings #1/#2/#5: the answering role was the
# only unverified LLM role in this repo, and its output flowed to the user,
# into `ConversationMemory`, AND (via memory) into the reconciler's prompt.
# The tests below reproduce the real observations.

_LEAKED_RAW = (
    "Wait, let me re-check thes. SQL[3] shows ARINC-429. Final plan: compare. "
    "<channel|>PN1267, PN1260 ile aynı arayüzleri kullanmaktadır."
)
_CLEAN_TURKISH = "PN1267, PN1260 ile aynı arayüzleri kullanmaktadır."
_DEGENERATE = "Ambargoyu export/import/" * 60


def test_build_messages_includes_reply_contract_note_before_language_reminder():
    messages = _build_messages("soru", FlowContext(results=[]), "", [])
    assert messages[-2] == {"role": "system", "content": REPLY_CONTRACT_NOTE}
    assert messages[-1]["content"] == _LANGUAGE_REMINDER


def test_build_messages_omits_contract_note_when_disabled():
    messages = _build_messages("soru", FlowContext(results=[]), "", [], contract=False)
    assert all(m["content"] != REPLY_CONTRACT_NOTE for m in messages)


def test_answer_returns_only_the_reply_field_never_the_json_envelope(monkeypatch):
    monkeypatch.setattr(
        answering_model_mod, "_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": _envelope("Merhaba", ["row1"])}}]},
    )
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m")
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == "Merhaba"
    assert "cited" not in reply and "{" not in reply


def test_answer_discards_reasoning_written_before_the_envelope(monkeypatch):
    """Direct reproduction of finding #1: even when the model writes its
    reasoning before the envelope, the user sees only the clean answer --
    and this happens in ONE call, no retry needed."""
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload)
        leaked_envelope = "Wait, let me re-check thes. Final plan: " + _envelope(_CLEAN_TURKISH)
        return {"choices": [{"message": {"content": leaked_envelope}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m")
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == _CLEAN_TURKISH
    assert "Wait" not in reply
    assert len(calls) == 1


def test_answer_retries_with_a_correction_when_envelope_missing(monkeypatch):
    """If the envelope never arrives (raw leak), a retry happens with a
    correction note; the broken output ITSELF is not put back into the
    prompt (to avoid priming the model on the same pattern)."""
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": _LEAKED_RAW}}]}
        return {"choices": [{"message": {"content": _envelope(_CLEAN_TURKISH)}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m")
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))

    assert reply == _CLEAN_TURKISH
    assert len(calls) == 2
    second_prompt = "\n".join(m["content"] for m in calls[1]["messages"])
    assert "valid JSON object" in second_prompt
    # The broken output was not sent back.
    assert "Wait, let me re-check" not in second_prompt


def test_answer_falls_back_to_canned_reply_when_every_attempt_is_degenerate(monkeypatch):
    """Findings #5 + #2 together: degenerate text is NOT delivered to the user
    and (since the orchestrator writes replies to memory) does not poison the
    rest of the session."""
    monkeypatch.setattr(
        answering_model_mod, "_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": _envelope(_DEGENERATE)}}]},
    )
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m")
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == _CONTRACT_FALLBACK_REPLY
    assert "Ambargoyu" not in reply


def test_answer_prefers_salvaged_text_over_canned_fallback(monkeypatch):
    """Even when the envelope never holds, if layer 2 (marker stripping)
    salvaged something, that is used instead of the canned reply -- we do
    not lose the turn unnecessarily."""
    monkeypatch.setattr(
        answering_model_mod, "_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": _LEAKED_RAW}}]},
    )
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m")
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == _CLEAN_TURKISH


def test_answer_respects_zero_contract_retries(monkeypatch):
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": _LEAKED_RAW}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m", contract_retries=0)
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert len(calls) == 1
    assert reply == _CLEAN_TURKISH  # text salvaged from a single attempt


def test_answer_with_contract_disabled_passes_raw_content_through(monkeypatch):
    """Emergency escape hatch: `contract=False` restores the pre-contract
    behavior EXACTLY (raw content, zero extra calls) -- the only knob left
    for an operator if a small model never holds the envelope."""
    calls = []

    def fake_post(url, payload, *, api_key, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": _LEAKED_RAW}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m", contract=False)
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == _LEAKED_RAW
    assert len(calls) == 1


def test_answer_validates_cited_ids_against_context_result_ids(monkeypatch, caplog):
    """If `cited` carries a fabricated id the answer is NOT dropped but a
    WARNING is logged (the mechanical rail for findings #3/#4)."""
    monkeypatch.setattr(
        answering_model_mod, "_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": _envelope("cevap", ["uydurma:99"])}}]},
    )
    result = RetrievalResult(id="row1", score=1.0, text="ad=PN1260", metadata={"ad": "PN1260"})
    model = OpenAICompatAnsweringModel(base_url="http://x/v1", model="m")
    with caplog.at_level("WARNING"):
        reply = asyncio.run(model.answer("soru", FlowContext(results=[result]), "", []))
    assert reply == "cevap"
    assert "unknown_cited" in caplog.text


def test_answer_native_path_also_enforces_the_contract(monkeypatch):
    # The contract is not /v1-specific -- it is enforced on the native `/api/chat` path too.
    monkeypatch.setattr(
        answering_model_mod, "_post_json",
        lambda *a, **k: {"message": {"content": "önsöz " + _envelope("native temiz")}},
    )
    model = OpenAICompatAnsweringModel(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=16384,
    )
    reply = asyncio.run(model.answer("soru", FlowContext(results=[]), "", []))
    assert reply == "native temiz"
