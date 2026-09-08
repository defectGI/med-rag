"""Output contract of the answering role (`medrag.api.reply_contract`).

The tests reproduce REAL findings from a live-test round -- not invented
inputs, but observed failures:
  #1 raw-reasoning leak (the `<channel|>` marker)
  #5 degenerate repetition loop ("Ambargoyu export/import/...")
"""

import json

from medrag.api.reply_contract import (
    ReplyContract,
    extract_json_object,
    has_reasoning_marker,
    is_degenerate,
    parse_reply,
    strip_reasoning,
)

# The real shape of finding #1: English internal reasoning -> `<channel|>` ->
# the actual Turkish answer. The leak had been written VERBATIM into the `reply`
# field.
_LEAKED_RAW = (
    "Wait, let me re-check thes. The user asks about PN1267 interfaces. "
    "SQL[3] shows ARINC-429. Final plan: compare both. "
    "<channel|>PN1267, PN1260 ile aynı arayüzleri kullanmaktadır."
)
_CLEAN_TURKISH = "PN1267, PN1260 ile aynı arayüzleri kullanmaktadır."

# The real shape of finding #5.
_DEGENERATE = "Ambargoyu export/import/" * 60


# --- extract_json_object (layer 1: structural immunity) ----------------------


def test_extract_json_object_discards_reasoning_before_the_envelope():
    """CORE of the root-cause fix: reasoning text BEFORE the envelope is
    dropped without needing to guess a list of markers to cut on."""
    raw = 'Wait, let me reconsider. Final plan: answer briefly.\n{"reply": "Merhaba", "cited": []}'
    assert extract_json_object(raw) == {"reply": "Merhaba", "cited": []}


def test_extract_json_object_discards_prose_after_the_envelope():
    raw = '{"reply": "Merhaba"}\n\nHmm, actually I should have mentioned the price.'
    assert extract_json_object(raw) == {"reply": "Merhaba"}


def test_extract_json_object_strips_markdown_fence():
    raw = '```json\n{"reply": "Merhaba"}\n```'
    assert extract_json_object(raw) == {"reply": "Merhaba"}


def test_extract_json_object_returns_none_without_object():
    assert extract_json_object("hiç JSON yok") is None
    assert extract_json_object("") is None


def test_extract_json_object_returns_none_for_non_dict_json():
    assert extract_json_object("[1, 2, 3]") is None


# --- strip_reasoning (layer 2: last resort) -----------------------------------


def test_strip_reasoning_keeps_only_text_after_final_marker():
    cleaned, marker = strip_reasoning(_LEAKED_RAW)
    assert cleaned == _CLEAN_TURKISH
    assert marker == "<channel|>"


def test_strip_reasoning_uses_last_marker_when_several_present():
    raw = "<think>ilk</think>ara metin</think>asıl cevap"
    cleaned, _marker = strip_reasoning(raw)
    assert cleaned == "asıl cevap"


def test_strip_reasoning_handles_harmony_final_channel():
    raw = "analysis: the user wants X<|channel|>final<|message|>Asıl cevap burada."
    cleaned, marker = strip_reasoning(raw)
    assert cleaned == "Asıl cevap burada."
    assert marker == "<|channel|>final<|message|>"


def test_strip_reasoning_leaves_text_untouched_when_no_marker():
    """With no marker the text is LEFT UNTOUCHED -- guessing where the
    reasoning ends and cutting the user's real answer would be worse than the
    leak itself."""
    cleaned, marker = strip_reasoning("Tamamen normal bir Türkçe cevap.")
    assert cleaned == "Tamamen normal bir Türkçe cevap."
    assert marker is None


def test_strip_reasoning_removes_leftover_special_tokens():
    cleaned, _marker = strip_reasoning("</think><|start|>assistant<|message|>Cevap")
    assert cleaned == "assistantCevap"


def test_strip_reasoning_empty_input():
    assert strip_reasoning("") == ("", None)


# --- is_degenerate (layer 3) ---------------------------------------------------


def test_is_degenerate_catches_repeat_loop():
    assert is_degenerate(_DEGENERATE) is True


def test_is_degenerate_ignores_normal_long_answer():
    text = (
        "PN1260 ürünü MIL-STD-810G standardının çeşitli metotlarına göre test "
        "edilmiştir ve RoHS uyumludur. Çalışma sıcaklığı aralığı teknik "
        "dokümanda belirtilmiştir; ayrıntı için veri sayfasına bakabilirsiniz. "
        "Başka bir özelliği merak ediyorsanız sorabilirsiniz."
    )
    assert is_degenerate(text) is False


def test_is_degenerate_ignores_markdown_separator_rows():
    """False-positive protection: markdown separator rows consist only of
    dashes/spaces (no letters/digits) and must not count as degenerate."""
    table = "| Özellik | PN1260 | PN1267 |\n" + "| --- " * 40 + "|\n| Kanal | 2 | 4 |"
    assert is_degenerate(table) is False


def test_is_degenerate_ignores_table_with_repeated_cell_values_below_threshold():
    """A legitimate table may have a uniform column (e.g. 12 rows of
    "belirtilmemiş") -- the threshold (20 consecutive repeats) must NOT call
    this degenerate."""
    table = "| Ürün | Sıcaklık |\n" + "| PN1260 | belirtilmemiş |\n" * 12
    assert is_degenerate(table) is False


def test_is_degenerate_ignores_short_text():
    assert is_degenerate("ha ha ha ha ha") is False


def test_is_degenerate_empty():
    assert is_degenerate("") is False


# --- parse_reply (end to end) ---------------------------------------------------


def test_parse_reply_structured_envelope_is_ok():
    raw = json.dumps({"reply": "Merhaba", "cited": ["row1"]})
    parsed = parse_reply(raw, evidence_ids=["row1"])
    assert parsed.status == "structured"
    assert parsed.reply == "Merhaba"
    assert parsed.cited == ("row1",)
    assert parsed.unknown_cited == ()
    assert parsed.ok is True


def test_parse_reply_structured_envelope_after_reasoning_is_ok():
    """Fix for finding #1: even if the model writes its reasoning BEFORE the
    envelope, the user only sees `reply` and the turn is accepted WITHOUT a
    RETRY."""
    raw = 'Wait, let me re-check. Final plan: ' + json.dumps({"reply": _CLEAN_TURKISH})
    parsed = parse_reply(raw)
    assert parsed.ok is True
    assert parsed.reply == _CLEAN_TURKISH
    assert "Wait" not in parsed.reply


def test_parse_reply_strips_reasoning_that_leaked_inside_the_envelope():
    """Defense in depth: if the model puts reasoning INSIDE the envelope,
    layer 2 is also applied on the enveloped path."""
    raw = json.dumps({"reply": _LEAKED_RAW})
    parsed = parse_reply(raw)
    assert parsed.reply == _CLEAN_TURKISH
    assert parsed.status == "structured"


def test_parse_reply_flags_cited_ids_that_are_not_in_the_evidence():
    """The rail for findings #3/#4: fabricated citations become MECHANICALLY
    visible. It does NOT drop the answer (observational only for now), but it
    doesn't stay silent either."""
    raw = json.dumps({"reply": "cevap", "cited": ["row1", "uydurma:99"]})
    parsed = parse_reply(raw, evidence_ids=["row1"])
    assert parsed.unknown_cited == ("uydurma:99",)
    assert parsed.ok is True  # a citation issue does not drop the turn


def test_parse_reply_without_evidence_ids_does_not_flag_anything():
    # If no source list is given (e.g. an evidence-free out_of_scope turn)
    # no validation runs -- flagging everything as "unknown" would be noise.
    raw = json.dumps({"reply": "cevap", "cited": ["row1"]})
    assert parse_reply(raw).unknown_cited == ()


def test_parse_reply_salvages_via_marker_when_envelope_missing():
    parsed = parse_reply(_LEAKED_RAW)
    assert parsed.status == "sanitized"
    assert parsed.reply == _CLEAN_TURKISH
    assert parsed.marker == "<channel|>"
    # Salvage is a FALLBACK -- the turn is retried because the primary
    # guarantee didn't hold.
    assert parsed.ok is False


def test_parse_reply_raw_when_neither_envelope_nor_marker():
    parsed = parse_reply("Sadece düz metin, zarf da işaret de yok.")
    assert parsed.status == "raw"
    assert parsed.ok is False
    assert parsed.reply == "Sadece düz metin, zarf da işaret de yok."


def test_parse_reply_marks_degenerate_envelope_as_not_ok():
    """Finding #5: degenerate text must not reach the user even if the
    envelope is VALID."""
    raw = json.dumps({"reply": _DEGENERATE})
    parsed = parse_reply(raw)
    assert parsed.status == "structured"
    assert parsed.degenerate is True
    assert parsed.ok is False


def test_parse_reply_empty_envelope_reply_falls_through_to_salvage():
    parsed = parse_reply(json.dumps({"reply": "   "}))
    assert parsed.ok is False


def test_parse_reply_never_raises_on_empty_input():
    parsed = parse_reply("")
    assert isinstance(parsed, ReplyContract)
    assert parsed.reply == ""
    assert parsed.ok is False


def test_parse_reply_ignores_non_list_cited():
    raw = json.dumps({"reply": "cevap", "cited": "row1"})
    assert parse_reply(raw, evidence_ids=["row1"]).cited == ()


def test_log_note_summarizes_status_and_signals():
    parsed = parse_reply(json.dumps({"reply": "cevap", "cited": ["yok:1"]}), evidence_ids=["row1"])
    note = parsed.log_note()
    assert "status=structured" in note
    assert "unknown_cited" in note


# --- has_reasoning_marker (preflight leak DETECTION) ---------------------------


def test_has_reasoning_marker_finds_think_tag():
    assert has_reasoning_marker("<think>hmm</think>PONG") is not None


def test_has_reasoning_marker_finds_harmony_channel():
    assert has_reasoning_marker("analysis<|channel|>final<|message|>PONG") is not None


def test_has_reasoning_marker_finds_partial_channel_marker():
    assert has_reasoning_marker(_LEAKED_RAW) is not None


def test_has_reasoning_marker_none_on_clean_text():
    assert has_reasoning_marker("PONG") is None
    assert has_reasoning_marker("Tamamen normal bir Türkçe cevap.") is None


def test_has_reasoning_marker_empty():
    assert has_reasoning_marker("") is None
