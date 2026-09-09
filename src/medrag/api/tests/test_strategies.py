"""Model-facing / developer-notes split of the strategy files.

Locks in two things:
1. The model-facing section returned by `load_strategy` for each strategies/*.md
   file contains NO tone/style language (the instructions now live in ONE
   place, answering_model.py::_BASE_PERSONA) -- the duplicated
   "conversational tone"/"technical language" phrases were moved from the
   strategies into the persona and must not leak back.
2. A strategy file WITHOUT the marker (`_DEV_NOTES_MARKER`) -- backward
   compatibility -- goes to the model EXACTLY as-is (old behavior).
"""
from __future__ import annotations

import re

from medrag.api.orchestrator import STRATEGIES_DIR, load_strategy

_ALL_STRATEGY_KEYS = [
    "medical_fact",
    "clinical_decision",
    "comparison",
    "interaction",
    "library",
    "out_of_scope",
]

# Tone/style expressions moved to the persona (answering_model.py::_BASE_PERSONA),
# where they must live in a single place -- no copy of these may reappear in
# individual strategy files.
_TONE_PATTERNS = [
    re.compile(r"conversational tone", re.IGNORECASE),
    re.compile(r"technical language", re.IGNORECASE),
]


def test_all_six_strategy_files_exist():
    for key in _ALL_STRATEGY_KEYS:
        assert (STRATEGIES_DIR / f"{key}.md").exists()


def test_model_facing_section_has_no_tone_language():
    """Tone instructions now live only in _BASE_PERSONA -- a copy must not
    appear in the MODEL-FACING section of any individual strategy file."""
    for key in _ALL_STRATEGY_KEYS:
        model_text = load_strategy(key)
        for pattern in _TONE_PATTERNS:
            assert not pattern.search(model_text), (
                f"tone language found in {key}.md model-facing section: "
                f"{pattern.pattern!r} -- this is _BASE_PERSONA's job now"
            )


def test_model_facing_section_excludes_dev_notes_marker_and_below():
    """The developer-notes marker and everything after it (decision refs, file
    pointers, ROADMAP citations) must never reach the model."""
    dev_note_smells = ["decision:", "ROADMAP", "chatbot/flows/", "see [README]"]
    for key in _ALL_STRATEGY_KEYS:
        model_text = load_strategy(key)
        assert "CHATBOT:DEV-NOTES" not in model_text
        for smell in dev_note_smells:
            assert smell not in model_text, (
                f"developer-notes smell in {key}.md model-facing section: {smell!r}"
            )


def test_load_strategy_without_marker_returns_whole_file_unchanged(tmp_path, monkeypatch):
    """Backward compatibility: a file without the marker -- as before --
    goes ENTIRELY to the model, no trimming at all."""
    whole_file = (
        "# Strategy: legacy_no_marker\n\n"
        "**Intent:** `legacy_no_marker`\n"
        "This entire file, including this developer-looking line about "
        "chatbot/flows/legacy.py and ROADMAP 99z, has no marker so it must "
        "all reach the model exactly as-is.\n"
    )
    (tmp_path / "legacy_no_marker.md").write_text(whole_file, encoding="utf-8")
    monkeypatch.setattr("medrag.api.orchestrator.STRATEGIES_DIR", tmp_path)

    result = load_strategy("legacy_no_marker")

    assert result == whole_file


def test_load_strategy_with_marker_returns_only_text_before_it(tmp_path, monkeypatch):
    model_part = "# Strategy: with_marker\n\nModel-facing guidance only.\n"
    dev_part = "## Geliştirici Notları\n\ndecision: 2026-07-24, see chatbot/flows/x.py\n"
    full_text = model_part + "\n<!-- CHATBOT:DEV-NOTES (modele gitmez) -->\n\n" + dev_part
    (tmp_path / "with_marker.md").write_text(full_text, encoding="utf-8")
    monkeypatch.setattr("medrag.api.orchestrator.STRATEGIES_DIR", tmp_path)

    result = load_strategy("with_marker")

    assert result == model_part.rstrip("\n")
    assert "DEV-NOTES" not in result
    assert "decision:" not in result


def test_out_of_scope_model_facing_section_still_effectively_a_capability_list():
    """out_of_scope.md keeps a concrete "what I can help with" list in its
    model-facing section, and its developer notes (Flow line) must not leak."""
    text = load_strategy("out_of_scope")
    assert "out_of_scope" in text
    assert "Geliştirici Notları" not in text
    assert "no_retrieval" not in text  # the Flow line from dev-notes must not leak
