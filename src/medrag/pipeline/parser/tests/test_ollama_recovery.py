"""ollama_recovery.py: the degenerate-output pattern check and the local-only
restart guard (see openai_compat.py's garbage-retry loop, which is the one
caller of both)."""

from __future__ import annotations

from medrag.pipeline.parser.llm import ollama_recovery


def test_repeated_char_is_garbage():
    assert ollama_recovery.is_garbage_output("?" * 40)


def test_short_repeated_char_is_not_garbage():
    # Below _MIN_LEN -- a short legitimate answer ("----", "N/A") shouldn't trip it.
    assert not ollama_recovery.is_garbage_output("---")


def test_normal_prose_is_not_garbage():
    text = ("The table lists operating temperature ranges for each product "
            "variant across three columns.")
    assert not ollama_recovery.is_garbage_output(text)


def test_whitespace_padding_is_ignored():
    # Newline/space-heavy but not character-degenerate.
    assert not ollama_recovery.is_garbage_output("word\n" * 20)


def test_mostly_one_char_with_minor_noise_is_still_garbage():
    assert ollama_recovery.is_garbage_output("?" * 36 + "abcd")


def test_restart_skips_remote_host():
    # A remote/non-local base_url must never trigger a kill -- verified by
    # the fact this returns False without needing to mock subprocess at all.
    assert ollama_recovery.restart_local_ollama("https://gpu-box.example.com/v1") is False


def test_restart_local_host_missing_ollama_binary(monkeypatch):
    monkeypatch.setattr(ollama_recovery, "_kill_ollama_process", lambda: None)
    monkeypatch.setattr(ollama_recovery, "_start_ollama_process", lambda: False)
    assert ollama_recovery.restart_local_ollama("http://localhost:11434/v1") is False


def test_restart_local_host_comes_back_up(monkeypatch):
    calls = []
    monkeypatch.setattr(ollama_recovery, "_kill_ollama_process", lambda: calls.append("kill"))
    monkeypatch.setattr(ollama_recovery, "_start_ollama_process", lambda: calls.append("start") or True)
    monkeypatch.setattr(ollama_recovery, "_server_version", lambda root: "0.1.0")
    monkeypatch.setattr(ollama_recovery.time, "sleep", lambda _: None)

    assert ollama_recovery.restart_local_ollama("http://localhost:11434/v1") is True
    assert calls == ["kill", "start"]
