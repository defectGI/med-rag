"""VLM failure forensics (_read_page classification) and known-answer probes.

The July 2026 full run lost 509 pages to an endpoint that answered HTTP 200
with unusable content; the output metadata couldn't say why. These tests pin
the two defenses: every read failure is classified and counted per kind, and
the health probes tell a usable endpoint from a merely reachable one.
"""

from medrag.pipeline.parser.llm.base import LLMError
from medrag.pipeline.parser.llm.health import probe_llm, probe_vlm
from medrag.pipeline.parser.parsers.pdf_parser import PdfParser


class _FakeVLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, **kwargs):
        return self.complete_vision(**kwargs)

    def complete_vision(self, **kwargs):
        self.calls += 1
        reply = self._replies.pop(0) if self._replies else self._last
        self._last = reply
        if isinstance(reply, Exception):
            raise reply
        return reply


def _parser(vlm):
    return PdfParser(vlm=vlm, vlm2=None, detector=None, table_struct=None)


VALID = '[{"type": "paragraph", "text": "hello"}]'


def test_read_page_ok_counts():
    p = _parser(_FakeVLM([VALID]))
    specs = p._read_page(p._vlm, b"png", "read it")
    assert specs and specs[0]["text"] == "hello"
    assert p._read_stats == {"ok": 1}
    assert p._last_read_failure is None


def test_read_page_transport_classified():
    p = _parser(_FakeVLM([LLMError("cannot reach host")]))
    assert p._read_page(p._vlm, b"png", "read it") is None
    assert p._read_stats == {"transport": 1}
    kind, detail = p._last_read_failure
    assert kind == "transport" and "cannot reach host" in detail


def test_read_page_empty_reply_classified():
    p = _parser(_FakeVLM(["", ""]))
    assert p._read_page(p._vlm, b"png", "read it") is None
    assert p._read_stats == {"empty-reply": 1}
    assert p._last_read_failure[0] == "empty-reply"


def test_read_page_bad_json_classified():
    p = _parser(_FakeVLM(["I think this page shows...", "still not json"]))
    assert p._read_page(p._vlm, b"png", "read it") is None
    assert p._read_stats == {"bad-json": 1}
    kind, detail = p._last_read_failure
    assert kind == "bad-json" and "still not json" in detail


def test_content_failures_do_not_trip_transport_kill_switch():
    # 3 pages of empty replies: the VLM must stay enabled (server is up,
    # disabling it would only mask a model/config problem).
    p = _parser(_FakeVLM([""] * 6))
    for _ in range(3):
        assert p._read_page(p._vlm, b"png", "read it") is None
    assert p._vlm is not None
    assert p._vlm_disabled is None
    assert p._read_stats == {"empty-reply": 3}


def test_transport_kill_switch_records_reason():
    p = _parser(_FakeVLM([LLMError("down"), LLMError("down")]))
    p._read_page(p._vlm, b"png", "read it")
    p._read_page(p._vlm, b"png", "read it")
    assert p._vlm is None
    assert "transport errors" in p._vlm_disabled


def test_probe_vlm_pass_and_fail():
    ok, detail = probe_vlm(_FakeVLM(["The image says: DOCRAG 4137"]))
    assert ok and detail == ""
    ok, detail = probe_vlm(_FakeVLM([""]))
    assert not ok and "unusable reply" in detail
    ok, detail = probe_vlm(_FakeVLM([LLMError("no route")]))
    assert not ok and "transport" in detail


def test_probe_llm_pass_and_fail():
    ok, _ = probe_llm(_FakeVLM(["PONG"]))
    assert ok
    ok, detail = probe_llm(_FakeVLM(["<think>hmm</think>"]))
    assert not ok and "unusable reply" in detail
