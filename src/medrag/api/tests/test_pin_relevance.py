"""Direct unit tests for the pure function `pin_relevance.looks_relevant_to_pin`.

Flow-specific tests (test_flows.py/test_doc_download_flow.py/
test_recommendation_flow.py) only verify that this function is wired correctly
into each flow's check_pin FALLBACK; the core logic lives here."""

from medrag.api.pin_relevance import looks_relevant_to_pin

_CANCEL_WORDS = {"iptal", "vazgeç"}


def test_empty_query_is_never_relevant_even_with_signal():
    assert looks_relevant_to_pin("   ", cancel_words=_CANCEL_WORDS, has_signal=True) is False


def test_cancel_word_is_never_relevant_even_with_signal():
    assert looks_relevant_to_pin("iptal", cancel_words=_CANCEL_WORDS, has_signal=True) is False


def test_cancel_word_case_insensitive():
    assert looks_relevant_to_pin("IPTAL", cancel_words=_CANCEL_WORDS, has_signal=True) is False


def test_no_signal_and_not_cancel_word_drops_pin():
    # Old behavior returned True here ("True until proven otherwise");
    # the new behavior inverts this.
    assert looks_relevant_to_pin("bu arada başka bir şey soracaktım", cancel_words=_CANCEL_WORDS, has_signal=False) is False


def test_signal_present_and_not_cancel_word_keeps_pin():
    assert looks_relevant_to_pin("de1000", cancel_words=_CANCEL_WORDS, has_signal=True) is True
