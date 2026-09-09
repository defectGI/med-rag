"""LLMIntentClassifier unit tests with a fake chat model (no network/GPU)."""

from __future__ import annotations

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from medrag.api.retrieval.core import IntentLabel, IntentResult
from medrag.api.retrieval.modules.intent_classification import (
    IntentClassifier,
    LLMIntentClassifier,
    classify_sync,
)


def _clf_returning(text: str) -> LLMIntentClassifier:
    fake = GenericFakeChatModel(messages=iter([AIMessage(content=text)]))
    return LLMIntentClassifier(fake)


def test_classifier_satisfies_protocol():
    assert isinstance(_clf_returning("medical_fact"), IntentClassifier)


def test_classify_each_label():
    for label in IntentLabel:
        clf = _clf_returning(label.value)
        res = asyncio.run(clf.classify("some query"))
        assert isinstance(res, IntentResult)
        assert res.label is label
        assert res.confidence is None  # fake model returns no logprobs


def test_classify_defensive_on_messy_output():
    clf = _clf_returning("  Intent = Medical_Fact\n")
    res = asyncio.run(clf.classify("arveles dozu nedir"))
    assert res.label is IntentLabel.MEDICAL_FACT
    assert res.raw_text is None  # genuine match, not a fallback


def test_classify_unparseable_output_carries_raw_text_for_tracing():
    # When the model returns something unrelated, the label falls back to
    # out_of_scope -- but the text that caused the drop must not be lost, so it
    # is carried on raw_text for tracing.
    clf = _clf_returning("bugun hava cok guzel degil mi")
    res = asyncio.run(clf.classify("arveles dozu nedir"))
    assert res.label is IntentLabel.OUT_OF_SCOPE
    assert res.raw_text == "bugun hava cok guzel degil mi"


def test_classify_genuine_out_of_scope_has_no_raw_text():
    # If the model genuinely said out_of_scope, that is not a fallback.
    clf = _clf_returning("out_of_scope")
    res = asyncio.run(clf.classify("hava durumu nasil"))
    assert res.label is IntentLabel.OUT_OF_SCOPE
    assert res.raw_text is None


def test_classify_sync_helper():
    clf = _clf_returning("comparison")
    res = classify_sync(clf, "kac urun var")
    assert isinstance(res, IntentResult)
    assert res.label is IntentLabel.COMPARISON


def test_confidence_wired_from_logprobs():
    # Exercise the classifier's confidence path with a message carrying logprobs.
    clf = _clf_returning("medical_fact")
    msg = AIMessage(
        content="medical_fact",
        response_metadata={"logprobs": {"content": [{"token": "medical_fact",
                                                      "logprob": -0.05}]}},
    )
    res = clf._to_result(msg)
    assert res.label is IntentLabel.MEDICAL_FACT
    assert res.confidence is not None
    assert 0.0 < res.confidence <= 1.0
