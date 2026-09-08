"""Defensive parser + logprobs->confidence unit tests (no network)."""

from __future__ import annotations

import math

import pytest

from medrag.api.retrieval.core import IntentLabel
from medrag.api.retrieval.modules.intent_classification.parsing import (
    confidence_from_logprobs,
    parse_label,
    parse_label_verbose,
)


@pytest.mark.parametrize("label", list(IntentLabel))
def test_parse_exact_label(label):
    assert parse_label(label.value) is label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  product_fact ", IntentLabel.PRODUCT_FACT),
        ("Product_Fact", IntentLabel.PRODUCT_FACT),
        ('"out_of_scope"', IntentLabel.OUT_OF_SCOPE),
        ("intent: doc_question", IntentLabel.DOC_QUESTION),
        ("The label is comparison.", IntentLabel.COMPARISON),
    ],
)
def test_parse_normalises_and_extracts(raw, expected):
    assert parse_label(raw) is expected


@pytest.mark.parametrize("raw", ["", "   ", "gibberish", "no valid token here"])
def test_parse_falls_back_to_out_of_scope(raw):
    assert parse_label(raw) is IntentLabel.OUT_OF_SCOPE


# --- parse_label_verbose: distinguishes genuine label from silent fallback ---


def test_verbose_exact_match_is_not_a_fallback():
    label, was_fallback = parse_label_verbose("out_of_scope")
    assert label is IntentLabel.OUT_OF_SCOPE
    assert was_fallback is False


@pytest.mark.parametrize("raw", ["", "   ", "gibberish", "no valid token here"])
def test_verbose_unparseable_is_a_fallback(raw):
    label, was_fallback = parse_label_verbose(raw)
    assert label is IntentLabel.OUT_OF_SCOPE
    assert was_fallback is True


def test_verbose_substring_match_is_not_a_fallback():
    label, was_fallback = parse_label_verbose("intent: doc_question")
    assert label is IntentLabel.DOC_QUESTION
    assert was_fallback is False


def test_confidence_none_when_absent():
    assert confidence_from_logprobs(None) is None
    assert confidence_from_logprobs({}) is None
    assert confidence_from_logprobs({"logprobs": None}) is None
    assert confidence_from_logprobs({"logprobs": {"content": []}}) is None


def test_confidence_from_valid_logprobs():
    meta = {"logprobs": {"content": [{"token": "product", "logprob": -0.1},
                                     {"token": "_fact", "logprob": -0.2}]}}
    conf = confidence_from_logprobs(meta)
    assert conf is not None
    assert math.isclose(conf, math.exp(-0.3), rel_tol=1e-9)
    assert 0.0 < conf <= 1.0


def test_confidence_malformed_is_none_not_error():
    assert confidence_from_logprobs({"logprobs": {"content": [{"token": "x"}]}}) is None
    assert confidence_from_logprobs({"logprobs": {"content": "nope"}}) is None
