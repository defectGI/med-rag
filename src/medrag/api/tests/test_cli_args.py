import pytest

from medrag.api.__main__ import _parse_args


def test_parses_query_and_intent():
    args = _parse_args(["soru metni", "--intent", "medical_fact"])
    assert args.query == "soru metni"
    assert args.intent == "medical_fact"
    assert args.k is None


def test_rejects_unknown_intent():
    with pytest.raises(SystemExit):
        _parse_args(["soru", "--intent", "not_a_real_intent"])


def test_requires_intent():
    with pytest.raises(SystemExit):
        _parse_args(["soru"])
