"""Metric tests with hand-computed expected values."""

from __future__ import annotations

import math

from medrag.api.retrieval.eval import metrics


def test_intent_perfect_score():
    y = ["a", "b", "c"]
    rep = metrics.intent_metrics(y, y)
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == 1.0
    assert rep.micro_f1 == 1.0
    assert rep.per_class["a"].support == 1


def test_intent_known_confusion():
    # 4 items, one 'a' misclassified as 'b'.
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    rep = metrics.intent_metrics(y_true, y_pred)
    assert rep.accuracy == 0.75
    # class a: tp=1 fp=0 fn=1 -> P=1.0 R=0.5 F1=2/3
    assert rep.per_class["a"].precision == 1.0
    assert rep.per_class["a"].recall == 0.5
    assert math.isclose(rep.per_class["a"].f1, 2 / 3)
    # class b: tp=2 fp=1 fn=0 -> P=2/3 R=1.0
    assert math.isclose(rep.per_class["b"].precision, 2 / 3)
    assert rep.per_class["b"].recall == 1.0
    assert rep.confusion["a"]["b"] == 1
    # single-label: micro-F1 == accuracy
    assert math.isclose(rep.micro_f1, rep.accuracy)


def test_recall_at_k():
    relevant = ["x", "y"]
    ranked = ["z", "x", "w", "y"]
    assert metrics.recall_at_k(relevant, ranked, 1) == 0.0
    assert metrics.recall_at_k(relevant, ranked, 2) == 0.5
    assert metrics.recall_at_k(relevant, ranked, 4) == 1.0
    assert metrics.recall_at_k([], ranked, 4) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["y"], ["a", "b", "y"]) == 1 / 3
    assert metrics.reciprocal_rank(["a"], ["a", "b"]) == 1.0
    assert metrics.reciprocal_rank(["q"], ["a", "b"]) == 0.0


def test_retrieval_aggregate():
    cases = [
        (["x"], ["x", "a", "b"]),  # rr=1, recall@1=1
        (["y"], ["a", "y", "b"]),  # rr=0.5, recall@1=0
    ]
    rep = metrics.retrieval_metrics(cases, ks=(1, 3))
    assert rep.mrr == 0.75
    assert rep.recall_at_k[1] == 0.5
    assert rep.recall_at_k[3] == 1.0
    assert rep.n == 2
