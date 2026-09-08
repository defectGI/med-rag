"""Evaluation metrics — hand-rolled, no heavy deps.

Intent (classification): accuracy, per-class precision/recall/F1, macro- and
micro-averaged F1, plus a confusion matrix.
Retrieval (ranking): recall@k and MRR.

Kept dependency-free on purpose so the ``eval`` extra stays empty
(pyproject.toml). Pure Python; small golden sets don't need vectorisation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


@dataclass
class ClassScore:
    precision: float
    recall: float
    f1: float
    support: int  # number of golden items truly in this class


@dataclass
class IntentReport:
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class: dict[str, ClassScore] = field(default_factory=dict)
    # confusion[true][pred] = count
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    n: int = 0


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def intent_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> IntentReport:
    """Compute classification metrics from parallel label sequences."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch"
        )
    n = len(y_true)
    if n == 0:
        return IntentReport(accuracy=0.0, macro_f1=0.0, micro_f1=0.0, n=0)

    labels = sorted(set(y_true) | set(y_pred))
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }

    correct = 0
    for t, p in zip(y_true, y_pred):
        support[t] += 1
        confusion[t][p] += 1
        if t == p:
            correct += 1
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class: dict[str, ClassScore] = {}
    for label in labels:
        prec_den = tp[label] + fp[label]
        rec_den = tp[label] + fn[label]
        precision = tp[label] / prec_den if prec_den else 0.0
        recall = tp[label] / rec_den if rec_den else 0.0
        per_class[label] = ClassScore(
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            support=support[label],
        )

    accuracy = correct / n
    macro_f1 = sum(c.f1 for c in per_class.values()) / len(labels)

    # Micro-averaged: pool tp/fp/fn across classes. For single-label
    # classification this equals accuracy, but we compute it explicitly so the
    # report stays correct if the setup ever changes.
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = _f1(micro_p, micro_r)

    return IntentReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        per_class=per_class,
        confusion=confusion,
        n=n,
    )


# ---------------------------------------------------------------------------
# Retrieval (ranking)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalReport:
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    n: int = 0


def recall_at_k(relevant: Sequence[str], ranked: Sequence[str], k: int) -> float:
    """Fraction of relevant ids present in the top-k of ``ranked``.

    Returns 0.0 when there are no relevant ids for the query.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    topk = set(ranked[:k])
    return len(rel & topk) / len(rel)


def reciprocal_rank(relevant: Sequence[str], ranked: Sequence[str]) -> float:
    """1/rank of the first relevant id in ``ranked`` (1-indexed), else 0.0."""
    rel = set(relevant)
    for idx, item in enumerate(ranked, start=1):
        if item in rel:
            return 1.0 / idx
    return 0.0


def retrieval_metrics(
    cases: Sequence[tuple[Sequence[str], Sequence[str]]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalReport:
    """Aggregate recall@k and MRR over ``(relevant_ids, ranked_ids)`` cases."""
    n = len(cases)
    if n == 0:
        return RetrievalReport(recall_at_k={k: 0.0 for k in ks}, mrr=0.0, n=0)

    recall_sums = {k: 0.0 for k in ks}
    rr_sum = 0.0
    for relevant, ranked in cases:
        for k in ks:
            recall_sums[k] += recall_at_k(relevant, ranked, k)
        rr_sum += reciprocal_rank(relevant, ranked)

    return RetrievalReport(
        recall_at_k={k: recall_sums[k] / n for k in ks},
        mrr=rr_sum / n,
        n=n,
    )


__all__ = [
    "ClassScore",
    "IntentReport",
    "RetrievalReport",
    "intent_metrics",
    "recall_at_k",
    "reciprocal_rank",
    "retrieval_metrics",
]
