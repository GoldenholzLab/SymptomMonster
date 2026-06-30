"""Set-based extraction metrics for benchmarking a model against ground truth.

Every patient is scored on the *set* of symptom terms attributed to them: a term
in both predicted and truth is a hit, one only in predicted is over-extraction,
one only in truth is a miss. Per-patient F1 is then averaged (macro) so that a
patient with two symptoms counts the same as one with ten. Callers are expected
to canonicalize terms (see ``equivalence``) before passing them in, so equality
here is plain string equality.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def precision_recall_f1(
    predicted: Iterable[str], truth: Iterable[str]
) -> tuple[float, float, float]:
    """Set-based precision, recall, and F1 for one patient.

    Identical non-empty sets give ``(1, 1, 1)``; disjoint sets give ``(0, 0, 0)``.
    Two empty sets are a perfect match ``(1, 1, 1)``, since agreeing that a patient
    has no symptoms is correct. An empty prediction against a non-empty truth (and
    the reverse) scores ``(0, 0, 0)``, since there is no true positive to divide by.
    """
    pred, gold = set(predicted), set(truth)

    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0

    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_f1(
    predictions: Mapping[str, Iterable[str]], truths: Mapping[str, Iterable[str]]
) -> float:
    """Mean per-patient F1 over the patients present in ``truths``.

    A patient missing from ``predictions`` is treated as an empty prediction, so
    a model that skipped them is penalized rather than silently dropped.
    """
    if not truths:
        return 0.0
    total = 0.0
    for pid, gold in truths.items():
        _, _, f1 = precision_recall_f1(predictions.get(pid, ()), gold)
        total += f1
    return total / len(truths)


def symptomatic_f1(
    predictions: Mapping[str, Iterable[str]], truths: Mapping[str, Iterable[str]]
) -> float:
    """Mean per-patient F1 restricted to patients whose truth set is non-empty.

    This isolates performance on the patients who actually have symptoms, where
    the prevalence of empty-vs-empty matches cannot inflate the score.
    """
    symptomatic = {pid: gold for pid, gold in truths.items() if set(gold)}
    if not symptomatic:
        return 0.0
    total = 0.0
    for pid, gold in symptomatic.items():
        _, _, f1 = precision_recall_f1(predictions.get(pid, ()), gold)
        total += f1
    return total / len(symptomatic)


def weighted_mae(
    predicted_rates: Sequence[float],
    reference_rates: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Weighted mean absolute error between two aligned rate vectors.

    The vectors are per-symptom prevalences in the same order; the result is the
    weighted average of ``|predicted - reference|``. Equal vectors give ``0.0``
    and the error grows with divergence. With ``weights=None`` every symptom is
    weighted equally (plain mean absolute error).
    """
    if len(predicted_rates) != len(reference_rates):
        raise ValueError("predicted_rates and reference_rates must have equal length")
    n = len(predicted_rates)
    if n == 0:
        return 0.0

    if weights is None:
        return sum(abs(p - r) for p, r in zip(predicted_rates, reference_rates, strict=True)) / n

    if len(weights) != n:
        raise ValueError("weights must match the length of the rate vectors")
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    weighted = sum(
        w * abs(p - r)
        for p, r, w in zip(predicted_rates, reference_rates, weights, strict=True)
    )
    return weighted / total_weight


def extraction_bias(
    predictions: Mapping[str, Iterable[str]], truths: Mapping[str, Iterable[str]]
) -> float:
    """Mean per-patient ``len(predicted) - len(truth)`` over patients in ``truths``.

    A positive value means the model attributes more symptoms than the ground
    truth (over-extraction); a negative value means it under-extracts. Zero means
    it matches the count on average, regardless of which specific terms it picked.
    """
    if not truths:
        return 0.0
    total = 0
    for pid, gold in truths.items():
        total += len(set(predictions.get(pid, ()))) - len(set(gold))
    return total / len(truths)
