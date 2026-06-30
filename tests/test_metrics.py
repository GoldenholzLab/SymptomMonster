"""Tests for set-based extraction metrics in ``symptommonster.benchmark.metrics``."""

from __future__ import annotations

import pytest

from symptommonster.benchmark.metrics import (
    extraction_bias,
    macro_f1,
    precision_recall_f1,
    symptomatic_f1,
    weighted_mae,
)

# --- precision_recall_f1 --------------------------------------------------


def test_identical_nonempty_sets_score_perfectly():
    p, r, f1 = precision_recall_f1(["headache", "nausea"], ["nausea", "headache"])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_disjoint_sets_score_zero():
    p, r, f1 = precision_recall_f1(["headache"], ["dizziness"])
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_partial_overlap_is_between_zero_and_one():
    # Predicted {headache, nausea}; truth {headache, fatigue}: one hit each side.
    p, r, f1 = precision_recall_f1(["headache", "nausea"], ["headache", "fatigue"])
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert 0.0 < f1 < 1.0


def test_both_empty_is_a_perfect_match():
    # Agreeing that a patient has no symptoms is correct.
    assert precision_recall_f1([], []) == (1.0, 1.0, 1.0)


def test_duplicates_do_not_change_the_score():
    # Metrics are set-based, so repeats are collapsed.
    p, r, f1 = precision_recall_f1(["headache", "headache"], ["headache"])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


# --- macro_f1 -------------------------------------------------------------


def test_macro_f1_perfect_across_patients():
    preds = {"a": ["headache"], "b": ["nausea", "fatigue"]}
    truth = {"a": ["headache"], "b": ["fatigue", "nausea"]}
    assert macro_f1(preds, truth) == pytest.approx(1.0)


def test_macro_f1_averages_per_patient_not_per_symptom():
    # Patient "a" is perfect (F1=1); patient "b" is fully wrong (F1=0).
    # Macro average weights each patient equally regardless of symptom count.
    preds = {"a": ["headache"], "b": ["dizziness", "tremor", "rash"]}
    truth = {"a": ["headache"], "b": ["nausea", "fatigue", "cough"]}
    assert macro_f1(preds, truth) == pytest.approx(0.5)


def test_macro_f1_penalizes_missing_prediction():
    # Patient "b" has no prediction at all -> counts as an empty set -> F1=0.
    preds = {"a": ["headache"]}
    truth = {"a": ["headache"], "b": ["nausea"]}
    assert macro_f1(preds, truth) == pytest.approx(0.5)


# --- symptomatic_f1 -------------------------------------------------------


def test_symptomatic_f1_ignores_patients_with_empty_truth():
    # Patient "c" has empty truth and should not enter the average at all. The
    # remaining two patients are perfect, so the score is 1.0.
    preds = {"a": ["headache"], "b": ["nausea"], "c": ["spurious"]}
    truth = {"a": ["headache"], "b": ["nausea"], "c": []}
    assert symptomatic_f1(preds, truth) == pytest.approx(1.0)


def test_symptomatic_f1_differs_from_macro_when_empties_present():
    # On the empty-truth patient the model over-extracts (F1=0 under macro), so
    # macro is dragged down while the symptomatic score stays at 1.0.
    preds = {"a": ["headache"], "c": ["spurious"]}
    truth = {"a": ["headache"], "c": []}
    assert symptomatic_f1(preds, truth) == pytest.approx(1.0)
    assert macro_f1(preds, truth) < symptomatic_f1(preds, truth)


# --- weighted_mae ---------------------------------------------------------


def test_weighted_mae_zero_when_equal():
    assert weighted_mae([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == pytest.approx(0.0)


def test_weighted_mae_unweighted_is_plain_mean_abs_error():
    # |0.1-0.2| + |0.4-0.2| over two entries -> (0.1 + 0.2) / 2 = 0.15.
    assert weighted_mae([0.1, 0.4], [0.2, 0.2]) == pytest.approx(0.15)


def test_weighted_mae_respects_weights():
    # Heavily weighting the entry with zero error pulls the average down.
    predicted = [0.1, 0.9]
    reference = [0.1, 0.1]
    heavy_on_match = weighted_mae(predicted, reference, weights=[1000.0, 1.0])
    heavy_on_miss = weighted_mae(predicted, reference, weights=[1.0, 1000.0])
    assert heavy_on_match < heavy_on_miss


def test_weighted_mae_empty_vectors_are_zero():
    assert weighted_mae([], []) == pytest.approx(0.0)


# --- extraction_bias ------------------------------------------------------


def test_extraction_bias_positive_when_over_extracting():
    # Model attributes more symptoms than truth on average.
    preds = {"a": ["x", "y", "z"], "b": ["p", "q"]}
    truth = {"a": ["x"], "b": ["p"]}
    assert extraction_bias(preds, truth) > 0


def test_extraction_bias_negative_when_under_extracting():
    preds = {"a": ["x"], "b": []}
    truth = {"a": ["x", "y"], "b": ["p", "q"]}
    assert extraction_bias(preds, truth) < 0


def test_extraction_bias_zero_when_counts_match():
    # Same number of terms per patient, even though the terms differ.
    preds = {"a": ["x", "y"], "b": ["p"]}
    truth = {"a": ["m", "n"], "b": ["q"]}
    assert extraction_bias(preds, truth) == pytest.approx(0.0)
