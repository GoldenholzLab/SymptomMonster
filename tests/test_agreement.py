"""Tests for inter-annotator agreement in ``symptommonster.benchmark.agreement``."""

from __future__ import annotations

import numpy as np
import pytest

from symptommonster.benchmark.agreement import (
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
)

# --- Cohen's kappa --------------------------------------------------------


def test_cohen_kappa_identical_raters_is_one():
    labels = [0, 1, 1, 0, 1, 0, 0, 1]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_cohen_kappa_chance_agreement_is_about_zero():
    # Balanced marginals (each rater says 1 half the time) with agreement exactly
    # at the chance rate: observed = expected = 0.5, so kappa = 0.
    rater_a = [0, 1, 0, 1, 0, 1, 0, 1]
    rater_b = [0, 0, 1, 1, 0, 0, 1, 1]
    assert cohen_kappa(rater_a, rater_b) == pytest.approx(0.0, abs=0.2)


def test_cohen_kappa_systematic_disagreement_is_negative():
    # Raters are perfect opposites: worse than chance -> negative kappa.
    rater_a = [0, 1, 0, 1, 0, 1]
    rater_b = [1, 0, 1, 0, 1, 0]
    assert cohen_kappa(rater_a, rater_b) < 0


# --- Fleiss' kappa --------------------------------------------------------


def test_fleiss_kappa_perfect_agreement_is_one():
    # Five raters, two categories; on every item all five pick the same category.
    counts = np.array(
        [
            [5, 0],
            [0, 5],
            [5, 0],
            [0, 5],
        ]
    )
    assert fleiss_kappa(counts) == pytest.approx(1.0)


def test_fleiss_kappa_below_one_with_disagreement():
    # Same setup but with split votes on each item -> agreement well under 1.
    counts = np.array(
        [
            [3, 2],
            [2, 3],
            [3, 2],
            [2, 3],
        ]
    )
    assert fleiss_kappa(counts) < 1.0


# --- Krippendorff's alpha -------------------------------------------------


def test_krippendorff_alpha_perfect_agreement_is_one():
    # Two raters, four items, complete agreement.
    reliability = [
        [1, 0, 1, 0],
        [1, 0, 1, 0],
    ]
    assert krippendorff_alpha(reliability) == pytest.approx(1.0)


def test_krippendorff_alpha_tolerates_missing_values():
    # A missing cell (None) is skipped; the items both raters cover agree, so the
    # coefficient is still perfect.
    reliability = [
        [1, 0, 1, None],
        [1, 0, 1, 0],
    ]
    assert krippendorff_alpha(reliability) == pytest.approx(1.0)


def test_krippendorff_alpha_drops_below_one_with_disagreement():
    reliability = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
    ]
    assert krippendorff_alpha(reliability) < 1.0
