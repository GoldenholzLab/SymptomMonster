"""Tests for the pure statistical primitives in ``symptommonster.stats.core``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from symptommonster.stats.core import (
    benjamini_hochberg,
    clopper_pearson,
    covariate_adjusted_residual,
    mean_difference_ci,
    paired_ttest,
)

# --- Benjamini-Hochberg ---------------------------------------------------


def test_bh_returns_results_in_input_order():
    # Deliberately unsorted input; outputs must line up with these positions.
    pvalues = [0.5, 0.001, 0.2, 0.04]
    qvalues, rejected = benjamini_hochberg(pvalues, q=0.10)

    assert len(qvalues) == len(pvalues)
    assert len(rejected) == len(pvalues)
    # The tiny p-value is the one that should clearly survive correction.
    assert rejected[1] is True
    # The largest p-value should not be rejected at q=0.10.
    assert rejected[0] is False


def test_bh_qvalues_are_at_least_the_raw_pvalues():
    pvalues = [0.01, 0.02, 0.03, 0.5, 0.9]
    qvalues, _ = benjamini_hochberg(pvalues, q=0.10)
    for p, q in zip(pvalues, qvalues, strict=False):
        assert q >= p - 1e-12
        assert 0.0 <= q <= 1.0


def test_bh_qvalues_are_monotone_in_sorted_p_order():
    pvalues = [0.2, 0.01, 0.04, 0.005, 0.5]
    qvalues, _ = benjamini_hochberg(pvalues)
    # Sort (p, q) by p and confirm q never decreases as p increases.
    paired = sorted(zip(pvalues, qvalues, strict=False), key=lambda pq: pq[0])
    sorted_q = [q for _, q in paired]
    for earlier, later in zip(sorted_q, sorted_q[1:], strict=False):
        assert later >= earlier - 1e-12


def test_bh_all_significant_when_p_all_tiny():
    pvalues = [1e-6, 1e-6, 1e-6]
    _, rejected = benjamini_hochberg(pvalues, q=0.05)
    assert all(rejected)


def test_bh_none_significant_when_p_all_large():
    pvalues = [0.6, 0.7, 0.8, 0.95]
    _, rejected = benjamini_hochberg(pvalues, q=0.10)
    assert not any(rejected)


def test_bh_empty_input():
    qvalues, rejected = benjamini_hochberg([])
    assert qvalues == []
    assert rejected == []


# --- paired t-test --------------------------------------------------------


def test_paired_ttest_identical_inputs_is_null():
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    t_stat, p_value = paired_ttest(values, values)
    assert t_stat == 0.0
    assert p_value == 1.0


def test_paired_ttest_detects_consistent_positive_shift():
    # Signal exceeds noise for every pair: a clear, significant positive effect.
    signal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    noise = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    t_stat, p_value = paired_ttest(signal, noise)
    assert t_stat > 0
    assert p_value < 0.05


def test_paired_ttest_sign_follows_direction():
    # When noise consistently exceeds signal, the statistic is negative.
    signal = [0.0, 0.0, 0.0, 0.0]
    noise = [1.0, 1.0, 1.0, 1.0]
    t_stat, p_value = paired_ttest(signal, noise)
    assert t_stat < 0
    assert p_value < 0.05


def test_paired_ttest_two_sided_pvalue_in_unit_interval():
    signal = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    noise = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    _, p_value = paired_ttest(signal, noise)
    assert 0.0 <= p_value <= 1.0


# --- Clopper-Pearson ------------------------------------------------------


def test_clopper_pearson_zero_successes_has_zero_lower_bound():
    low, high = clopper_pearson(0, 20)
    assert low == 0.0
    assert high > 0.0


def test_clopper_pearson_all_successes_has_unit_upper_bound():
    low, high = clopper_pearson(20, 20)
    assert high == 1.0
    assert low < 1.0


def test_clopper_pearson_interval_contains_point_estimate():
    k, n = 7, 20
    low, high = clopper_pearson(k, n)
    assert low <= k / n <= high
    assert 0.0 <= low <= high <= 1.0


def test_clopper_pearson_more_data_tightens_interval():
    low_small, high_small = clopper_pearson(5, 10)
    low_big, high_big = clopper_pearson(50, 100)
    # Same proportion, ten times the data: the interval should be narrower.
    assert (high_big - low_big) < (high_small - low_small)


def test_clopper_pearson_empty_sample_is_whole_interval():
    low, high = clopper_pearson(0, 0)
    assert low == 0.0
    assert high == 1.0


# --- mean difference CI ---------------------------------------------------


def test_mean_difference_ci_brackets_the_mean_difference():
    signal = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    noise = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    expected_mean = sum(s - n for s, n in zip(signal, noise, strict=False)) / len(signal)
    low, high = mean_difference_ci(signal, noise)
    assert low <= expected_mean <= high


def test_mean_difference_ci_is_symmetric_about_the_mean():
    signal = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    noise = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    mean = sum(s - n for s, n in zip(signal, noise, strict=False)) / len(signal)
    low, high = mean_difference_ci(signal, noise)
    # A t-interval is centered on the point estimate.
    assert (mean - low) == pytest.approx(high - mean, abs=1e-9)


def test_mean_difference_ci_collapses_with_zero_variance():
    # Every pair has the same difference: no spread, so the interval is a point.
    signal = [1.0, 1.0, 1.0, 1.0]
    noise = [0.0, 0.0, 0.0, 0.0]
    low, high = mean_difference_ci(signal, noise)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_mean_difference_ci_higher_confidence_is_wider():
    signal = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    noise = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    low_90, high_90 = mean_difference_ci(signal, noise, confidence=0.90)
    low_99, high_99 = mean_difference_ci(signal, noise, confidence=0.99)
    assert (high_99 - low_99) > (high_90 - low_90)
    assert math.isfinite(low_90) and math.isfinite(high_99)


# --- covariate-adjusted residual ------------------------------------------


def test_covariate_residual_flags_signal_above_background():
    # Everyone reports the symptom while the noise arm rarely does: the residual
    # above the covariate-predicted background is clearly positive.
    n = 20
    signal = [1.0] * n
    noise = [1.0 if i % 7 == 0 else 0.0 for i in range(n)]
    covariates = pd.DataFrame({"age": [float(20 + (i * 7) % 50) for i in range(n)]})
    mean_residual, p_value = covariate_adjusted_residual(signal, noise, covariates)
    assert mean_residual > 0
    assert p_value < 0.05


def test_covariate_residual_is_null_when_signal_matches_noise():
    # Signal equals noise: nothing remains once the fitted noise model is removed.
    n = 30
    noise = [1.0 if i % 3 == 0 else 0.0 for i in range(n)]
    covariates = pd.DataFrame({"age": [float(20 + (i * 11) % 50) for i in range(n)]})
    mean_residual, p_value = covariate_adjusted_residual(noise, noise, covariates)
    assert abs(mean_residual) < 0.05
    assert p_value > 0.2


def test_covariate_residual_falls_back_when_noise_is_empty():
    # No noise events: the expected background is zero, so the residual is just
    # the signal rate and the test reduces to signal-vs-zero.
    n = 20
    signal = [1.0 if i % 2 == 0 else 0.0 for i in range(n)]
    noise = [0.0] * n
    covariates = pd.DataFrame({"age": [float(20 + (i * 7) % 50) for i in range(n)]})
    mean_residual, p_value = covariate_adjusted_residual(signal, noise, covariates)
    assert mean_residual == pytest.approx(sum(signal) / n)
    assert 0.0 <= p_value <= 1.0
