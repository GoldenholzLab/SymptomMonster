"""Tests for the Beta-Binomial comparison in ``symptommonster.stats.core``."""

from __future__ import annotations

import pytest

from symptommonster.stats.core import BayesResult, beta_binomial_compare


def test_equal_rates_give_roughly_even_odds():
    # Same successes over the same totals: neither rate is favored.
    result = beta_binomial_compare(10, 40, 10, 40, draws=20000, seed=0)
    assert isinstance(result, BayesResult)
    assert result.prob_signal_gt_noise == pytest.approx(0.5, abs=0.1)
    # The posterior of the difference should sit on top of zero.
    assert result.mean_diff == pytest.approx(0.0, abs=0.05)


def test_strong_signal_over_empty_noise_is_near_certain():
    # High signal rate, almost no noise: signal almost surely exceeds noise.
    result = beta_binomial_compare(45, 50, 1, 50, draws=20000, seed=0)
    assert result.prob_signal_gt_noise > 0.9
    assert result.mean_diff > 0.0


def test_strong_noise_over_weak_signal_is_near_zero():
    # The mirror image: noise dominates, so P(signal > noise) collapses.
    result = beta_binomial_compare(1, 50, 45, 50, draws=20000, seed=0)
    assert result.prob_signal_gt_noise < 0.1
    assert result.mean_diff < 0.0


def test_credible_interval_brackets_the_mean_difference():
    result = beta_binomial_compare(30, 50, 10, 50, draws=20000, seed=1)
    assert result.ci_lower <= result.mean_diff <= result.ci_upper
    # Difference of proportions stays within [-1, 1].
    assert -1.0 <= result.ci_lower <= result.ci_upper <= 1.0


def test_same_seed_is_deterministic():
    a = beta_binomial_compare(20, 60, 8, 60, draws=10000, seed=7)
    b = beta_binomial_compare(20, 60, 8, 60, draws=10000, seed=7)
    assert a.prob_signal_gt_noise == b.prob_signal_gt_noise
    assert a.mean_diff == b.mean_diff
    assert a.ci_lower == b.ci_lower
    assert a.ci_upper == b.ci_upper


def test_different_seeds_stay_close():
    # Monte-Carlo noise should be small at this many draws, so two seeds agree
    # closely even though they are not bit-identical.
    a = beta_binomial_compare(30, 50, 10, 50, draws=20000, seed=1)
    b = beta_binomial_compare(30, 50, 10, 50, draws=20000, seed=2)
    assert a.prob_signal_gt_noise == pytest.approx(b.prob_signal_gt_noise, abs=0.05)
    assert a.mean_diff == pytest.approx(b.mean_diff, abs=0.03)
