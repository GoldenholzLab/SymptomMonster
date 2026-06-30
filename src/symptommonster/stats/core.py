"""Statistical primitives for per-pair signal-vs-noise testing.

Each (group, symptom) gives a per-patient paired difference d = signal - noise,
where signal and noise are 0/1 symptom indicators from the real and surrogate
runs. The primary test is one-sample on d; FDR control is applied across a
group; rates and their difference are reported with exact / t intervals. A
Beta-Binomial posterior provides a distribution-free cross-check, and a
covariate-adjusted residual test confirms a gap is not explained by patient mix.

These are pure functions over arrays (the covariate test also takes a frame);
no file IO, so they stay easy to unit test. Heavy optional dependencies are
imported inside the functions that need them to keep import time low.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.10) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR control. Returns (qvalues, rejected) in input order.

    The q-value of a hypothesis is the smallest FDR level at which it would be
    rejected; we compute it by the standard cumulative-min from the largest
    p-value down, which makes the q-values monotone in sorted-p order and never
    smaller than the raw p-value. `rejected` marks hypotheses significant at the
    requested `q`.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return [], []

    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)

    # Step-up q-values: running min of (m/k) * p_(k) from the largest p down.
    scaled = p[order] * m / ranks
    qvals_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    qvals_sorted = np.clip(qvals_sorted, 0.0, 1.0)

    qvalues = np.empty(m, dtype=float)
    qvalues[order] = qvals_sorted
    rejected = qvalues <= q
    return qvalues.tolist(), rejected.tolist()


def paired_ttest(signal: Sequence[float], noise: Sequence[float]) -> tuple[float, float]:
    """One-sample two-sided t-test on the paired difference d = signal - noise.

    Returns (t_stat, p_value). Degenerate handling when the differences have zero
    variance (every patient agrees): if the mean difference is also zero there is
    nothing to detect, so we return (0.0, 1.0); otherwise the effect is perfectly
    consistent and we report an infinite t with p = 0.0 (signed +/-inf by the
    direction of the mean).
    """
    d = np.asarray(signal, dtype=float) - np.asarray(noise, dtype=float)
    n = d.size
    if n < 2:
        # A single pair carries no within-sample variance estimate.
        return (0.0, 1.0)

    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        if mean == 0.0:
            return (0.0, 1.0)
        return (math.inf if mean > 0 else -math.inf, 0.0)

    t_stat, p_value = stats.ttest_1samp(d, 0.0)
    return float(t_stat), float(p_value)


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval on the proportion.

    Returns (lower, upper) bounds in [0, 1] for the success probability given `k`
    successes in `n` trials. The interval is inverted from the Beta distribution;
    the endpoints degenerate correctly: k=0 has lower bound 0.0 and k=n has upper
    bound 1.0. An empty sample returns the whole interval (0.0, 1.0).
    """
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


def mean_difference_ci(
    signal: Sequence[float],
    noise: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Student-t confidence interval on the mean paired difference (proportion scale).

    Centered on mean(signal - noise). With one pair, or with zero variance, the
    interval collapses to the point estimate, since there is no spread to express.
    """
    d = np.asarray(signal, dtype=float) - np.asarray(noise, dtype=float)
    n = d.size
    if n == 0:
        return (0.0, 0.0)
    mean = float(d.mean())
    if n < 2:
        return (mean, mean)

    se = float(d.std(ddof=1)) / math.sqrt(n)
    if se == 0.0:
        return (mean, mean)
    tcrit = float(stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1))
    return (mean - tcrit * se, mean + tcrit * se)


def covariate_adjusted_residual(signal: Sequence[float], noise: Sequence[float], covariates) -> tuple[float, float]:
    """Covariate-adjusted confirmation that a signal exceeds its background.

    A raw signal-minus-noise gap could reflect patient mix rather than the drug.
    To guard against that, fit the per-patient noise indicator on the covariates
    to obtain an expected background rate for each patient, then test the drugged
    residuals (signal minus expected) against zero with a one-sample two-sided
    t-test. Returns (mean_residual, p_value) on the proportion scale; a small
    p-value means the gap survives adjustment for the supplied covariates.

    `covariates` is a row-per-patient frame aligned to `signal` and `noise`.
    """
    expected = _noise_expected(noise, covariates)
    return _residual_ttest(signal, expected)


def _noise_expected(noise: Sequence[float], covariates) -> np.ndarray:
    """Expected background rate per patient from a logistic fit of the noise arm.

    Falls back to the unconditional background rate when the fit is undefined: a
    symptom that is all-present or all-absent in the noise arm has no MLE, and a
    rank-deficient design or a non-converging fit leaves nothing better than the
    overall mean.
    """
    import pandas as pd
    import statsmodels.api as sm

    y = np.asarray(noise, dtype=float)
    n = y.size
    total = y.sum()
    if total == 0.0:
        return np.zeros(n)
    if total == n:
        return np.ones(n)

    try:
        design = pd.get_dummies(covariates.reset_index(drop=True), drop_first=True).astype(float)
        design = sm.add_constant(design, has_constant="add")
        model = sm.GLM(y, design, family=sm.families.Binomial()).fit()
        predicted = np.asarray(model.predict(design), dtype=float)
        if predicted.shape == y.shape and np.all(np.isfinite(predicted)):
            return predicted
    except Exception:
        pass
    return np.full(n, float(y.mean()))


def _residual_ttest(signal: Sequence[float], expected: np.ndarray) -> tuple[float, float]:
    """One-sample two-sided t-test of (signal - expected) against zero."""
    r = np.asarray(signal, dtype=float) - np.asarray(expected, dtype=float)
    n = r.size
    if n < 2:
        return (float(r.mean()) if n else 0.0, 1.0)
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return (mean, 1.0 if mean == 0.0 else 0.0)
    t_stat = mean / (sd / math.sqrt(n))
    return (mean, float(2.0 * stats.t.sf(abs(t_stat), df=n - 1)))


@dataclass
class BayesResult:
    """Posterior summary for one (group, symptom) comparison.

    `prob_signal_gt_noise` is the posterior probability the signal rate exceeds
    the noise rate; the remaining fields summarize the posterior of their
    difference (signal_rate - noise_rate) on the proportion scale.
    """

    prob_signal_gt_noise: float
    mean_diff: float
    ci_lower: float
    ci_upper: float


def beta_binomial_compare(
    signal_count: int,
    signal_total: int,
    noise_count: int,
    noise_total: int,
    *,
    draws: int = 10000,
    seed: int = 0,
    prior: tuple[float, float] = (1.0, 1.0),
) -> BayesResult:
    """Monte-Carlo comparison of two binomial rates under Beta priors.

    A sensitivity analysis that asks the t-test's question without its
    distributional assumptions. The posterior for each rate is
    Beta(prior_a + count, prior_b + total - count); we draw from both and report
    P(signal > noise) and the mean and central 95% credible interval of the
    difference. Deterministic given `seed`. Equal counts over equal totals give P
    near 0.5; a strong signal over an empty noise floor pushes P toward 1.
    """
    a, b = prior
    rng = np.random.default_rng(seed)

    signal_rate = rng.beta(a + signal_count, b + max(signal_total - signal_count, 0), size=draws)
    noise_rate = rng.beta(a + noise_count, b + max(noise_total - noise_count, 0), size=draws)
    diff = signal_rate - noise_rate

    return BayesResult(
        prob_signal_gt_noise=float(np.mean(signal_rate > noise_rate)),
        mean_diff=float(np.mean(diff)),
        ci_lower=float(np.percentile(diff, 2.5)),
        ci_upper=float(np.percentile(diff, 97.5)),
    )
