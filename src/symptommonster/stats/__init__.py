"""Stages 4-5: paired frequentist testing with FDR control, plus a Bayesian check."""

from .core import (
    BayesResult,
    benjamini_hochberg,
    beta_binomial_compare,
    clopper_pearson,
    covariate_adjusted_residual,
    mean_difference_ci,
    paired_ttest,
)
from .run import run_bayes, run_stats

__all__ = [
    "benjamini_hochberg",
    "paired_ttest",
    "clopper_pearson",
    "mean_difference_ci",
    "covariate_adjusted_residual",
    "BayesResult",
    "beta_binomial_compare",
    "run_stats",
    "run_bayes",
]
