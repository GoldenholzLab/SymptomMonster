"""Inter-annotator agreement coefficients, implemented from their definitions.

All three correct the observed agreement for the agreement expected by chance, so
a value near 0 means "no better than chance" and 1 means perfect. They differ in
shape of input: Cohen's kappa compares two raters' label vectors, Fleiss' kappa
takes a per-item category-count table for any number of raters, and Krippendorff's
alpha tolerates missing values in a raters-by-items matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cohen_kappa(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Cohen's kappa for two raters over paired nominal labels.

    Identical label vectors give ``1.0``; agreement at the chance rate gives ``~0``.
    Labels may be any hashable integers (binary is the common case here). When both
    raters use a single category for every item, observed and expected agreement are
    both 1, which we report as perfect agreement.
    """
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    if a.shape != b.shape:
        raise ValueError("rater_a and rater_b must have the same length")
    n = a.size
    if n == 0:
        return float("nan")

    observed = np.mean(a == b)

    # Expected agreement: sum over categories of the product of marginal rates.
    categories = np.union1d(a, b)
    expected = 0.0
    for c in categories:
        expected += np.mean(a == c) * np.mean(b == c)

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa from a ``(n_items, n_categories)`` table of rating counts.

    ``counts[i, c]`` is how many raters assigned item ``i`` to category ``c``. Every
    item must receive the same number of ratings. Perfect agreement -> ``1.0``;
    chance-level -> ``~0``. If raters never disagree because the expected agreement
    is already 1 (a single category used throughout), we report ``1.0``.
    """
    counts = np.asarray(counts, dtype=float)
    n_items, n_categories = counts.shape
    if n_items == 0:
        return float("nan")

    raters_per_item = counts.sum(axis=1)
    n = raters_per_item[0]
    if not np.allclose(raters_per_item, n):
        raise ValueError("every item must have the same number of ratings")
    if n < 2:
        return float("nan")

    # Per-item agreement P_i: fraction of rater pairs that concur.
    p_item = (np.sum(counts**2, axis=1) - n) / (n * (n - 1))
    p_bar = float(np.mean(p_item))

    # Expected agreement from overall category proportions.
    p_category = counts.sum(axis=0) / (n_items * n)
    p_expected = float(np.sum(p_category**2))

    if p_expected >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_expected) / (1.0 - p_expected)


def krippendorff_alpha(reliability: Sequence[Sequence[float | None]]) -> float:
    """Krippendorff's alpha (nominal metric) over a raters-by-items matrix.

    ``reliability[r][i]`` is rater ``r``'s value for item ``i``, or ``None`` where the
    rater did not judge that item. Only items rated by at least two raters contribute.
    Perfect agreement -> ``1.0``; chance -> ``~0``; systematic disagreement -> negative.

    Computed straight from the coincidence matrix:
        alpha = 1 - D_observed / D_expected
    where the nominal distance is 0 for equal values and 1 otherwise.
    """
    matrix = [list(row) for row in reliability]
    if not matrix:
        return float("nan")
    n_items = len(matrix[0])
    if any(len(row) != n_items for row in matrix):
        raise ValueError("all raters must cover the same number of items")

    # Build the coincidence matrix. For each item with m>=2 present values, every
    # ordered pair of distinct positions contributes 1/(m-1) to coincidence[a, b].
    # Its margins n_v then give the chance model directly.
    coincidence: dict[tuple[object, object], float] = {}
    for i in range(n_items):
        present = [row[i] for row in matrix if row[i] is not None]
        m = len(present)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for j, a in enumerate(present):
            for k, b in enumerate(present):
                if j != k:
                    coincidence[(a, b)] = coincidence.get((a, b), 0.0) + weight

    n_total = sum(coincidence.values())
    if n_total == 0:
        return float("nan")

    # Margins: total coincidence mass per value.
    margins: dict[object, float] = {}
    for (a, _b), c in coincidence.items():
        margins[a] = margins.get(a, 0.0) + c
    values = list(margins)

    # Nominal metric: only off-diagonal coincidences are disagreements.
    d_observed = sum(c for (a, b), c in coincidence.items() if a != b)

    # Expected disagreement under independence, with the finite-sample correction.
    d_expected = 0.0
    for a in values:
        for b in values:
            if a != b:
                d_expected += margins[a] * margins[b]
    d_expected /= n_total - 1

    if d_expected == 0:
        return 1.0 if d_observed == 0 else 0.0
    return 1.0 - d_observed / d_expected
