"""Pure helpers for subgroup heterogeneity testing.

A signal that holds overall may be carried by one stratum (an age band, a site)
and absent in another. ``subgroup_heterogeneity`` asks whether the event rate
genuinely differs across strata via a chi-square test on the stratum-by-outcome
table; ``stratify`` is the bookkeeping that buckets patients by covariate value.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from scipy.stats import chi2_contingency


def subgroup_heterogeneity(strata_counts: Sequence[tuple[int, int]]) -> float:
    """Chi-square p-value for differing event proportions across strata.

    Each ``(n_event, n_total)`` is one stratum; the test runs on the 2xK table of
    ``[events, non-events]``. Equal proportions across strata give a high p-value
    (no heterogeneity); divergent proportions give a low one.

    Returns ``nan`` for degenerate tables: fewer than two strata with any
    patients, or a row/column collapse that makes the test ill-defined, such as an
    all-zero outcome where chi-square cannot be computed.
    """
    table = []
    for n_event, n_total in strata_counts:
        if n_total <= 0:
            continue
        if n_event < 0 or n_event > n_total:
            raise ValueError("n_event must lie within [0, n_total]")
        table.append([n_event, n_total - n_event])

    if len(table) < 2:
        return float("nan")

    # A column that is entirely zero (every patient an event, or none) leaves the
    # contingency test with a zero marginal; there is no proportion to compare.
    col_sums = [sum(row[c] for row in table) for c in (0, 1)]
    if 0 in col_sums:
        return float("nan")

    try:
        _, p, _, _ = chi2_contingency(table)
    except ValueError:
        return float("nan")
    return float(p)


def stratify(values: Mapping[str, str], patient_ids: Iterable[str]) -> dict[str, list[str]]:
    """Group patient ids by their stratum value.

    Patients missing from ``values`` (no covariate recorded) are dropped rather
    than bucketed under a sentinel, so a stratum only ever holds real members.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    for pid in patient_ids:
        stratum = values.get(pid)
        if stratum is None or stratum == "":
            continue
        buckets[stratum].append(pid)
    return dict(buckets)
