"""Orchestrate per-pair testing over the signal and noise normalized runs.

Both runs are normalized JSONL of ``{patient_id, group, symptoms}``. For every
group we take its signal symptoms as the candidate set, then for each symptom
build paired 0/1 vectors over the patients present in *both* runs, test the
paired difference, control FDR (within each group by default, or across all
pairs with ``scope="global"``), and report rates with exact intervals. When a
covariate table is supplied, each pair also gets a covariate-adjusted residual
test and its own FDR pass. `run_stats` writes the frequentist tables; `run_bayes`
writes the Beta-Binomial sensitivity tables. Percentages are used throughout to
match how the rates are read in the paper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from symptommonster.io import read_jsonl

from .core import (
    benjamini_hochberg,
    beta_binomial_compare,
    clopper_pearson,
    covariate_adjusted_residual,
    mean_difference_ci,
    paired_ttest,
)

# Per-pair frequentist table.
STATS_FIELDS = [
    "symptom",
    "n_patients",
    "total_patients",
    "signal_pct",
    "noise_n",
    "noise_pct",
    "signal_minus_noise",
    "t_stat",
    "p_value",
    "q_value",
    "significant",
    "direction",
    "diff_ci_lower",
    "diff_ci_upper",
    "signal_ci_lower",
    "signal_ci_upper",
    "noise_ci_lower",
    "noise_ci_upper",
]

# Appended only when covariates are supplied: the covariate-adjusted residual test.
LR_FIELDS = ["lr_resid_mean", "lr_resid_p", "lr_resid_q", "lr_resid_significant"]

# Combined cross-group summary (a thin slice of the per-pair table).
SUMMARY_FIELDS = [
    "group",
    "symptom",
    "signal_pct",
    "noise_pct",
    "signal_minus_noise",
    "p_value",
    "q_value",
    "significant",
    "direction",
]

# Bayesian per-group table = identifying columns plus the posterior summary.
BAYES_FIELDS = [
    "group",
    "symptom",
    "n_patients",
    "total_patients",
    "noise_n",
    "signal_pct",
    "noise_pct",
    "signal_minus_noise",
    "posterior_prob_signal_gt_noise",
    "posterior_mean_diff",
    "bayes_ci_lower",
    "bayes_ci_upper",
    "bayes_significant",
]


def _load_by_group(path: str) -> dict[str, dict[str, set[str]]]:
    """Read normalized JSONL into group -> patient_id -> set(symptoms)."""
    by_group: dict[str, dict[str, set[str]]] = {}
    for row in read_jsonl(path):
        group = row.get("group")
        by_group.setdefault(group, {})[row["patient_id"]] = set(row.get("symptoms", []))
    return by_group


def _paired_patients(
    signal_group: dict[str, set[str]],
    noise_group: dict[str, set[str]],
) -> list[str]:
    """Patients present in both runs, in a stable order, so pairing is well defined."""
    return sorted(set(signal_group) & set(noise_group))


def _candidate_symptoms(signal_group: dict[str, set[str]]) -> list[str]:
    """Symptoms to test for a group = everything its signal run ever reported."""
    seen: set[str] = set()
    for symptoms in signal_group.values():
        seen.update(symptoms)
    return sorted(seen)


def _load_covariates(path: str) -> pd.DataFrame:
    """Read the covariate table, keyed by patient_id."""
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if "patient_id" not in frame.columns:
        raise ValueError("covariates CSV must have a patient_id column")
    return frame.set_index("patient_id")


def _covariate_rows(covariates: pd.DataFrame, patients: list[str]) -> pd.DataFrame:
    """Covariate rows for `patients`, in order, with patient_id dropped."""
    missing = [p for p in patients if p not in covariates.index]
    if missing:
        raise ValueError(f"covariates missing for {len(missing)} patients (for example {missing[0]!r})")
    return covariates.loc[patients].reset_index(drop=True)


def _assign_fdr(
    group_rows: dict[str, list[dict]],
    p_key: str,
    q_key: str,
    sig_key: str,
    q: float,
    scope: str,
) -> None:
    """Fill q-value and significance columns by BH-FDR, within group or globally."""
    if scope == "global":
        index = [(group, i) for group, rows in group_rows.items() for i in range(len(rows))]
        pvalues = [group_rows[g][i][p_key] for g, i in index]
        qvalues, rejected = benjamini_hochberg(pvalues, q=q)
        for (g, i), qvalue, is_rejected in zip(index, qvalues, rejected, strict=True):
            group_rows[g][i][q_key] = qvalue
            group_rows[g][i][sig_key] = bool(is_rejected)
    else:
        for rows in group_rows.values():
            qvalues, rejected = benjamini_hochberg([r[p_key] for r in rows], q=q)
            for row, qvalue, is_rejected in zip(rows, qvalues, rejected, strict=True):
                row[q_key] = qvalue
                row[sig_key] = bool(is_rejected)


def run_stats(
    *,
    signal: str,
    noise: str,
    out_dir: str,
    fdr_q: float = 0.10,
    alpha: float = 0.05,
    scope: str = "within-drug",
    covariates: str | None = None,
    seed: int = 0,
) -> None:
    """Paired t-test per (group, symptom) with BH-FDR; write CSVs.

    `scope` chooses the FDR family: "within-drug" (the primary analysis) corrects
    each group on its own; "global" pools every pair into one correction as a
    sensitivity analysis. Supplying `covariates` adds the covariate-adjusted
    residual test (its own column block and FDR pass). `seed` is accepted for a
    uniform entrypoint signature; the frequentist path is deterministic.
    """
    confidence = 1.0 - alpha
    signal_by_group = _load_by_group(signal)
    noise_by_group = _load_by_group(noise)
    cov_table = _load_covariates(covariates) if covariates else None

    stats_dir = Path(out_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    group_rows: dict[str, list[dict]] = {}
    for group in sorted(signal_by_group, key=lambda g: (g is None, g)):
        signal_group = signal_by_group[group]
        noise_group = noise_by_group.get(group, {})
        patients = _paired_patients(signal_group, noise_group)
        n = len(patients)
        if n == 0:
            continue
        cov_rows = _covariate_rows(cov_table, patients) if cov_table is not None else None

        rows: list[dict] = []
        for symptom in _candidate_symptoms(signal_group):
            sig = [1.0 if symptom in signal_group[p] else 0.0 for p in patients]
            noi = [1.0 if symptom in noise_group[p] else 0.0 for p in patients]
            n_sig = int(sum(sig))
            n_noi = int(sum(noi))

            t_stat, p_value = paired_ttest(sig, noi)
            diff_lo, diff_hi = mean_difference_ci(sig, noi, confidence=confidence)
            sig_lo, sig_hi = clopper_pearson(n_sig, n, confidence=confidence)
            noi_lo, noi_hi = clopper_pearson(n_noi, n, confidence=confidence)
            mean_diff = (n_sig - n_noi) / n

            row = {
                "symptom": symptom,
                "n_patients": n_sig,
                "total_patients": n,
                "signal_pct": 100.0 * n_sig / n,
                "noise_n": n_noi,
                "noise_pct": 100.0 * n_noi / n,
                "signal_minus_noise": 100.0 * mean_diff,
                "t_stat": t_stat,
                "p_value": p_value,
                "direction": "increased" if mean_diff > 0 else "decreased",
                "diff_ci_lower": 100.0 * diff_lo,
                "diff_ci_upper": 100.0 * diff_hi,
                "signal_ci_lower": 100.0 * sig_lo,
                "signal_ci_upper": 100.0 * sig_hi,
                "noise_ci_lower": 100.0 * noi_lo,
                "noise_ci_upper": 100.0 * noi_hi,
            }
            if cov_rows is not None:
                lr_mean, lr_p = covariate_adjusted_residual(sig, noi, cov_rows)
                row["lr_resid_mean"] = 100.0 * lr_mean
                row["lr_resid_p"] = lr_p
            rows.append(row)

        if rows:
            group_rows[group] = rows

    _assign_fdr(group_rows, "p_value", "q_value", "significant", fdr_q, scope)
    if cov_table is not None:
        _assign_fdr(group_rows, "lr_resid_p", "lr_resid_q", "lr_resid_significant", fdr_q, scope)

    fields = STATS_FIELDS + (LR_FIELDS if cov_table is not None else [])
    summary_rows: list[dict] = []
    for group, rows in group_rows.items():
        rows.sort(key=lambda r: r["signal_minus_noise"], reverse=True)
        pd.DataFrame(rows, columns=fields).to_csv(stats_dir / f"{group}.csv", index=False)
        for row in rows:
            summary_rows.append({"group": group, **{k: row[k] for k in SUMMARY_FIELDS[1:]}})

    pd.DataFrame(summary_rows, columns=SUMMARY_FIELDS).to_csv(
        Path(out_dir) / "summary.csv", index=False
    )


def run_bayes(
    *,
    signal: str,
    noise: str,
    out_dir: str,
    draws: int = 10000,
    threshold: float = 0.95,
    seed: int = 0,
) -> None:
    """Beta-Binomial posterior P(signal>noise) per (group, symptom); write CSVs."""
    signal_by_group = _load_by_group(signal)
    noise_by_group = _load_by_group(noise)

    bayes_dir = Path(out_dir) / "bayes"
    bayes_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for group in sorted(signal_by_group, key=lambda g: (g is None, g)):
        signal_group = signal_by_group[group]
        noise_group = noise_by_group.get(group, {})
        patients = _paired_patients(signal_group, noise_group)
        n = len(patients)
        if n == 0:
            continue

        rows: list[dict] = []
        for symptom in _candidate_symptoms(signal_group):
            n_sig = sum(1 for p in patients if symptom in signal_group[p])
            n_noi = sum(1 for p in patients if symptom in noise_group[p])
            result = beta_binomial_compare(n_sig, n, n_noi, n, draws=draws, seed=seed)

            rows.append(
                {
                    "group": group,
                    "symptom": symptom,
                    "n_patients": n_sig,
                    "total_patients": n,
                    "noise_n": n_noi,
                    "signal_pct": 100.0 * n_sig / n,
                    "noise_pct": 100.0 * n_noi / n,
                    "signal_minus_noise": 100.0 * (n_sig - n_noi) / n,
                    "posterior_prob_signal_gt_noise": result.prob_signal_gt_noise,
                    "posterior_mean_diff": 100.0 * result.mean_diff,
                    "bayes_ci_lower": 100.0 * result.ci_lower,
                    "bayes_ci_upper": 100.0 * result.ci_upper,
                    "bayes_significant": result.prob_signal_gt_noise >= threshold,
                }
            )

        if not rows:
            continue

        rows.sort(key=lambda r: r["posterior_prob_signal_gt_noise"], reverse=True)
        pd.DataFrame(rows, columns=BAYES_FIELDS).to_csv(bayes_dir / f"{group}.csv", index=False)
        summary_rows.extend(rows)

    pd.DataFrame(summary_rows, columns=BAYES_FIELDS).to_csv(
        Path(out_dir) / "bayes_summary.csv", index=False
    )
