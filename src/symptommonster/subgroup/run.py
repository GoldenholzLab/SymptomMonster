"""Stage 6 driver: stratify top signals by covariate and test for heterogeneity.

For each (group, symptom) pair we measure the symptom's prevalence in the signal
arm and in the noise arm within every level of a covariate (institution, sex, age
band, ...). The per-stratum signal-minus-noise gap shows where the effect lives,
and a chi-square heterogeneity test over the signal arm flags pairs whose rate
genuinely varies across strata rather than by chance.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from ..io import read_jsonl
from .core import stratify, subgroup_heterogeneity


def _index_indicators(path: Path) -> dict[tuple[str, str], dict[str, set[str]]]:
    """Index a normalized JSONL as group -> {patient_id -> symptom set}.

    A patient with no symptoms still registers (empty set) so they count in the
    stratum denominator: a symptom's prevalence is over everyone in the arm, not
    only over the symptomatic.
    """
    by_group: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for row in read_jsonl(path):
        pid = row["patient_id"]
        group = row.get("group") or ""
        symptoms = set(row.get("symptoms", []))
        by_group[group][pid] = by_group[group].get(pid, set()) | symptoms
    return by_group


def _load_covariates(path: Path, columns: list[str]) -> dict[str, dict[str, str]]:
    """Load the covariates CSV as column -> {patient_id -> value}.

    Only the requested stratum columns are retained; blank cells are absent,
    so a patient missing a covariate is excluded from that column's strata.
    """
    per_column: dict[str, dict[str, str]] = {col: {} for col in columns}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return per_column
        available = {name.lower(): name for name in reader.fieldnames}
        pid_col = available.get("patient_id")
        if pid_col is None:
            raise ValueError(f"{path} must have a patient_id column")
        resolved = {}
        for col in columns:
            actual = available.get(col.lower())
            if actual is None:
                raise ValueError(f"covariate column {col!r} not found in {path}")
            resolved[col] = actual
        for row in reader:
            pid = (row.get(pid_col) or "").strip()
            if not pid:
                continue
            for col, actual in resolved.items():
                value = (row.get(actual) or "").strip()
                if value:
                    per_column[col][pid] = value
    return per_column


def _pairs_from_summary(path: Path) -> list[tuple[str, str]]:
    """Read (group, symptom) pairs to test from a prior summary CSV.

    Accepts a ``group`` (or legacy ``drug``) column alongside ``symptom``; the rest
    of the summary is ignored. Order and uniqueness are preserved.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = {name.lower(): name for name in (reader.fieldnames or [])}
        group_col = fields.get("group") or fields.get("drug")
        symptom_col = fields.get("symptom")
        if group_col is None or symptom_col is None:
            raise ValueError(f"{path} must have group/symptom columns")
        for row in reader:
            pair = ((row.get(group_col) or "").strip(), (row.get(symptom_col) or "").strip())
            if all(pair) and pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _pairs_from_signal(
    signal: dict[tuple[str, str], dict[str, set[str]]]
) -> list[tuple[str, str]]:
    """Every (group, symptom) pair that appears anywhere in the signal arm."""
    pairs: set[tuple[str, str]] = set()
    for group, patients in signal.items():
        for symptoms in patients.values():
            for symptom in symptoms:
                pairs.add((group, symptom))
    return sorted(pairs)


def run_subgroup(
    *,
    signal: str,
    noise: str,
    covariates: str,
    strata: str,
    top_from: str | None,
    out: str,
) -> None:
    """Stratified signal/noise analysis -> ``out`` CSV.

    ``strata`` is a comma-separated list of covariate column names. Pairs to test
    come from ``top_from`` (a summary CSV) when given, else from the signal arm.
    """
    strata_columns = [c.strip() for c in strata.split(",") if c.strip()]
    if not strata_columns:
        raise ValueError("strata must name at least one covariate column")

    signal_index = _index_indicators(Path(signal))
    noise_index = _index_indicators(Path(noise))
    covariate_values = _load_covariates(Path(covariates), strata_columns)

    if top_from:
        pairs = _pairs_from_summary(Path(top_from))
    else:
        pairs = _pairs_from_signal(signal_index)

    rows = []
    for group, symptom in pairs:
        signal_patients = signal_index.get(group, {})
        noise_patients = noise_index.get(group, {})
        if not signal_patients:
            continue

        for stratum_type in strata_columns:
            values = covariate_values[stratum_type]
            signal_strata = stratify(values, signal_patients)
            noise_strata = stratify(values, noise_patients)

            # Tabulate every stratum that has signal-arm patients; carry the matching
            # noise-arm rate, and collect signal events for the heterogeneity test.
            event_counts: list[tuple[int, int]] = []
            stratum_rows = []
            for stratum_value in sorted(signal_strata):
                sig_pids = signal_strata[stratum_value]
                noi_pids = noise_strata.get(stratum_value, [])
                n_sig = len(sig_pids)
                n_noi = len(noi_pids)
                sig_events = sum(symptom in signal_patients[p] for p in sig_pids)
                noi_events = sum(symptom in noise_patients[p] for p in noi_pids)

                signal_pct = 100.0 * sig_events / n_sig if n_sig else 0.0
                noise_pct = 100.0 * noi_events / n_noi if n_noi else 0.0
                stratum_rows.append(
                    {
                        "group": group,
                        "symptom": symptom,
                        "stratum_type": stratum_type,
                        "stratum_value": stratum_value,
                        "n_patients": n_sig,
                        "signal_pct": round(signal_pct, 2),
                        "noise_pct": round(noise_pct, 2),
                        "signal_minus_noise": round(signal_pct - noise_pct, 2),
                    }
                )
                event_counts.append((sig_events, n_sig))

            het_p = subgroup_heterogeneity(event_counts)
            het_str = "" if math.isnan(het_p) else round(het_p, 6)
            for stratum_row in stratum_rows:
                stratum_row["heterogeneity_p"] = het_str
                rows.append(stratum_row)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "symptom",
        "stratum_type",
        "stratum_value",
        "n_patients",
        "signal_pct",
        "noise_pct",
        "signal_minus_noise",
        "heterogeneity_p",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
