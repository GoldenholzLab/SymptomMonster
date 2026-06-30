"""Stage 7 driver: score every model's extractions against annotator ground truth.

Annotators live as one CSV each in a directory; models live as one normalized
JSONL each in another. We build a single ground-truth set per patient from the
annotators (under a chosen aggregation), canonicalize every term so synonyms
align, then report macro F1, symptomatic-only F1, a population-prevalence
weighted MAE, and extraction bias per model, one CSV row per model.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import isnan, nan
from pathlib import Path

from ..io import read_jsonl
from .agreement import krippendorff_alpha
from .equivalence import EquivalenceMatcher
from .metrics import (
    extraction_bias,
    macro_f1,
    precision_recall_f1,
    symptomatic_f1,
    weighted_mae,
)

_EMPTY_TOKENS = {"", "none", "n/a", "na", "no symptoms", "denies"}


def _read_annotator_csv(path: Path) -> dict[str, set[str]]:
    """Load one annotator CSV as patient_id -> set of raw symptom terms.

    Expects a ``patient_id`` column and a ``symptoms`` column of comma-separated
    terms; "none"/empty means the annotator recorded no symptoms (an empty set,
    which still counts as a judged patient).
    """
    annotations: dict[str, set[str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return annotations
        fields = {name.lower(): name for name in reader.fieldnames}
        pid_col = fields.get("patient_id")
        sym_col = fields.get("symptoms")
        if pid_col is None or sym_col is None:
            raise ValueError(f"{path} must have patient_id and symptoms columns")
        for row in reader:
            pid = (row.get(pid_col) or "").strip()
            if not pid:
                continue
            raw = (row.get(sym_col) or "").strip()
            if raw.lower() in _EMPTY_TOKENS:
                annotations[pid] = set()
                continue
            annotations[pid] = {t.strip() for t in raw.split(",") if t.strip()}
    return annotations


def _build_ground_truth(
    annotator_sets: list[dict[str, set[str]]],
    matcher: EquivalenceMatcher,
    gt_mode: str,
) -> dict[str, set[str]]:
    """Aggregate per-annotator sets into one canonical truth set per patient.

    gt_mode:
      - ``union``: a symptom counts if any annotator recorded it.
      - ``majority``: it counts if at least half of the annotators who saw the
        patient recorded it (ties at exactly half count, matching ">= half").
      - ``intersection``: it counts only if every annotator who saw the patient
        recorded it.
    All terms are canonicalized first so synonyms agree before voting.
    """
    by_patient: dict[str, list[set[str]]] = defaultdict(list)
    for annotator in annotator_sets:
        for pid, symptoms in annotator.items():
            by_patient[pid].append({matcher.canonical(s) for s in symptoms})

    truth: dict[str, set[str]] = {}
    for pid, votes in by_patient.items():
        if gt_mode == "union":
            truth[pid] = set().union(*votes) if votes else set()
        elif gt_mode == "majority":
            counts: dict[str, int] = defaultdict(int)
            for vote in votes:
                for term in vote:
                    counts[term] += 1
            threshold = len(votes) / 2
            truth[pid] = {term for term, c in counts.items() if c >= threshold}
        elif gt_mode == "intersection":
            truth[pid] = set.intersection(*votes) if votes else set()
        else:
            raise ValueError(f"unknown gt_mode: {gt_mode!r}")
    return truth


def _read_model_jsonl(
    path: Path, matcher: EquivalenceMatcher
) -> tuple[dict[str, set[str]], float]:
    """Load one model's normalized JSONL as canonical sets plus mean inference time.

    Returns ``(predictions, mean_inference_s)``; the time is ``nan`` when no record
    carries an ``extraction_time_s`` field.
    """
    predictions: dict[str, set[str]] = {}
    times: list[float] = []
    for row in read_jsonl(path):
        pid = row["patient_id"]
        predictions[pid] = {matcher.canonical(s) for s in row.get("symptoms", [])}
        t = row.get("extraction_time_s")
        if t is not None:
            times.append(float(t))
    mean_time = sum(times) / len(times) if times else nan
    return predictions, mean_time


def _prevalence_vectors(
    predictions: dict[str, set[str]],
    truth: dict[str, set[str]],
    patients: list[str],
) -> tuple[list[float], list[float]]:
    """Per-symptom prevalence for model and truth over a fixed patient denominator.

    Both vectors are aligned to the union of symptoms; patients absent from either
    map contribute a zero, which is what keeps a never-extracted symptom honest.
    """
    n = len(patients)
    symptoms = sorted(
        {s for p in patients for s in predictions.get(p, set())}
        | {s for p in patients for s in truth.get(p, set())}
    )
    pred_rates, truth_rates = [], []
    for symptom in symptoms:
        pred_rates.append(sum(symptom in predictions.get(p, set()) for p in patients) / n)
        truth_rates.append(sum(symptom in truth.get(p, set()) for p in patients) / n)
    return pred_rates, truth_rates


def run_benchmark(*, extractions: str, annotations: str, gt_mode: str, out: str) -> None:
    """Score each model JSONL in ``extractions`` against ``annotations`` -> ``out`` CSV."""
    matcher = EquivalenceMatcher()

    annotator_paths = sorted(Path(annotations).glob("*.csv"))
    if not annotator_paths:
        raise FileNotFoundError(f"no annotator CSVs in {annotations}")
    annotator_sets = [_read_annotator_csv(p) for p in annotator_paths]
    truth = _build_ground_truth(annotator_sets, matcher, gt_mode)

    # Inter-annotator agreement on the patients every annotator judged, reported to
    # stderr as context for the scores (not part of the per-model output).
    _report_agreement(annotator_sets, matcher)

    model_paths = sorted(Path(extractions).glob("*.jsonl"))
    if not model_paths:
        raise FileNotFoundError(f"no model JSONL files in {extractions}")

    rows = []
    for path in model_paths:
        predictions, mean_time = _read_model_jsonl(path, matcher)
        # Score over patients with ground truth; that is the evaluable set.
        patients = sorted(truth)
        pred_rates, truth_rates = _prevalence_vectors(predictions, truth, patients)
        rows.append(
            {
                "model": path.stem,
                "macro_f1": round(macro_f1(predictions, truth), 6),
                "symptomatic_f1": round(symptomatic_f1(predictions, truth), 6),
                "weighted_mae": round(weighted_mae(pred_rates, truth_rates, weights=truth_rates), 6),
                "mean_inference_s": "" if isnan(mean_time) else round(mean_time, 4),
                "bias": round(extraction_bias(predictions, truth), 6),
            }
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "macro_f1",
        "symptomatic_f1",
        "weighted_mae",
        "mean_inference_s",
        "bias",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_f1(
    annotator_sets: list[dict[str, set[str]]], matcher: EquivalenceMatcher
) -> list[float]:
    """Mean per-patient set-F1 for each annotator pair over the patients both judged.

    Set-F1 is symmetric, so a pair contributes one score regardless of order.
    """
    canon = [
        {pid: {matcher.canonical(s) for s in syms} for pid, syms in annotator.items()}
        for annotator in annotator_sets
    ]
    scores: list[float] = []
    for i in range(len(canon)):
        for j in range(i + 1, len(canon)):
            shared = sorted(set(canon[i]) & set(canon[j]))
            if not shared:
                continue
            f1s = [precision_recall_f1(canon[i][pid], canon[j][pid])[2] for pid in shared]
            scores.append(sum(f1s) / len(f1s))
    return scores


def _report_agreement(
    annotator_sets: list[dict[str, set[str]]], matcher: EquivalenceMatcher
) -> None:
    """Print inter-annotator agreement to stderr as context for the model scores.

    Pairwise set-F1 is the headline measure, matching the set-F1 used to score the
    models: information-retrieval agreement has no stable negative class, so a
    chance-corrected coefficient understates it. Krippendorff's alpha follows as a
    conventional cross-check over the patients every annotator judged.
    """
    if len(annotator_sets) < 2:
        return

    f1s = _pairwise_f1(annotator_sets, matcher)
    if f1s:
        print(
            f"inter-annotator pairwise F1: {min(f1s):.3f} to {max(f1s):.3f} "
            f"(mean {sum(f1s) / len(f1s):.3f}, {len(f1s)} pairs)",
            file=sys.stderr,
        )

    shared = set(annotator_sets[0])
    for annotator in annotator_sets[1:]:
        shared &= set(annotator)
    if not shared:
        return

    canon = [
        {pid: {matcher.canonical(s) for s in annotator[pid]} for pid in shared}
        for annotator in annotator_sets
    ]
    vocabulary = sorted({s for annotator in canon for syms in annotator.values() for s in syms})
    if not vocabulary:
        print("inter-annotator alpha: n/a (no symptoms among shared patients)", file=sys.stderr)
        return

    # One binary judgement per (patient, symptom) per annotator.
    reliability = [
        [1.0 if symptom in annotator[pid] else 0.0 for pid in sorted(shared) for symptom in vocabulary]
        for annotator in canon
    ]
    alpha = krippendorff_alpha(reliability)
    print(
        f"inter-annotator Krippendorff alpha: {alpha:.4f} "
        f"({len(shared)} shared patients, {len(vocabulary)} symptoms)",
        file=sys.stderr,
    )
