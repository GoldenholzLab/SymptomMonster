"""Orchestrate the three tiers into a build phase and an apply phase.

Build collects every unique raw term, routes each through the tiers, and writes
the reviewable mapping CSV. Apply reads that CSV (after any human edits) and
rewrites the extraction records with canonical symptoms. Splitting the two lets
a reviewer inspect and correct the mapping before it touches the data.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable

from ..io import ExtractionRecord, NormalizedRecord, read_jsonl, write_jsonl
from ..llm import get_client
from .mapping import FILTERED, MappingRow, save_mapping
from .text import is_non_symptom, preprocess_term, split_compound
from .tiers import CtcaeMatcher, LLMTier, apply_rules, load_preferred_terms, load_rules


def _canonical(term: str) -> str:
    """Present a canonical term consistently in Title Case."""
    return term.strip().title()


def build_mapping(
    raw_terms: Iterable[str],
    *,
    matcher: CtcaeMatcher | None = None,
    rules: list | None = None,
    llm_tier: LLMTier | None = None,
    exclude: set[str] | None = None,
) -> list[MappingRow]:
    """Route each unique raw term through the tiers into mapping rows.

    Per raw term: split into compound parts, preprocess each, then for each part
    drop it (tier "filtered") if it is a non-symptom or in `exclude`, else try
    tier 1 (fuzzy vocabulary), tier 2 (rules), and finally collect the residue
    for a single batched tier-3 call. A compound raw term yields one row per part.
    """
    exclude_lower = {e.strip().lower() for e in (exclude or set())}
    rows: list[MappingRow] = []

    # Stage parts so tier 3 runs once over the whole residue, not per term.
    residual_terms: list[str] = []
    pending: list[tuple[str, str]] = []  # (raw_term, preprocessed) awaiting tier 3

    seen: set[str] = set()
    for raw in raw_terms:
        if raw in seen:
            continue
        seen.add(raw)

        if is_non_symptom(raw):
            rows.append(MappingRow(raw, "", FILTERED, "filtered"))
            continue

        parts = split_compound(raw)
        for part in parts:
            pre = preprocess_term(part)
            if not pre or is_non_symptom(pre) or pre.lower() in exclude_lower:
                rows.append(MappingRow(raw, pre, FILTERED, "filtered"))
                continue

            if matcher is not None:
                hit = matcher.match(pre)
                if hit:
                    rows.append(MappingRow(raw, pre, _canonical(hit), "ctcae"))
                    continue

            if rules:
                hit = apply_rules(pre, rules)
                if hit:
                    rows.append(MappingRow(raw, pre, _canonical(hit), "rule"))
                    continue

            pending.append((raw, pre))
            residual_terms.append(pre)

    if pending:
        tier = llm_tier or LLMTier(None)
        grouped = tier.group(sorted(set(residual_terms)))
        for raw, pre in pending:
            canonical = _canonical(grouped.get(pre, pre))
            rows.append(MappingRow(raw, pre, canonical, "llm"))

    return rows


def run_build(
    *,
    inputs: list[str],
    ctcae: str | None,
    rules: str | None,
    backend: str,
    model: str | None,
    out_mapping: str,
) -> None:
    """Read extraction inputs, build the mapping over their raw terms, save it."""
    raw_terms: list[str] = []
    seen: set[str] = set()
    for path in inputs:
        for row in read_jsonl(path):
            for term in ExtractionRecord.from_dict(row).symptoms:
                if isinstance(term, str) and term.strip() and term not in seen:
                    seen.add(term)
                    raw_terms.append(term)

    matcher = None
    if ctcae:
        matcher = CtcaeMatcher(load_preferred_terms(ctcae))

    compiled_rules = load_rules(rules)

    vocabulary = load_preferred_terms(ctcae) if ctcae else None
    client = get_client(backend, model) if model else None
    llm_tier = LLMTier(client, vocabulary=vocabulary)

    rows = build_mapping(
        raw_terms, matcher=matcher, rules=compiled_rules, llm_tier=llm_tier
    )
    save_mapping(out_mapping, rows)


def _preprocessed_lookup(mapping_path: str) -> dict[str, str]:
    """Build a preprocessed_term -> normalized_term view of the mapping CSV.

    Apply keys on the preprocessed part, since a compound raw term contributes
    several rows that each need to resolve independently.
    """
    lookup: dict[str, str] = {}
    with open(mapping_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pre = (row.get("preprocessed_term") or "").strip()
            norm = (row.get("normalized_term") or "").strip()
            if pre:
                lookup[pre] = norm
    return lookup


def run_apply(*, input: str, mapping: str, out: str) -> None:
    """Apply a built mapping to extraction records, writing normalized records.

    Each raw symptom is split and preprocessed the same way the mapping was
    built, looked up, and kept only if it resolves to a real (non-filtered)
    canonical term. Canonicals are deduplicated per record.
    """
    lookup = _preprocessed_lookup(mapping)

    def normalized_rows():
        for row in read_jsonl(input):
            record = ExtractionRecord.from_dict(row)
            canonicals: list[str] = []
            for raw in record.symptoms:
                if not isinstance(raw, str) or not raw.strip():
                    continue
                for part in split_compound(raw):
                    pre = preprocess_term(part)
                    norm = lookup.get(pre)
                    if norm and norm != FILTERED and norm not in canonicals:
                        canonicals.append(norm)
            yield NormalizedRecord(
                patient_id=record.patient_id,
                group=record.group,
                symptoms=canonicals,
                symptoms_raw=record.symptoms,
            ).to_dict()

    write_jsonl(out, normalized_rows())
