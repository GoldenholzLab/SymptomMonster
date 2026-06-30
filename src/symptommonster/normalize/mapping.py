"""The reviewable mapping artifact: one CSV row per raw term.

The CSV is the human checkpoint between building a mapping and applying it. A
reviewer can audit or hand-edit any row, then the apply step reads it straight
back. Columns: raw_term, preprocessed_term, normalized_term, tier.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_FIELDS = ("raw_term", "preprocessed_term", "normalized_term", "tier")

# Sentinel written for terms judged not to be symptoms; dropped on apply.
FILTERED = "[FILTERED]"


@dataclass
class MappingRow:
    raw_term: str
    preprocessed_term: str
    normalized_term: str
    tier: str  # one of: ctcae, rule, llm, filtered


def save_mapping(path: str, rows: list[MappingRow]) -> None:
    """Write rows to the mapping CSV, sorted for a stable, reviewable diff."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r.normalized_term, r.raw_term))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "raw_term": row.raw_term,
                    "preprocessed_term": row.preprocessed_term,
                    "normalized_term": row.normalized_term,
                    "tier": row.tier,
                }
            )


def load_mapping(path: str) -> dict[str, str]:
    """Read the mapping CSV back into a raw_term -> normalized_term lookup."""
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("raw_term") or "").strip()
            if raw:
                mapping[raw] = (row.get("normalized_term") or "").strip()
    return mapping
