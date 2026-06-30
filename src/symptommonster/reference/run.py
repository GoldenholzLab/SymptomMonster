"""Join published reference rates to the observed signal summary.

Each group has a JSON dump of reference adverse-event rates, a list of
`{"term": ..., "rate": ...}` entries scraped from the literature. We parse each
rate string to a midpoint percentage, match the term to a symptom in the signal
summary, and emit one comparison row per (group, symptom) so the observed signal
rate can be plotted against the published reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .parse import rate_midpoint


def _load_dump(path: Path) -> dict[str, float]:
    """Parse one per-group dump into a {normalized_term: midpoint_pct} map.

    Entries whose rate string carries no numeric content are dropped. When a
    term repeats, the first parseable rate wins, since dumps are ordered most- to
    least-specific by convention.
    """
    entries = json.loads(path.read_text())
    rates: dict[str, float] = {}
    for entry in entries:
        term = str(entry.get("term", "")).strip()
        if not term:
            continue
        midpoint = rate_midpoint(str(entry.get("rate", "")))
        if midpoint is None:
            continue
        rates.setdefault(term.lower(), midpoint)
    return rates


def run_reference(*, dumps: str, signal: str, out: str) -> None:
    """Build `reference_comparison.csv` from per-group reference dumps.

    `dumps` is a directory of `<group>.json` files; `signal` is the signal
    summary CSV (with `group`, `symptom`, `signal_pct` columns). Reference
    terms are matched to summary symptoms case-insensitively. The output has
    columns: group, symptom, observed_signal_pct, reference_rate, source.
    """
    dumps_dir = Path(dumps)
    summary = pd.read_csv(signal)

    rates_by_group: dict[str, dict[str, float]] = {}
    for path in sorted(dumps_dir.glob("*.json")):
        rates_by_group[path.stem] = _load_dump(path)

    rows: list[dict[str, object]] = []
    for record in summary.to_dict("records"):
        group = str(record.get("group", ""))
        symptom = str(record.get("symptom", ""))
        reference = rates_by_group.get(group, {}).get(symptom.lower())
        if reference is None:
            continue
        rows.append(
            {
                "group": group,
                "symptom": symptom,
                "observed_signal_pct": float(record.get("signal_pct", 0.0)),
                "reference_rate": reference,
                "source": group,
            }
        )

    columns = ["group", "symptom", "observed_signal_pct", "reference_rate", "source"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
