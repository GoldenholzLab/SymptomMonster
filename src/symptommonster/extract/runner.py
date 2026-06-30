"""Drive extraction over a note source and stream results to a JSONL file.

The run is resumable: records are appended one patient at a time, and `resume`
skips ids already present in the output. A `seed` fixes a deterministic patient
order so a `limit`ed run is reproducible.
"""

from __future__ import annotations

import json
import random
import sys

from symptommonster.io import (
    PatientNotes,
    append_jsonl,
    open_note_source,
    read_done_ids,
)
from symptommonster.llm import get_client

from .extractor import SymptomExtractor, load_template


def _load_alias_map(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"alias map at {path!r} must be a JSON object of name -> token")
    return {str(name): str(token) for name, token in data.items()}


def _select(
    patients: list[PatientNotes],
    *,
    seed: int,
    limit: int | None,
    done: set[str],
) -> list[PatientNotes]:
    if seed:
        random.Random(seed).shuffle(patients)
    if limit is not None:
        patients = patients[:limit]
    return [p for p in patients if p.patient_id not in done]


def run_extract(
    *,
    notes: str,
    alias_map: str,
    prompt: str | None,
    backend: str,
    model: str,
    out: str,
    limit: int | None = None,
    seed: int = 0,
    temperature: float = 0.05,
    num_ctx: int = 131072,
    resume: bool = False,
) -> None:
    aliases = _load_alias_map(alias_map)
    source = open_note_source(notes)
    template = load_template(prompt)
    # num_ctx pins the Ollama context window; the MLX backend uses the model's own.
    client_kwargs = {"options": {"num_ctx": num_ctx}} if backend == "ollama" else {}
    client = get_client(backend, model, **client_kwargs)
    extractor = SymptomExtractor(client, aliases, template, temperature=temperature)

    done = read_done_ids(out) if resume else set()
    # `seed` is applied before `limit`, so the limited subset is a stable prefix
    # of the shuffled order regardless of how much of it has already been done.
    selected = _select(list(source), seed=seed, limit=limit, done=done)

    total = len(selected)
    for index, patient in enumerate(selected, start=1):
        try:
            record = extractor.extract(patient)
        except Exception as error:  # noqa: BLE001 - one bad patient must not end the run
            print(f"[{index}/{total}] {patient.patient_id} failed: {error}", file=sys.stderr)
            continue
        append_jsonl(out, record.to_dict())
        seconds = record.extraction_time_s or 0.0
        print(
            f"[{index}/{total}] {patient.patient_id}: "
            f"{len(record.symptoms)} symptoms in {seconds:.1f}s",
            file=sys.stderr,
        )
