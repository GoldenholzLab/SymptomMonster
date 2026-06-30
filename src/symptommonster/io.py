"""Input/output layer: JSONL helpers, on-disk record formats, and note sources.

Stages exchange JSONL (one JSON object per line, streamed, never loaded whole).
Extraction emits ExtractionRecord; normalization rewrites those into
NormalizedRecord. A NoteSource yields PatientNotes from wherever the notes live,
a directory tree or a JSONL export, without the rest of the pipeline knowing the
origin.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# --- JSONL helpers --------------------------------------------------------


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each line of a JSONL file as a dict, skipping blank lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows to a JSONL file, creating parent directories. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append a single row. Used for resumable runs that checkpoint as they go."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_done_ids(path: str | Path, key: str = "patient_id") -> set[str]:
    """Return the set of `key` values already present, so a run can resume."""
    path = Path(path)
    if not path.exists():
        return set()
    return {row[key] for row in read_jsonl(path) if key in row}


# --- Records exchanged between stages -------------------------------------


@dataclass
class ExtractionRecord:
    patient_id: str
    group: str | None = None
    symptoms: list[str] = field(default_factory=list)
    raw_response: str | None = None
    extraction_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None or k == "group"}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ExtractionRecord:
        return cls(
            patient_id=row["patient_id"],
            group=row.get("group"),
            symptoms=list(row.get("symptoms", [])),
            raw_response=row.get("raw_response"),
            extraction_time_s=row.get("extraction_time_s"),
        )


@dataclass
class NormalizedRecord:
    patient_id: str
    group: str | None = None
    symptoms: list[str] = field(default_factory=list)
    symptoms_raw: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "patient_id": self.patient_id,
            "group": self.group,
            "symptoms": sorted(set(self.symptoms)),
        }
        if self.symptoms_raw is not None:
            row["symptoms_raw"] = self.symptoms_raw
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> NormalizedRecord:
        return cls(
            patient_id=row["patient_id"],
            group=row.get("group"),
            symptoms=list(row.get("symptoms", [])),
            symptoms_raw=row.get("symptoms_raw"),
        )


# --- Note source contract -------------------------------------------------


@dataclass(frozen=True)
class PatientNotes:
    """One patient's notes, split into a pre-treatment and post-treatment window.

    `group` is the cohort label (typically the index treatment). It drives two
    things downstream: which alias the masker substitutes, and which pool the
    surrogate noise floor draws from. Leave it None when the cohort is implicit.
    """

    patient_id: str
    pre_notes: list[str] = field(default_factory=list)
    post_notes: list[str] = field(default_factory=list)
    group: str | None = None


@runtime_checkable
class NoteSource(Protocol):
    """Anything that can enumerate patients and look one up by id."""

    def __iter__(self) -> Iterator[PatientNotes]: ...

    def get(self, patient_id: str) -> PatientNotes | None: ...


# --- Reference note sources for local inputs ------------------------------


class DirectoryNoteSource:
    """Read notes from a directory laid out one folder per patient::

        root/<patient_id>/<pre>/*.txt      baseline (pre-treatment) notes
        root/<patient_id>/<post>/*.txt     follow-up (post-treatment) notes

    `group_map` optionally assigns each patient to a cohort label; without it
    every group is None. The subfolder names are configurable so you can point
    this at an existing layout without renaming anything.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        pre: str = "pre",
        post: str = "post",
        group_map: Mapping[str, str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.root = Path(root)
        self.pre = pre
        self.post = post
        self.group_map = dict(group_map or {})
        self.encoding = encoding

    def _read_window(self, directory: Path) -> list[str]:
        if not directory.is_dir():
            return []
        return [p.read_text(encoding=self.encoding, errors="ignore") for p in sorted(directory.glob("*.txt"))]

    def get(self, patient_id: str) -> PatientNotes | None:
        patient_dir = self.root / patient_id
        if not patient_dir.is_dir():
            return None
        return PatientNotes(
            patient_id=patient_id,
            pre_notes=self._read_window(patient_dir / self.pre),
            post_notes=self._read_window(patient_dir / self.post),
            group=self.group_map.get(patient_id),
        )

    def __iter__(self) -> Iterator[PatientNotes]:
        for patient_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            notes = self.get(patient_dir.name)
            if notes is not None:
                yield notes


class JsonlNoteSource:
    """Read notes from a JSONL file of ``{patient_id, pre_notes, post_notes, group}``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[PatientNotes]:
        for row in read_jsonl(self.path):
            yield PatientNotes(
                patient_id=row["patient_id"],
                pre_notes=list(row.get("pre_notes", [])),
                post_notes=list(row.get("post_notes", [])),
                group=row.get("group"),
            )

    def get(self, patient_id: str) -> PatientNotes | None:
        for notes in self:
            if notes.patient_id == patient_id:
                return notes
        return None


def open_note_source(path: str | Path, **kwargs) -> DirectoryNoteSource | JsonlNoteSource:
    """Pick a source by what `path` is: a directory tree or a ``.jsonl`` export."""
    path = Path(path)
    if path.is_dir():
        return DirectoryNoteSource(path, **kwargs)
    if path.suffix == ".jsonl":
        return JsonlNoteSource(path)
    raise ValueError(f"cannot infer a note source from {path!r}; expected a directory or .jsonl file")


__all__ = [
    "PatientNotes",
    "NoteSource",
    "DirectoryNoteSource",
    "JsonlNoteSource",
    "open_note_source",
    "ExtractionRecord",
    "NormalizedRecord",
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "read_done_ids",
]
