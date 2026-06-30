"""Scrambled-surrogate noise floor: a per-pair empirical null.

The signal run measures how often a symptom appears for treated patients. But
some apparent rate is just the base rate of clinicians writing that word, not a
treatment effect. To separate the two we build a surrogate cohort whose notes
carry no on-treatment signal by construction, run the *same* extraction over it,
and use the surrogate rate as the null each real rate is tested against.

The construction, per group: pool every patient's PRE-treatment notes, then give
each surrogate patient both a "pre" and a "post" window drawn from that pre-pool
(preferentially from other patients). Because both windows are pre-treatment
text, no drug exposure separates them, so any pre/post difference the extractor
reports is noise. Counts are preserved per patient so the surrogate cohort has
the same shape as the real one, which is what makes the paired test valid.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from symptommonster.io import PatientNotes, open_note_source, write_jsonl


def _draw(
    rng: np.random.Generator,
    pool: list[str],
    own: set[int],
    count: int,
) -> list[str]:
    """Draw `count` notes from `pool`, preferring indices not in `own`.

    Sampling is without replacement from the eligible (other-patient) notes when
    there are enough of them; otherwise we fall back to sampling the whole pool
    with replacement so the requested count is always met. Returning the drawn
    text (not indices) keeps the surrogate self-contained.
    """
    if count <= 0 or not pool:
        return []
    eligible = [i for i in range(len(pool)) if i not in own]
    if len(eligible) >= count:
        chosen = rng.choice(eligible, size=count, replace=False)
    else:
        # Not enough foreign notes to fill the window: sample the full pool with
        # replacement. This only bites for tiny groups, where some self-overlap
        # is unavoidable and harmless (the null is still treatment-free).
        chosen = rng.choice(len(pool), size=count, replace=True)
    return [pool[int(i)] for i in chosen]


def build_surrogate_pairs(patients: Sequence[PatientNotes], seed: int = 0) -> list[PatientNotes]:
    """Build a treatment-free surrogate cohort, deterministic under `seed`.

    For each group independently we pool all pre-treatment notes, then resample a
    pre and a post window for every patient from that pool (preferring other
    patients' notes), preserving each patient's original window sizes. The
    returned patients keep their ids and group so the surrogate run lines up
    one-to-one with the signal run.
    """
    rng = np.random.default_rng(seed)

    # Group patients while remembering each one's place in the per-group pool, so
    # we can exclude a patient's own pre-notes when drawing their surrogate.
    by_group: dict[str | None, list[PatientNotes]] = defaultdict(list)
    for patient in patients:
        by_group[patient.group].append(patient)

    # Iterate groups in a stable order so the seed fully determines the output.
    surrogates: list[PatientNotes] = []
    for group in sorted(by_group, key=lambda g: (g is None, g)):
        members = by_group[group]

        pool: list[str] = []
        owned: list[set[int]] = []  # pool indices contributed by each member
        for member in members:
            start = len(pool)
            pool.extend(member.pre_notes)
            owned.append(set(range(start, len(pool))))

        for member, own in zip(members, owned, strict=True):
            surrogates.append(
                PatientNotes(
                    patient_id=member.patient_id,
                    pre_notes=_draw(rng, pool, own, len(member.pre_notes)),
                    post_notes=_draw(rng, pool, own, len(member.post_notes)),
                    group=group,
                )
            )

    return surrogates


def run_noise(*, notes: str, seed: int = 0, out: str) -> None:
    """Build the surrogate cohort from `notes` and write it as a notes JSONL.

    The output has the same shape as any note source, so it feeds straight back
    through `extract` to produce the noise-floor symptom rates.
    """
    patients = list(open_note_source(notes))
    surrogates = build_surrogate_pairs(patients, seed=seed)
    write_jsonl(
        out,
        (
            {
                "patient_id": s.patient_id,
                "group": s.group,
                "pre_notes": s.pre_notes,
                "post_notes": s.post_notes,
            }
            for s in surrogates
        ),
    )
