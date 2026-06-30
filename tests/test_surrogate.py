"""Tests for the surrogate noise floor in ``symptommonster.noise.surrogate``."""

from __future__ import annotations

from symptommonster.io import PatientNotes
from symptommonster.noise.surrogate import build_surrogate_pairs


def _tiny_cohort() -> list[PatientNotes]:
    """A small two-group cohort with group-tagged, all-distinct note text.

    Every note string names its group, so a surrogate note's origin group is
    visible from the text, which lets the within-group pooling property be
    checked directly. Window sizes vary per patient so the count-preservation
    property is non-trivial.
    """
    return [
        # Group A: three patients with varied window sizes.
        PatientNotes("a1", ["A-pre-1a", "A-pre-1b"], ["A-post-1"], group="A"),
        PatientNotes("a2", ["A-pre-2a"], ["A-post-2a", "A-post-2b"], group="A"),
        PatientNotes("a3", ["A-pre-3a", "A-pre-3b", "A-pre-3c"], ["A-post-3"], group="A"),
        # Group B: two patients.
        PatientNotes("b1", ["B-pre-1a", "B-pre-1b"], ["B-post-1a", "B-post-1b"], group="B"),
        PatientNotes("b2", ["B-pre-2a"], ["B-post-2"], group="B"),
    ]


def test_counts_are_preserved_per_patient():
    cohort = _tiny_cohort()
    surrogates = build_surrogate_pairs(cohort, seed=0)

    by_id = {p.patient_id: p for p in surrogates}
    assert set(by_id) == {p.patient_id for p in cohort}
    for original in cohort:
        produced = by_id[original.patient_id]
        assert len(produced.pre_notes) == len(original.pre_notes)
        assert len(produced.post_notes) == len(original.post_notes)
        assert produced.group == original.group


def test_pooling_stays_within_group():
    cohort = _tiny_cohort()
    surrogates = build_surrogate_pairs(cohort, seed=0)

    # The full set of pre-notes available to each group's pool.
    group_pre_pool = {
        "A": {note for p in cohort if p.group == "A" for note in p.pre_notes},
        "B": {note for p in cohort if p.group == "B" for note in p.pre_notes},
    }

    for produced in surrogates:
        own_pool = group_pre_pool[produced.group]
        other_pool = group_pre_pool["B" if produced.group == "A" else "A"]
        # Both windows are drawn from this group's pre-note pool only.
        for note in produced.pre_notes + produced.post_notes:
            assert note in own_pool
            assert note not in other_pool


def test_post_window_is_drawn_from_pretreatment_pool():
    # The whole point of the surrogate is that the "post" window carries no
    # treatment signal: it is sampled from pre-treatment text.
    cohort = _tiny_cohort()
    surrogates = build_surrogate_pairs(cohort, seed=0)

    all_pre = {note for p in cohort for note in p.pre_notes}
    real_post = {note for p in cohort for note in p.post_notes}
    for produced in surrogates:
        for note in produced.post_notes:
            assert note in all_pre
            assert note not in real_post


def test_same_seed_is_deterministic():
    cohort = _tiny_cohort()
    first = build_surrogate_pairs(cohort, seed=42)
    second = build_surrogate_pairs(cohort, seed=42)

    assert [(p.patient_id, p.pre_notes, p.post_notes) for p in first] == [
        (p.patient_id, p.pre_notes, p.post_notes) for p in second
    ]


def test_different_seeds_can_differ():
    # Not a hard guarantee for every seed, but with a pool this size two seeds
    # should produce at least one different draw.
    cohort = _tiny_cohort()
    a = build_surrogate_pairs(cohort, seed=1)
    b = build_surrogate_pairs(cohort, seed=2)
    a_notes = [(p.patient_id, p.pre_notes, p.post_notes) for p in a]
    b_notes = [(p.patient_id, p.pre_notes, p.post_notes) for p in b]
    assert a_notes != b_notes


def test_empty_cohort_returns_empty():
    assert build_surrogate_pairs([], seed=0) == []
