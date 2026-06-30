"""Tests for text normalization in ``symptommonster.normalize.text``."""

from __future__ import annotations

from symptommonster.normalize.text import preprocess_term, split_compound

# --- preprocess_term ------------------------------------------------------


def test_lowercases_and_strips_leading_modifier():
    assert preprocess_term("Severe Headache") == "headache"


def test_strips_trailing_location_qualifier():
    assert preprocess_term("numbness in left hand") == "numbness"


def test_plain_term_is_unchanged():
    assert preprocess_term("nausea") == "nausea"


def test_is_idempotent_on_clean_terms():
    once = preprocess_term("worsening dizziness")
    assert preprocess_term(once) == once


def test_collapses_internal_whitespace():
    assert preprocess_term("  mild    fatigue  ") == "fatigue"


def test_empty_string_stays_empty():
    assert preprocess_term("") == ""


def test_never_reduces_a_term_to_nothing():
    # The whole term looks like a modifier; stripping it would leave nothing, so
    # the function must keep something meaningful rather than return "".
    assert preprocess_term("severe") != ""


# --- split_compound -------------------------------------------------------


def test_splits_on_slash():
    assert split_compound("nausea/vomiting") == ["nausea", "vomiting"]


def test_splits_on_and():
    assert split_compound("nausea and vomiting") == ["nausea", "vomiting"]


def test_single_term_is_one_element_list():
    assert split_compound("headache") == ["headache"]


def test_splits_on_comma():
    assert split_compound("headache, nausea, fatigue") == [
        "headache",
        "nausea",
        "fatigue",
    ]


def test_split_trims_whitespace_around_parts():
    parts = split_compound("dizziness /  tremor")
    assert parts == ["dizziness", "tremor"]
