"""Tests for drug-name masking in ``symptommonster.masking``."""

from __future__ import annotations

import pytest

from symptommonster.masking import DrugLeakError, DrugMasker


def test_mask_substitutes_on_word_boundary():
    masker = DrugMasker({"Drugzol": "alpha"})
    out = masker.mask("Patient started Drugzol last week.")
    # The source name is gone and the replacement token has taken its place.
    assert "Drugzol" not in out
    assert "alpha" in out.lower()


def test_mask_preserves_capitalization():
    masker = DrugMasker({"drugzol": "druga"})
    # Title case in, title case out; all-caps in, all-caps out.
    assert masker.mask("Drugzol was prescribed.") == "Druga was prescribed."
    assert masker.mask("DRUGZOL was prescribed.") == "DRUGA was prescribed."


def test_mask_is_case_insensitive_on_input():
    masker = DrugMasker({"Drugzol": "DrugA"})
    lowered = masker.mask("started drugzol today")
    assert "drugzol" not in lowered.lower()
    assert "druga" in lowered.lower()


def test_mask_does_not_touch_partial_words():
    # "Drugzoless" merely contains the name as a substring; word boundaries mean
    # it is left alone.
    masker = DrugMasker({"Drugzol": "DrugA"})
    out = masker.mask("the Drugzoless regimen")
    assert "Drugzoless" in out


def test_mask_handles_multiple_distinct_names():
    masker = DrugMasker({"Drugzol": "alpha", "Tokenax": "beta"})
    out = masker.mask("Drugzol and Tokenax together")
    assert "alpha" in out.lower() and "beta" in out.lower()
    assert "Drugzol" not in out and "Tokenax" not in out


def test_mask_prefers_longest_name_first():
    # A multi-word name must be consumed whole rather than split by its prefix.
    masker = DrugMasker({"Drugzol": "alpha", "Drugzol Acid": "beta"})
    out = masker.mask("Drugzol Acid was given")
    # The longer alias wins, so we see its token and not the shorter one's.
    assert "beta" in out.lower()
    assert "alpha" not in out.lower()
    assert "Drugzol" not in out


def test_assert_no_leak_passes_on_clean_text():
    masker = DrugMasker({"Drugzol": "DrugA"})
    # No exception means the masked text is clean.
    masker.assert_no_leak("DrugA was prescribed and tolerated.")


def test_assert_no_leak_raises_on_residual_name():
    masker = DrugMasker({"Drugzol": "DrugA"})
    with pytest.raises(DrugLeakError):
        masker.assert_no_leak("Drugzol slipped through unmasked.")


def test_assert_no_leak_checks_explicit_names_argument():
    masker = DrugMasker({"Drugzol": "DrugA"})
    # A name not in the alias map can still be checked when passed explicitly.
    with pytest.raises(DrugLeakError):
        masker.assert_no_leak("Tokenax remains here", names=["Tokenax"])


def test_mask_then_assert_no_leak_round_trip():
    masker = DrugMasker({"Drugzol": "DrugA", "Tokenax": "DrugB"})
    masked = masker.mask("Drugzol and Tokenax were both started.")
    # Whatever the substitution did, the source names must be gone.
    masker.assert_no_leak(masked)
