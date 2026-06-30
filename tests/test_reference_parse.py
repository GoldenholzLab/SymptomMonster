"""Tests for published-rate parsing in ``symptommonster.reference.parse``."""

from __future__ import annotations

import pytest

from symptommonster.reference.parse import parse_rate_range, rate_midpoint

# --- parse_rate_range -----------------------------------------------------


def test_parse_explicit_to_range():
    assert parse_rate_range("10% to 25%") == (10, 25)


def test_parse_hyphen_range():
    assert parse_rate_range("1-10%") == (1, 10)


def test_parse_upper_bounded_only():
    # "at most 1%", so no lower bound is implied.
    assert parse_rate_range("<=1%") == (None, 1)


def test_parse_lower_bounded_only():
    # "more than 10%", so no upper bound is implied.
    assert parse_rate_range(">10%") == (10, None)


def test_parse_single_value_is_a_point_range():
    assert parse_rate_range("5%") == (5, 5)


def test_parse_empty_string_is_unbounded():
    assert parse_rate_range("") == (None, None)


# --- rate_midpoint --------------------------------------------------------


def test_midpoint_of_a_range():
    assert rate_midpoint("10% to 20%") == pytest.approx(15)


def test_midpoint_of_upper_bounded_only():
    # With only an upper bound of 1, the representative rate is half of it.
    assert rate_midpoint("<=1%") == pytest.approx(0.5)


def test_midpoint_of_single_value():
    assert rate_midpoint("5%") == pytest.approx(5)


def test_midpoint_of_empty_string_is_none():
    assert rate_midpoint("") is None


def test_midpoint_lies_within_parsed_range():
    low, high = parse_rate_range("2-8%")
    mid = rate_midpoint("2-8%")
    assert low <= mid <= high
