"""Parse published adverse-event rate strings into numeric ranges.

Reference sources report rates in free-form text: bare percentages ("5%"),
explicit ranges ("10% to 25%", "1-10%"), and open-ended bounds ("<1%", ">10%").
These helpers normalize that text into a `(low, high)` pair and a single
midpoint so observed signal rates can be plotted against the literature. Pure
string handling with no IO, so the parsing rules stay trivial to unit test.
"""

from __future__ import annotations

import re

# A percentage token: digits with an optional decimal, before an optional '%'.
_PCT = r"(\d+(?:\.\d+)?)"
# "10% to 25%" or "10 - 25%": two percentages joined by "to", "-", or an en/em dash.
_RANGE_RE = re.compile(_PCT + r"\s*%?\s*(?:to|[-\u2013\u2014])\s*" + _PCT + r"\s*%")
# A single percentage anywhere in the string.
_SINGLE_RE = re.compile(_PCT + r"\s*%")
# Open-ended lower bound: <=, <, or the unicode less-than-or-equal glyph.
_LE_RE = re.compile(r"(?:<=|<|\u2264|\u2266)\s*" + _PCT + r"\s*%")
# Open-ended upper bound: >=, >, or the unicode greater-than-or-equal glyph.
_GE_RE = re.compile(r"(?:>=|>|\u2265|\u2267)\s*" + _PCT + r"\s*%")


def parse_rate_range(text: str) -> tuple[float | None, float | None]:
    """Return the (low, high) percentage bounds described by `text`.

    Resolution order, first match wins: an explicit two-sided range maps to
    both bounds; "<= N%" (and "< N%") is an upper bound only `(None, N)`;
    ">= N%" (and "> N%") is a lower bound only `(N, None)`; a lone "N%" maps to
    `(N, N)`. Empty, "n/a", or non-numeric input yields `(None, None)`.
    """
    if not isinstance(text, str):
        return (None, None)
    s = text.strip()
    if not s or s.lower() in {"n/a", "na", "none", "nd"}:
        return (None, None)

    m = _RANGE_RE.search(s)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    # Open-ended bounds are checked before the bare-percentage fallback so the
    # comparator ("<", ">") is not swallowed by the single-value match.
    m = _LE_RE.search(s)
    if m:
        return (None, float(m.group(1)))
    m = _GE_RE.search(s)
    if m:
        return (float(m.group(1)), None)

    m = _SINGLE_RE.search(s)
    if m:
        value = float(m.group(1))
        return (value, value)

    return (None, None)


def rate_midpoint(text: str) -> float | None:
    """Return a single representative percentage for `text`, or None.

    Both bounds present -> their mean; an upper bound only -> half of it (the
    midpoint of the implied 0..high interval); a lower bound only -> the bound
    itself; neither -> None.
    """
    low, high = parse_rate_range(text)
    if low is not None and high is not None:
        return (low + high) / 2.0
    if high is not None:
        return high / 2.0
    if low is not None:
        return low
    return None
