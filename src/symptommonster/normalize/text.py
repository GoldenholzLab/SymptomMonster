"""Deterministic text cleanup and non-symptom filtering, applied before any tier.

Pure string functions: lowercasing, modifier and qualifier stripping, compound
splitting, and detection of terms that are not symptoms at all. Kept free of
clinical knowledge so the same handling holds whatever vocabulary the caller
supplies downstream.
"""

from __future__ import annotations

import re

# Generic intensity/temporal modifiers that prefix a symptom without changing
# its identity. "severe headache" and "headache" are the same concept here.
_LEADING_MODIFIERS = (
    "severe",
    "mild",
    "moderate",
    "worsening",
    "worsened",
    "chronic",
    "acute",
    "intermittent",
    "occasional",
    "new",
    "recurrent",
    "persistent",
    "transient",
    "slight",
    "increased",
    "decreased",
)

_LEADING_MODIFIER_RE = re.compile(
    r"^(?:" + "|".join(map(re.escape, _LEADING_MODIFIERS)) + r")\s+"
)

# Trailing "in/of/... [the] [laterality] <body part>" qualifiers. The symptom is
# the head noun; the location rarely distinguishes the canonical term.
_LOCATION_RE = re.compile(
    r"\s+(?:in|of|on|at|over|around)"
    r"(?:\s+(?:the|his|her|my))?\s+"
    r"(?:(?:left|right|bilateral|both|upper|lower)\s+)*"
    r"\w+(?:\s+(?:and|&)\s+\w+)?$",
    re.IGNORECASE,
)

# Bare trailing laterality with no preposition, for example "weakness left".
_TRAILING_LATERALITY_RE = re.compile(r"\s+(?:left|right|bilateral)$", re.IGNORECASE)

# Trailing generic nouns that carry no information of their own.
_TRAILING_GENERIC_RE = re.compile(r"\s+(?:symptoms?|sensations?)$", re.IGNORECASE)

# Compound separators: slash, conjunctions, comma, ampersand, semicolon.
_SPLIT_RE = re.compile(r"\s*(?:/|,|;|&|\band\b)\s*", re.IGNORECASE)


def preprocess_term(term: str) -> str:
    """Reduce a raw term to a bare canonical-ish form (single term, no split).

    Lowercases, collapses whitespace, then peels a leading modifier and trailing
    location/laterality/generic qualifiers. Idempotent on already-clean terms.
    """
    s = " ".join(term.lower().split())
    if not s:
        return ""

    s = _LEADING_MODIFIER_RE.sub("", s).strip()

    # Strip trailing qualifiers, but only while something meaningful remains so
    # we never reduce a term to nothing.
    for pattern in (_LOCATION_RE, _TRAILING_LATERALITY_RE, _TRAILING_GENERIC_RE):
        stripped = pattern.sub("", s).strip()
        if stripped:
            s = stripped

    return s


def split_compound(term: str) -> list[str]:
    """Split a compound term into its parts on common separators.

    "nausea/vomiting" and "nausea and vomiting" both become two terms; a single
    term comes back as a one-element list.
    """
    return [part.strip() for part in _SPLIT_RE.split(term) if part.strip()]


# Explicit negations / absence statements.
_NEGATION_RE = re.compile(
    r"^(?:none|no\s+(?:symptoms?|side\s*effects?|complaints?|change|issues?)"
    r"|n/?a|not\s+applicable|denies?|negative|nil|unremarkable|wnl)\b",
    re.IGNORECASE,
)

# Hedging / non-attribution preambles that are statements, not symptoms.
_META_PREAMBLE_RE = re.compile(
    r"^(?:no\s+(?:mention|clear|prior|documented|evidence)"
    r"|not\s+(?:attributable|definitively|clearly|reported)"
    r"|since\s+starting|caused\s+by|attributable|due\s+to)\b",
    re.IGNORECASE,
)

# Bookkeeping fields a model may echo from its own output schema.
_META_WORD_RE = re.compile(
    r"^(?:rationale|attribution|confidence|reasoning|note|unknown|other|n\.?o\.?s\.?)$",
    re.IGNORECASE,
)

_DATE_RE = re.compile(r"^\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?$")
_NUMERIC_RE = re.compile(r"^[\d.,%\s]+$")

# Two characters or fewer is almost always a fragment, not a clinical term.
_FRAGMENT_RE = re.compile(r"^\W*\w{1,2}\W*$")

# Pathologically long strings are sentences the model failed to distil.
_MAX_LEN = 150


def is_non_symptom(term: str) -> bool:
    """True when `term` is not a real symptom and should be dropped.

    Catches empties, negations, hedges, bare dates/numbers, schema bookkeeping
    words, and tiny fragments. Genuine symptoms fall through to the tiers.
    """
    s = term.strip()
    if not s or len(s) > _MAX_LEN:
        return True
    return bool(
        _NEGATION_RE.match(s)
        or _META_PREAMBLE_RE.match(s)
        or _META_WORD_RE.match(s)
        or _DATE_RE.match(s)
        or _NUMERIC_RE.match(s)
        or _FRAGMENT_RE.match(s)
    )
