"""Replace drug names with fictitious tokens so the model cannot read identity.

This module applies a caller-supplied substitution map and checks that nothing
leaked; it holds no drug names of its own. The two-layer scheme the method relies
on lives in that alias map, which sends the index drug and each co-medication to
distinct tokens, so the model can neither recognize the drug under study nor infer
its class from the company it keeps.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


class DrugLeakError(RuntimeError):
    """Raised when a source drug name survives masking."""


def _match_case(source: str, token: str) -> str:
    """Render `token` with the capitalization pattern of the matched `source`."""
    if source.isupper():
        return token.upper()
    if source.istitle():
        return token[:1].upper() + token[1:]
    return token


class DrugMasker:
    """Substitute names from an alias map, case-insensitively and case-preserving.

    `alias_map` maps each real name to its replacement token. Matching is on word
    boundaries and longest-name-first, so "valproic acid" is handled before
    "valproic" and partial words are never touched.
    """

    def __init__(self, alias_map: Mapping[str, str]) -> None:
        self._aliases = dict(alias_map)
        self._lookup = {name.lower(): token for name, token in self._aliases.items()}
        names = sorted(self._aliases, key=len, reverse=True)
        if names:
            alternation = "|".join(re.escape(n) for n in names)
            self._pattern: re.Pattern[str] | None = re.compile(rf"\b({alternation})\b", re.IGNORECASE)
        else:
            self._pattern = None

    def mask(self, text: str) -> str:
        if not text or self._pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            return _match_case(match.group(0), self._lookup[match.group(0).lower()])

        return self._pattern.sub(replace, text)

    def assert_no_leak(self, text: str, names: Iterable[str] | None = None) -> None:
        """Raise `DrugLeakError` if any source name still appears in `text`.

        The error is deliberately vague: it confirms a leak without echoing the
        name that leaked.
        """
        for name in names if names is not None else self._aliases:
            if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                raise DrugLeakError("masked text still contains a source drug name")
