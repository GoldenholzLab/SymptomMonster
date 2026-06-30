"""Tiers 1 to 3 of normalization: fuzzy vocabulary, regex rules, model grouping.

Tier 1 (CtcaeMatcher) matches a term against the caller's preferred-term list,
tier 2 (load_rules / apply_rules) applies the packaged regex synonym rules, and
tier 3 (LLMTier) asks a model to group whatever residue remains.
"""

from __future__ import annotations

import json
import re
import tomllib
from difflib import get_close_matches
from importlib.resources import files

from ..llm import LLMClient, LLMRequest, extract_json

# --- Tier 1: fuzzy match against the caller's preferred-term list ----------


def load_preferred_terms(path: str) -> list[str]:
    """Read a preferred-term vocabulary from JSON.

    Accepts either a list of strings or a list of objects carrying a "term" key;
    any other shape yields an empty list rather than raising.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        return []
    terms: list[str] = []
    for item in data:
        if isinstance(item, str):
            terms.append(item.strip())
        elif isinstance(item, dict):
            value = item.get("term")
            if isinstance(value, str):
                terms.append(value.strip())
    return [t for t in terms if t]


class CtcaeMatcher:
    """Fuzzy lookup from a free-text term to a canonical preferred term.

    The preferred-term vocabulary (for example, a CTCAE export) is always a user
    input and is never shipped with the package. Matching is fuzzy so minor
    spelling and spacing variants still land on the canonical term.
    """

    def __init__(self, preferred_terms: list[str], *, cutoff: float = 0.9) -> None:
        self._cutoff = cutoff
        # Lowercased key -> original-cased canonical, for case-insensitive match
        # while preserving the vocabulary's own capitalization on output.
        self._by_lower: dict[str, str] = {}
        for term in preferred_terms:
            key = term.strip().lower()
            if key:
                self._by_lower.setdefault(key, term.strip())

    def match(self, term: str) -> str | None:
        """Return the closest preferred term within the cutoff, or None.

        An empty vocabulary always returns None, so the pipeline degrades to the
        later tiers instead of failing.
        """
        if not self._by_lower:
            return None
        key = term.strip().lower()
        if not key:
            return None
        if key in self._by_lower:
            return self._by_lower[key]
        hit = get_close_matches(key, self._by_lower.keys(), n=1, cutoff=self._cutoff)
        return self._by_lower[hit[0]] if hit else None


# --- Tier 2: regex synonym rules ------------------------------------------


def load_rules(path: str | None = None) -> list[tuple[re.Pattern[str], str]]:
    """Compile the synonym rules into (pattern, canonical) pairs, in file order.

    With no `path`, loads the packaged ``synonym_rules.toml``. Each rule's
    patterns are OR-joined into one case-insensitive regex.
    """
    if path is not None:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    else:
        resource = files("symptommonster.resources.rules") / "synonym_rules.toml"
        data = tomllib.loads(resource.read_text(encoding="utf-8"))

    compiled: list[tuple[re.Pattern[str], str]] = []
    for rule in data.get("rule", []):
        canonical = rule["canonical"]
        patterns = rule.get("patterns", [])
        if not patterns:
            continue
        combined = "|".join(f"(?:{p})" for p in patterns)
        compiled.append((re.compile(combined, re.IGNORECASE), canonical))
    return compiled


def apply_rules(term: str, rules: list[tuple[re.Pattern[str], str]]) -> str | None:
    """Return the canonical term for the first matching rule, else None."""
    for pattern, canonical in rules:
        if pattern.search(term):
            return canonical
    return None


# --- Tier 3: model grouping of the residual long tail ---------------------


def _load_prompt() -> str:
    resource = files("symptommonster.resources.prompts") / "normalize_tier3.txt"
    return resource.read_text(encoding="utf-8")


class LLMTier:
    """Batched model grouping of residual raw terms to canonical terms.

    Only the long tail of colloquial phrasings reaches this tier. When no client
    is configured the tier is an identity map, so the whole pipeline still runs
    (without the model-driven grouping) on a machine with no local model.
    """

    def __init__(
        self,
        client: LLMClient | None,
        *,
        vocabulary: list[str] | None = None,
        batch_size: int = 40,
    ) -> None:
        self._client = client
        self._vocabulary = vocabulary or []
        self._batch_size = batch_size

    def group(self, terms: list[str]) -> dict[str, str]:
        """Map each residual term to a canonical term.

        Without a client this is the identity map, so residuals pass through
        unchanged rather than being dropped.
        """
        if self._client is None:
            return {t: t for t in terms}

        template = _load_prompt()
        vocab_blob = json.dumps(sorted(self._vocabulary)) if self._vocabulary else "[]"

        mapping: dict[str, str] = {}
        for start in range(0, len(terms), self._batch_size):
            batch = terms[start : start + self._batch_size]
            prompt = template.replace("{vocabulary}", vocab_blob).replace(
                "{terms}", json.dumps(batch)
            )
            raw = self._client.generate(LLMRequest(prompt=prompt))
            mapping.update(self._parse(raw, batch))

        # Anything the model declined to return falls back to itself.
        for term in terms:
            mapping.setdefault(term, term)
        return mapping

    @staticmethod
    def _parse(response: str, batch: list[str]) -> dict[str, str]:
        """Pull the {raw: canonical} object out of a model response, tolerantly."""
        data = extract_json(response)
        if isinstance(data, dict) and isinstance(data.get("mapping"), dict):
            data = data["mapping"]
        if not isinstance(data, dict):
            return {}
        return {
            str(raw): str(canonical).strip()
            for raw, canonical in data.items()
            if isinstance(canonical, str) and canonical.strip()
        }
