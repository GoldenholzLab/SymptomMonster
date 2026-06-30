"""Clinically-generic symptom synonym matching.

Free-text symptom terms vary ("drowsiness" vs "somnolence" vs "sleepiness")
without changing the underlying concept. ``EquivalenceMatcher`` folds each term
to a canonical representative so that downstream set comparisons count synonyms
as the same symptom. The default groups are deliberately generic textbook
synonyms; supply your own groups to match a specific coding vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable

# Generic symptom synonym sets. Each frozenset is one concept; the matcher treats
# all members as equivalent and picks a canonical name for them. These are common
# medical-English equivalences, intentionally free of any study, drug, or cohort
# specifics so the package carries no embedded data.
DEFAULT_GROUPS: list[frozenset[str]] = [
    frozenset({"somnolence", "drowsiness", "sleepiness", "drowsy", "sleepy", "sedation"}),
    frozenset({"dizziness", "lightheadedness", "lightheaded", "woozy", "vertigo"}),
    frozenset({"fatigue", "tiredness", "tired", "exhaustion", "lethargy"}),
    frozenset({"nausea", "nauseous", "nauseated", "queasiness"}),
    frozenset({"headache", "cephalgia", "head pain"}),
    frozenset({"insomnia", "sleeplessness", "difficulty sleeping", "trouble sleeping"}),
    frozenset({"confusion", "disorientation", "disoriented", "confused"}),
    frozenset({"irritability", "irritable", "agitation", "agitated"}),
    frozenset({"anxiety", "anxious", "nervousness", "nervous"}),
    frozenset({"depression", "depressed", "low mood", "depressed mood"}),
    frozenset({"rash", "skin rash", "exanthem"}),
    frozenset({"tremor", "tremors", "shaking", "shakiness"}),
    frozenset({"weakness", "asthenia", "muscle weakness"}),
    frozenset({"blurred vision", "blurry vision", "vision blurring"}),
    frozenset({"diarrhea", "diarrhoea", "loose stools"}),
    frozenset({"constipation", "infrequent stools"}),
    frozenset({"vomiting", "emesis", "throwing up"}),
    frozenset({"weight gain", "weight increase"}),
    frozenset({"weight loss", "weight decrease"}),
    frozenset({"appetite loss", "decreased appetite", "loss of appetite", "anorexia"}),
    frozenset({"memory impairment", "memory loss", "forgetfulness", "memory problems"}),
    frozenset({"paresthesia", "tingling", "numbness", "pins and needles"}),
    frozenset({"palpitations", "racing heart", "heart racing"}),
    frozenset({"dry mouth", "xerostomia"}),
    frozenset({"itching", "pruritus", "itchiness"}),
]


def _normalize(term: str) -> str:
    """Lowercase and collapse whitespace/underscores for matching."""
    return " ".join(term.lower().replace("_", " ").split())


class EquivalenceMatcher:
    """Maps symptom terms to a canonical representative within synonym groups."""

    def __init__(self, groups: Iterable[Iterable[str]] | None = None) -> None:
        source = DEFAULT_GROUPS if groups is None else groups
        # term -> canonical. The canonical for a group is its alphabetically first
        # member, so the choice is stable and independent of input ordering.
        self._canonical: dict[str, str] = {}
        for group in source:
            members = sorted(_normalize(t) for t in group if t and t.strip())
            if not members:
                continue
            representative = members[0]
            for member in members:
                self._canonical[member] = representative

    def canonical(self, term: str) -> str:
        """Return the group representative for ``term``, or the term itself if unknown.

        Unknown terms fall through normalized (lowercased, whitespace-collapsed) so
        that exact-but-uncatalogued matches still align.
        """
        key = _normalize(term)
        return self._canonical.get(key, key)

    def equivalent(self, a: str, b: str) -> bool:
        """True if ``a`` and ``b`` are the same term or share a synonym group."""
        return self.canonical(a) == self.canonical(b)
