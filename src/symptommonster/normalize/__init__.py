"""Stage 2: normalize raw symptom terms to a canonical vocabulary in three tiers.

Tier 1 fuzzy-matches the caller's preferred-term list, tier 2 applies packaged
regex synonym rules, and tier 3 asks a model to group whatever remains. The two
phases, building a reviewable mapping and then applying it, are exposed here.
"""

from .mapping import MappingRow, load_mapping, save_mapping
from .pipeline import build_mapping, run_apply, run_build
from .text import is_non_symptom, preprocess_term, split_compound
from .tiers import CtcaeMatcher, LLMTier, apply_rules, load_preferred_terms, load_rules

__all__ = [
    "preprocess_term",
    "split_compound",
    "is_non_symptom",
    "CtcaeMatcher",
    "load_preferred_terms",
    "load_rules",
    "apply_rules",
    "LLMTier",
    "MappingRow",
    "save_mapping",
    "load_mapping",
    "build_mapping",
    "run_build",
    "run_apply",
]
