"""Drug-name masking applied before any note reaches a model."""

from .masker import DrugLeakError, DrugMasker

__all__ = ["DrugMasker", "DrugLeakError"]
