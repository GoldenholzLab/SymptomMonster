"""Stage 1: mask drug names, then extract symptom mentions with a local model."""

from .extractor import SymptomExtractor, load_template, render
from .runner import run_extract

__all__ = ["SymptomExtractor", "load_template", "render", "run_extract"]
