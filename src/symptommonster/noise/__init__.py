"""Stage 3: build the scrambled-surrogate noise floor (the per-pair empirical null)."""

from .surrogate import build_surrogate_pairs, run_noise

__all__ = ["build_surrogate_pairs", "run_noise"]
