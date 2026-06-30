"""SymptomMonster: pharmacovigilance signal detection from clinical notes.

The package is organized as one subpackage per pipeline stage (extract,
normalize, noise, stats, subgroup, benchmark, reference, figures). Stages
communicate through documented file formats rather than shared state, so each
runs independently from the command line. See ``docs/PIPELINE.md``.
"""

__version__ = "0.1.0"
