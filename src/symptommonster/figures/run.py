"""Entrypoints that load documented-format tables and render figures and tables.

`run_figures` dispatches on a figure name (or "all"), loads the CSV each figure
needs, renders it, and saves PNG + PDF via the theme. A figure whose input path
was not supplied is skipped with a note on stderr rather than failing the batch.
`run_tables` builds the manuscript tables and writes them as CSV.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import (
    benchmark_lollipop,
    pipeline_schematic,
    signal_grid,
    supp,
    tables,
    theme,
)

# Each figure names the input it consumes ("signal", "benchmark", "reference",
# "matrix", or None for the data-free schematic) and a builder that turns the
# loaded frame into a Figure. "pipeline" ignores its input entirely.
_FIGURES: dict[str, tuple[str | None, Callable[..., plt.Figure]]] = {
    "pipeline": (None, lambda _df: pipeline_schematic.render()),
    "signal_grid": ("signal", lambda df: signal_grid.render(df)),
    "benchmark_lollipop": ("benchmark", lambda df: benchmark_lollipop.render(df)),
    "time_vs_efficacy": ("benchmark", lambda df: supp.render_time_vs_efficacy(df)),
    "noise_vs_literature": ("reference", lambda df: supp.render_noise_vs_literature(df)),
    "dendrogram": ("matrix", lambda df: supp.render_dendrogram(df)),
}


def run_figures(
    *,
    which: str,
    signal: str | None,
    benchmark: str | None,
    reference: str | None,
    matrix: str | None,
    out_dir: str,
) -> None:
    """Render `which` ("all" or a single figure name) into `out_dir`.

    The relevant input CSV is loaded with pandas and passed to the matching
    renderer; the result is saved as PNG + PDF under the figure's name. A figure
    whose input was not provided is skipped with a stderr note.
    """
    inputs = {"signal": signal, "benchmark": benchmark, "reference": reference, "matrix": matrix}
    theme.set_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if which == "all":
        names = list(_FIGURES)
    elif which in _FIGURES:
        names = [which]
    else:
        raise ValueError(
            f"unknown figure {which!r}; choose 'all' or one of: " + ", ".join(_FIGURES)
        )

    for name in names:
        source, build = _FIGURES[name]
        frame: pd.DataFrame | None = None
        if source is not None:
            path = inputs[source]
            if not path:
                print(f"skipping {name}: no --{source} input provided", file=sys.stderr)
                continue
            frame = pd.read_csv(path)
        fig = build(frame)
        theme.save(fig, name, out)
        plt.close(fig)


def run_tables(*, signal: str, demographics: str | None, out_dir: str) -> None:
    """Build Table 1 (if demographics given) and Table 2; write them as CSV.

    Table 2 is built from the signal summary; Table 1 from the demographics file
    when supplied. Each is written under `out_dir` as `table1.csv` / `table2.csv`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if demographics:
        table1 = tables.build_table1(pd.read_csv(demographics))
        table1.to_csv(out / "table1.csv", index=False)
    else:
        print("skipping table1: no --demographics input provided", file=sys.stderr)

    table2 = tables.build_table2(pd.read_csv(signal))
    table2.to_csv(out / "table2.csv", index=False)
