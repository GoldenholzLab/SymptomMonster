"""Model-comparison lollipop over the benchmark table.

A horizontal lollipop ranks the benchmarked models by a quality metric: a stem
from the axis to a dot at each model's score, sorted best to worst, with the
leading model drawn in the highlight colour and the rest in the comparator
colour. Consumes the benchmark table; carries no embedded scores.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import ASH, COMPARATOR, GRAPHITE, HIGHLIGHT, INK, grid

# Preference order of quality columns: the first one present is plotted. Higher
# is better for the F1 metrics; weighted MAE is error, so lower is better.
_QUALITY_COLUMNS = ("macro_f1", "symptomatic_f1")
_ERROR_COLUMNS = ("weighted_mae",)


def _pick_metric(benchmark: pd.DataFrame) -> tuple[str, bool]:
    """Return `(column, higher_is_better)` for the quality axis to plot."""
    for column in _QUALITY_COLUMNS:
        if column in benchmark.columns:
            return column, True
    for column in _ERROR_COLUMNS:
        if column in benchmark.columns:
            return column, False
    raise ValueError(
        "benchmark frame needs one of: "
        + ", ".join(_QUALITY_COLUMNS + _ERROR_COLUMNS)
    )


def render(benchmark: pd.DataFrame, *, fig: plt.Figure | None = None) -> plt.Figure:
    """Render the model-comparison lollipop from the benchmark table.

    `benchmark` follows the documented schema: `model` plus one or more of
    `macro_f1`, `symptomatic_f1`, `weighted_mae`, `mean_inference_s`, `bias`.
    Models are sorted by the chosen metric with the leader emphasised; an empty
    frame renders a single annotated blank panel.
    """
    if fig is None:
        fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)

    if benchmark is None or benchmark.empty or "model" not in benchmark.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "no benchmark data", ha="center", va="center", color=ASH)
        return fig

    metric, higher_is_better = _pick_metric(benchmark)
    # Sort so the best model lands at the top of the horizontal axis.
    ordered = benchmark.sort_values(metric, ascending=higher_is_better).reset_index(drop=True)
    models = list(ordered["model"].astype(str))
    scores = ordered[metric].astype(float).to_numpy()
    y = np.arange(len(models))

    # The leader is the last row under the sort (top of the plot).
    leader = len(models) - 1
    colors = [HIGHLIGHT if i == leader else COMPARATOR for i in range(len(models))]

    baseline = min(0.0, float(scores.min()))
    for yi, score, color in zip(y, scores, colors, strict=False):
        ax.plot([baseline, score], [yi, yi], color=ASH, linewidth=0.9, zorder=1)
        ax.scatter([score], [yi], s=46, facecolor=color, edgecolor=INK, linewidth=0.5, zorder=4)
        ax.text(
            score, yi + 0.18, f"{score:.3g}",
            ha="center", va="bottom", fontsize=6.6, color=GRAPHITE, zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_ylim(-0.6, len(models) - 0.4)
    pad = 0.08 * (float(scores.max()) - baseline or 1.0)
    ax.set_xlim(baseline, float(scores.max()) + pad)
    ax.set_xlabel(metric.replace("_", " "))
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", which="both", length=2.5)
    grid(ax, axis="x")

    fig.tight_layout()
    return fig
