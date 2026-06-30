"""Reusable drawing primitives shared across figures.

A rounded flow box and a shrink-aware connector, factored out so the schematic
figure stays declarative. Each takes an Axes and draws in data coordinates; none
touches files or global state.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .theme import ASH, GRAPHITE, INK, WHITE


def flow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: str,
    *,
    edge_color: str = INK,
    title_color: str = INK,
    detail_color: str = GRAPHITE,
    stage_label: str | None = None,
) -> tuple[float, float]:
    """Draw a rounded stage box with an optional corner label, title, and detail.

    Returns `(x_left, x_right)` for arrow attachment. An empty `detail` centres
    the lone title vertically rather than leaving it floating high in the box.
    """
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.06",
        linewidth=0.9, edgecolor=edge_color, facecolor=WHITE, zorder=2,
    )
    ax.add_patch(box)
    if stage_label:
        ax.text(
            x + 0.12, y + h - 0.12, stage_label,
            ha="left", va="top",
            fontsize=6.2, color=ASH, family="sans-serif", zorder=3,
        )
    title_y = y + h * (0.50 if not detail else 0.60)
    ax.text(
        x + w / 2, title_y, title,
        ha="center", va="center",
        fontsize=8.6, fontweight="bold", color=title_color,
        linespacing=1.10, zorder=3,
    )
    if detail:
        ax.text(
            x + w / 2, y + h * 0.22, detail,
            ha="center", va="center",
            fontsize=6.9, color=detail_color, linespacing=1.22, zorder=3,
        )
    return x, x + w


def connect(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = INK,
    linestyle: str = "-",
    lw: float = 0.7,
    head: bool = True,
    shrink_a: float = 0.0,
    shrink_b: float = 0.0,
) -> None:
    """Draw a connector between two points, optionally arrow-headed.

    `shrink_a`/`shrink_b` hold the endpoints just shy of box borders so the line
    stops at the rounded corner instead of overshooting it.
    """
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>" if head else "-",
            color=color, linewidth=lw, linestyle=linestyle,
            mutation_scale=7, shrinkA=shrink_a, shrinkB=shrink_b,
            zorder=1, capstyle="round", joinstyle="round",
        )
    )
