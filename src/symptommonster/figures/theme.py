"""Shared visual theme for every figure: palette, typography, output policy.

One module owns the look so all figures render with the same colours, fonts, and
geometry. The palette carries fixed semantic roles: `HIGHLIGHT` for emphasised
or significant results, `COMPARATOR` for the baseline/control series they are
read against, and a neutral grey ramp for everything structural. Figures import
these names rather than hard-coding hex so the theme can change in one place.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Neutral grey ramp, dark to light: axis ink down to faint fills and page tints.
INK = "#1a1a1a"
GRAPHITE = "#3a3a3a"
SLATE = "#6a6a6a"
ASH = "#9a9a9a"
MIST = "#d4d4d4"
PAPER = "#f4f4f4"
WHITE = "#ffffff"

# Highlight family: significant findings, the primary/production series.
HIGHLIGHT = "#9E2A2B"
HIGHLIGHT_LIGHT = "#C56262"
HIGHLIGHT_FAINT = "#F1DCDC"

# Comparator family: the baseline, control, or candidate series read against
# the highlight. Splits the semantic load so highlight stays reserved for the
# result the eye should land on.
COMPARATOR = "#5B7B9A"
COMPARATOR_LIGHT = "#9DB3C6"
COMPARATOR_FAINT = "#DCE4EC"

# Muted, colour-blind-aware swatches for categorical annotation (for example,
# group class markers on the dendrogram). Never used to encode a quantity.
SWATCH_PALETTE = [
    "#5B7B9A",
    "#B58B4C",
    "#6F8E6A",
    "#8E5A7C",
    "#C8742C",
    "#4F6F7C",
    "#A0866B",
    "#7A6E8E",
]


def set_style() -> None:
    """Apply the shared rcParams. Call once before rendering any figure.

    A two-size type ramp keeps the hierarchy obvious: 10pt bold for titles, 8pt
    regular for everything else. Fonts list metric-compatible substitutes so the
    output looks the same on machines without the primary face installed.
    """
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue", "Helvetica", "Arial",
                "Liberation Sans", "Arimo", "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )


def grid(ax: plt.Axes, axis: str = "y") -> None:
    """Apply the standard hairline gridline to one axis, drawn behind the data."""
    ax.grid(axis=axis, alpha=0.30, linewidth=0.5, color=ASH)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, basename: str, out_dir: str | Path, *, dpi: int = 1200) -> None:
    """Write `fig` as both a print-resolution PNG and a vector PDF in `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{basename}.png", dpi=dpi)
    fig.savefig(out / f"{basename}.pdf")


def bootstrap_ci(
    count: int,
    total: int,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percent-scale bootstrap CI `(lower, upper)` for a binomial rate.

    Resamples the binomial count at the observed rate and reads the central
    `confidence` interval off the percentiles. An empty sample has no rate to
    bound, so it returns `(0.0, 0.0)`.
    """
    if total == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    p = count / total
    samples = rng.binomial(total, p, size=n_bootstrap) / total * 100.0
    alpha = 1.0 - confidence
    return (
        float(np.percentile(samples, alpha / 2 * 100)),
        float(np.percentile(samples, (1 - alpha / 2) * 100)),
    )
