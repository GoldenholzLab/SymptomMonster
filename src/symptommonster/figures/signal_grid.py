"""Multi-group signal-minus-noise grid.

One small panel per group, each a bar chart of signal-minus-noise (percentage
points) across symptoms. Significant increases are filled in the highlight
colour, significant decreases in the comparator colour, and everything else in
neutral grey, so the real findings read at a glance. Panels share a common
symptom order; groups are paired into rows by similar dynamic range, and each
row shares a y-range whose pixel height tracks its span, so a given deflection
renders at a consistent height across the figure. Consumes the signal summary
table; carries no embedded values.
"""

from __future__ import annotations

import math
import string

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import ASH, COMPARATOR, GRAPHITE, HIGHLIGHT, INK, MIST, grid


def _present(summary: pd.DataFrame, column: str, default: object) -> pd.Series:
    """Return `summary[column]` if present, else a default-filled column."""
    if column in summary.columns:
        return summary[column]
    return pd.Series([default] * len(summary), index=summary.index)


def _ordered_symptoms(summary: pd.DataFrame) -> list[str]:
    """Order symptoms by significant-count across groups, then mean delta.

    The most broadly observed symptoms come first; rare group-specific ones
    trail. Ties on the significant count break by descending mean delta.
    """
    sig = _present(summary, "significant", False).astype(bool)
    delta = _present(summary, "signal_minus_noise", 0.0).astype(float)
    work = pd.DataFrame(
        {"symptom": summary["symptom"], "sig": sig, "delta": delta}
    )
    agg = work.groupby("symptom").agg(sig=("sig", "sum"), delta=("delta", "mean"))
    agg = agg.sort_values(["sig", "delta"], ascending=[False, False])
    return list(agg.index)


def _panel_frame(summary: pd.DataFrame, group: str, symptoms: list[str]) -> pd.DataFrame:
    """Build a per-group, per-symptom frame aligned to `symptoms`.

    Missing (group, symptom) pairs are zero-filled so every panel has the same
    bar positions. Significance and direction default to non-significant when the
    column is absent or the row is missing.
    """
    sub = summary[summary["group"] == group]
    sub = sub.set_index("symptom")
    deltas, sig, sig_dec, ci_lo, ci_hi = [], [], [], [], []
    for symptom in symptoms:
        if symptom in sub.index:
            row = sub.loc[symptom]
            if isinstance(row, pd.DataFrame):  # duplicate symptom rows: take first
                row = row.iloc[0]
            delta = float(row.get("signal_minus_noise", 0.0))
            is_sig = bool(row.get("significant", False))
            direction = str(row.get("direction", "increased")).lower()
            deltas.append(delta)
            sig.append(is_sig and direction != "decreased")
            sig_dec.append(is_sig and direction == "decreased")
            ci_lo.append(float(row.get("diff_ci_lower", delta)))
            ci_hi.append(float(row.get("diff_ci_upper", delta)))
        else:
            deltas.append(0.0)
            sig.append(False)
            sig_dec.append(False)
            ci_lo.append(0.0)
            ci_hi.append(0.0)
    return pd.DataFrame(
        {
            "symptom": symptoms,
            "delta": deltas,
            "sig": sig,
            "sig_dec": sig_dec,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    )


def _row_pairs(groups: list[str], frames: list[pd.DataFrame]) -> list[list[int]]:
    """Pair groups into rows by similar dynamic range.

    Each group's reach is its largest absolute deflection. Sorting by that reach
    and pairing adjacent ranks puts two similarly scaled groups side by side, so
    a shared row y-range stays informative for both. An odd group out forms a
    single-panel row of its own. Within a pair the panels are ordered by label so
    the left panel is the alphabetically earlier group.
    """
    reach = [(i, float(np.abs(f["delta"].to_numpy()).max()) if len(f) else 0.0) for i, f in enumerate(frames)]
    order = [i for i, _ in sorted(reach, key=lambda item: item[1], reverse=True)]
    pairs: list[list[int]] = []
    for k in range(0, len(order), 2):
        members = order[k : k + 2]
        members.sort(key=lambda i: str(groups[i]))
        pairs.append(members)
    return pairs


def _pair_yrange(frames: list[pd.DataFrame], members: list[int], *, pad: float = 0.5) -> tuple[float, float]:
    """Shared (ymin, ymax) for a row, covering both panels' CI envelopes and zero."""
    lo = min(min(float(frames[i]["ci_lo"].min()), float(frames[i]["delta"].min()), 0.0) for i in members)
    hi = max(max(float(frames[i]["ci_hi"].max()), float(frames[i]["delta"].max()), 0.0) for i in members)
    return math.floor((lo - pad) * 2) / 2, math.ceil((hi + pad) * 2) / 2


def _minor_ticks(ax: plt.Axes, *, step_minor: float = 0.5) -> None:
    """Major ticks at every integer, half-height minor ticks between them."""
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step_minor))
    ax.tick_params(axis="y", which="major", length=2.5, width=0.8, color=INK)
    ax.tick_params(axis="y", which="minor", length=1.3, width=0.6, color=INK)


def _draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    *,
    ymin: float,
    ymax: float,
    show_xlabels: bool,
) -> None:
    """Draw one group's signal-minus-noise bars with a zero baseline."""
    x = np.arange(len(frame))
    delta = frame["delta"].to_numpy()
    ci_lo = frame["ci_lo"].to_numpy()
    ci_hi = frame["ci_hi"].to_numpy()
    sig = frame["sig"].to_numpy(dtype=bool)
    sig_dec = frame["sig_dec"].to_numpy(dtype=bool)

    err_lo = np.maximum(delta - ci_lo, 0)
    err_hi = np.maximum(ci_hi - delta, 0)

    # Three-way encoding: significant increase (highlight), significant decrease
    # (comparator), otherwise neutral grey. A pair is significant in at most one
    # direction, so increase takes precedence on any tie.
    face = np.where(sig, HIGHLIGHT, np.where(sig_dec, COMPARATOR, MIST))
    edge = np.where(sig, HIGHLIGHT, np.where(sig_dec, COMPARATOR, ASH))
    ax.bar(
        x, delta, width=0.74, color=face, edgecolor=edge, linewidth=0.7,
        yerr=[err_lo, err_hi], ecolor=ASH,
        error_kw={"elinewidth": 0.5, "capthick": 0.5}, capsize=0, zorder=3,
    )
    ax.axhline(0, color=GRAPHITE, linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(
            list(frame["symptom"]), rotation=30, ha="right", rotation_mode="anchor"
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(frame) - 0.4)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="x", which="both", length=0, pad=2)
    _minor_ticks(ax, step_minor=0.5)
    grid(ax, axis="y")
    ax.set_title(title, loc="left", pad=4)


def _panel_label(index: int) -> str:
    """Reading-order panel letter (A, B, ...), falling back to a number past Z."""
    return string.ascii_uppercase[index] if index < 26 else str(index + 1)


def render(summary: pd.DataFrame, *, fig: plt.Figure | None = None) -> plt.Figure:
    """Render the per-group signal-minus-noise grid from the signal summary.

    `summary` follows the documented schema: `group`, `symptom`,
    `signal_minus_noise`, `significant`, `direction` (optional `diff_ci_lower` /
    `diff_ci_upper` add whiskers). Groups are paired into rows by largest
    deflection and laid out two per row; an empty frame renders a single
    annotated blank panel.
    """
    if fig is None:
        fig = plt.figure(figsize=(7.2, 9.4))

    if summary is None or summary.empty or "group" not in summary.columns:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "no signal data", ha="center", va="center", color=ASH)
        return fig

    symptoms = _ordered_symptoms(summary)
    groups = sorted(set(summary["group"].astype(str)))
    frames = [_panel_frame(summary, g, symptoms) for g in groups]

    pairs = _row_pairs(groups, frames)
    yranges = [_pair_yrange(frames, members) for members in pairs]
    nrows = len(pairs)
    ncols = 2 if any(len(members) == 2 for members in pairs) else 1
    height_ratios = [hi - lo for (lo, hi) in yranges]

    # Every panel sets the same x-limits explicitly, so the columns stay aligned
    # without sharex, which lets the bottom panel of an odd final column keep its
    # symptom labels instead of being treated as an inner row.
    axes = fig.subplots(nrows, ncols, gridspec_kw={"height_ratios": height_ratios})
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    last_row_size = len(pairs[-1])
    label = 0
    for r, members in enumerate(pairs):
        ymin, ymax = yranges[r]
        for c in range(ncols):
            if c < len(members):
                i = members[c]
                # Show x labels on the bottom row, and on a panel whose own row is
                # the bottom one actually filled in that column.
                below_empty = r == nrows - 2 and c >= last_row_size
                _draw_panel(
                    axes[r, c], frames[i], f"{_panel_label(label)}.  {groups[i]}",
                    ymin=ymin, ymax=ymax, show_xlabels=(r == nrows - 1) or below_empty,
                )
                label += 1
            else:
                axes[r, c].axis("off")

    handles = [
        mpatches.Patch(facecolor=HIGHLIGHT, edgecolor=HIGHLIGHT, label="Significant increase"),
        mpatches.Patch(facecolor=COMPARATOR, edgecolor=COMPARATOR, label="Significant decrease"),
        mpatches.Patch(facecolor=MIST, edgecolor=ASH, label="Not significant"),
    ]
    fig.supylabel("Signal - Noise (%)", fontsize=8, x=0.015)
    fig.tight_layout(rect=(0.025, 0.0, 1.0, 0.955))
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
        ncol=3, frameon=False, fontsize=7, handlelength=1.0,
        columnspacing=1.1, handletextpad=0.4,
    )
    return fig
