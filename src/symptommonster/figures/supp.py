"""Supplement figures.

Three small figures that support the main results, each rendered from a
documented-format table and carrying no embedded values:

- ``render_time_vs_efficacy`` - benchmark inference time against error, framed
  as a deployment-decision quadrant.
- ``render_noise_vs_literature`` - observed signal rates against published
  reference rates, exposing the documentation gap.
- ``render_dendrogram`` - hierarchical clustering of groups by symptom profile.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage

from .theme import ASH, COMPARATOR, GRAPHITE, HIGHLIGHT, INK, SLATE, grid


def _blank(fig: plt.Figure, message: str) -> plt.Figure:
    """Render a single annotated empty panel when the input is unusable."""
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=ASH)
    return fig


# --------------------------------------------------------------------------- #
# Time vs efficacy                                                            #
# --------------------------------------------------------------------------- #

# Columns the figure needs. `reasoning` and `production` are optional refinements.
_BENCHMARK_REQUIRED = ("model", "mean_inference_s", "weighted_mae")


def _pick_production(
    times: np.ndarray,
    errors: np.ndarray,
    *,
    time_threshold: float,
    error_threshold: float,
) -> int:
    """Index of the production model: fastest point clearing both thresholds.

    Falls back to the fastest point at/under the error ceiling, then to the
    fastest point overall, so a frame with no model inside the quadrant still
    resolves to a sensible emphasis.
    """
    in_quadrant = np.flatnonzero((times <= time_threshold) & (errors <= error_threshold))
    if in_quadrant.size:
        return int(in_quadrant[np.argmin(times[in_quadrant])])
    under_error = np.flatnonzero(errors <= error_threshold)
    if under_error.size:
        return int(under_error[np.argmin(times[under_error])])
    return int(np.argmin(times))


def render_time_vs_efficacy(
    benchmark: pd.DataFrame,
    *,
    fig: plt.Figure | None = None,
    time_threshold: float | None = None,
    error_threshold: float | None = None,
) -> plt.Figure:
    """Render the time-vs-error quadrant scatter from the benchmark table.

    `benchmark` follows the documented schema: `model`, `mean_inference_s`,
    `weighted_mae`, plus optional `reasoning` / `production` flags. The two
    thresholds default to the medians so the lower-left "clears both bars"
    quadrant frames the points; pass explicit values to fix them to a study's
    operating points. Marker shape splits reasoning from non-reasoning models so
    colour stays free for the production-vs-candidate distinction. An empty or
    malformed frame renders a single annotated blank panel.
    """
    if fig is None:
        fig = plt.figure(figsize=(6.9, 4.2))

    if benchmark is None or benchmark.empty or not set(_BENCHMARK_REQUIRED).issubset(benchmark.columns):
        return _blank(fig, "no benchmark data")

    models = list(benchmark["model"].astype(str))
    times = benchmark["mean_inference_s"].astype(float).to_numpy()
    errors = benchmark["weighted_mae"].astype(float).to_numpy()
    if "reasoning" in benchmark.columns:
        reasoning = benchmark["reasoning"].fillna(False).astype(bool).to_numpy()
    else:
        reasoning = np.zeros(len(models), dtype=bool)

    # Pad the axes outward so points and labels never pin to an edge.
    x_span = float(times.max()) - float(times.min()) or 1.0
    y_span = float(errors.max()) - float(errors.min()) or 1.0
    x_lo = float(times.min()) - 0.10 * x_span
    x_hi = float(times.max()) + 0.20 * x_span
    y_lo = max(0.0, float(errors.min()) - 0.10 * y_span)
    y_hi = float(errors.max()) + 0.10 * y_span

    if time_threshold is None:
        time_threshold = float(np.median(times))
    if error_threshold is None:
        error_threshold = float(np.median(errors))

    ax = fig.add_subplot(111)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Faint wash on the operational quadrant (fast enough AND accurate enough).
    ax.axvspan(
        x_lo, time_threshold,
        ymin=0.0, ymax=(error_threshold - y_lo) / (y_hi - y_lo),
        facecolor=HIGHLIGHT, alpha=0.06, zorder=0,
    )
    ax.axhline(error_threshold, color=ASH, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(time_threshold, color=ASH, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        time_threshold - 0.01 * (x_hi - x_lo), y_lo + 0.03 * (y_hi - y_lo),
        f"<= {time_threshold:g} s", ha="right", va="bottom", color=SLATE, zorder=1,
    )
    ax.text(
        x_lo + 0.015 * (x_hi - x_lo), error_threshold - 0.01 * (y_hi - y_lo),
        f"<= {error_threshold:g}", ha="left", va="top", color=SLATE, zorder=1,
    )

    if "production" in benchmark.columns and benchmark["production"].fillna(False).astype(bool).any():
        prod_idx = int(np.flatnonzero(benchmark["production"].fillna(False).astype(bool).to_numpy())[0])
    else:
        prod_idx = _pick_production(
            times, errors, time_threshold=time_threshold, error_threshold=error_threshold,
        )

    for i, (model, x, y, is_reasoning) in enumerate(zip(models, times, errors, reasoning, strict=False)):
        is_prod = i == prod_idx
        ax.scatter(
            [x], [y],
            marker="D" if is_reasoning else "o",
            s=95 if is_prod else 55,
            facecolor=HIGHLIGHT if is_prod else COMPARATOR,
            edgecolor=INK, linewidth=0.9 if is_prod else 0.6, zorder=4,
        )
        ax.annotate(
            model, xy=(x, y), xytext=(10, 6), textcoords="offset points",
            color=HIGHLIGHT if is_prod else INK,
            fontweight="bold" if is_prod else "normal", zorder=5,
        )

    # Callout naming why the highlighted point is the deployment choice.
    ax.annotate(
        "production: fastest model\nclearing both thresholds",
        xy=(times[prod_idx], errors[prod_idx]),
        xytext=(times[prod_idx] + 0.22 * (x_hi - x_lo), errors[prod_idx] - 0.18 * (y_hi - y_lo)),
        textcoords="data", color=HIGHLIGHT, ha="left", va="top",
        arrowprops=dict(
            arrowstyle="-", color=HIGHLIGHT, linewidth=0.7,
            shrinkA=4, shrinkB=4, connectionstyle="arc3,rad=-0.15",
        ),
        zorder=6,
    )

    ax.set_xlabel("mean inference time / item  (s)")
    ax.set_ylabel("weighted MAE  (percentage points)")
    grid(ax, axis="both")

    # Self-contained legend: colour = production vs candidate; shape = reasoning
    # vs non-reasoning. Drawn in the empty upper-right region.
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=HIGHLIGHT,
               markeredgecolor=INK, markersize=8, label="Production model"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COMPARATOR,
               markeredgecolor=INK, markersize=7, label="Candidate model"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
               markeredgecolor=SLATE, markersize=7, label="Reasoning model"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=SLATE, markersize=7, label="Non-reasoning model"),
    ]
    ax.legend(
        handles=handles, loc="upper right", frameon=False,
        fontsize=7.5, handletextpad=0.4, labelspacing=0.5, borderaxespad=0.6,
    )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Observed signal rate vs published reference rate                            #
# --------------------------------------------------------------------------- #


def render_noise_vs_literature(comparison: pd.DataFrame, *, fig: plt.Figure | None = None) -> plt.Figure:
    """Render the observed-vs-reference scatter from the comparison table.

    `comparison` follows the documented schema: `group`, `symptom`,
    `observed_signal_pct`, `reference_rate`, `source` (an optional `significant`
    column splits the marker colour). The identity line and an origin OLS fit
    (observed = alpha * reference) make systematic under-reporting visible, and
    an inset histograms the observed/reference ratio. Pairs with a non-positive
    reference rate are dropped; an empty result renders a blank panel.
    """
    if fig is None:
        fig = plt.figure(figsize=(7.0, 4.6))

    needed = {"observed_signal_pct", "reference_rate"}
    if comparison is None or comparison.empty or not needed.issubset(comparison.columns):
        return _blank(fig, "no reference comparison")

    pairs = comparison[comparison["reference_rate"].fillna(0) > 0].copy()
    if pairs.empty:
        return _blank(fig, "no reference comparison")

    ref = pairs["reference_rate"].to_numpy(dtype=float)
    obs = pairs["observed_signal_pct"].to_numpy(dtype=float)
    ratios = obs / ref

    if "significant" in pairs.columns:
        sig_mask = pairs["significant"].fillna(False).astype(bool).to_numpy()
    else:
        sig_mask = np.zeros(len(pairs), dtype=bool)

    # OLS through the origin: observed = alpha * reference.
    denom = float(np.dot(ref, ref))
    alpha = float(np.dot(ref, obs) / denom) if denom > 0 else 0.0

    ax = fig.add_axes((0.10, 0.13, 0.86, 0.82))
    lim = max(float(ref.max()), float(obs.max())) * 1.08 or 1.0

    ax.fill_between([0, lim], [0, 0], [0, lim], color=ASH, alpha=0.05, zorder=0)
    ax.plot([0, lim], [0, lim], color=ASH, linestyle="--", linewidth=0.9, zorder=1,
            label="y = x  (perfect agreement)")
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, alpha * xs, color=HIGHLIGHT, linewidth=1.2, zorder=2,
            label=f"OLS  observed = {alpha:.2f} * reference")

    ax.scatter(
        ref[~sig_mask], obs[~sig_mask],
        s=28, facecolor=COMPARATOR, edgecolor="white", linewidth=0.5,
        alpha=0.8, zorder=3, label="not significant",
    )
    if sig_mask.any():
        ax.scatter(
            ref[sig_mask], obs[sig_mask],
            s=36, facecolor=HIGHLIGHT, edgecolor=INK, linewidth=0.6,
            alpha=0.95, zorder=4, label="significant",
        )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, float(obs.max()) * 1.45 or 1.0)
    ax.set_xlabel("published reference rate  (%)")
    ax.set_ylabel("observed signal rate  (%)")
    grid(ax, axis="both")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5,
              handletextpad=0.5, labelspacing=0.4, borderaxespad=0.4)

    # Inset: distribution of observed/reference ratios with a median marker.
    inset = fig.add_axes((0.55, 0.55, 0.36, 0.32))
    top = max(1.2, float(np.percentile(ratios, 95))) if len(ratios) else 1.2
    inset.hist(ratios, bins=np.linspace(0, top, 18),
               color=COMPARATOR, edgecolor=INK, linewidth=0.5, alpha=0.85)
    inset.axvline(1.0, color=ASH, linestyle="--", linewidth=0.7)
    median_ratio = float(np.median(ratios))
    inset.axvline(median_ratio, color=HIGHLIGHT, linestyle="-", linewidth=1.0)
    inset.text(median_ratio + 0.03, inset.get_ylim()[1] * 0.92,
               f"median  {median_ratio:.2f}", color=HIGHLIGHT, ha="left", va="top")
    inset.set_xlabel("observed / reference")
    inset.set_ylabel("pairs")
    inset.tick_params(axis="both", length=2)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    grid(inset, axis="y")
    inset.set_title(f"median = {median_ratio * 100:.0f}% of reference",
                    color=GRAPHITE, pad=4, loc="left", fontsize=7.5)

    return fig


# --------------------------------------------------------------------------- #
# Mechanism clustering dendrogram                                             #
# --------------------------------------------------------------------------- #


def _profile_matrix(matrix: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return a non-negative profile matrix and its group labels.

    The first column is the group label; the rest are per-symptom values.
    Negative deltas are clipped to zero so cosine similarity runs over a
    non-negative adverse-signal profile rather than mixing signed directions.
    """
    labels = matrix.iloc[:, 0].astype(str).tolist()
    values = matrix.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return np.clip(values.to_numpy(dtype=float), 0.0, None), labels


def _cosine_distance(profiles: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance `1 - similarity` over row-normalised profiles."""
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = profiles / norms
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    return 1.0 - sim


def render_dendrogram(matrix: pd.DataFrame, *, fig: plt.Figure | None = None) -> plt.Figure:
    """Render the group clustering dendrogram from the group-by-symptom matrix.

    `matrix` has the group label in its first column and one numeric column per
    symptom (signal-minus-noise). Groups with similar symptom profiles join low
    in the tree. Fewer than two groups cannot be clustered, so that case renders
    a single annotated blank panel.
    """
    if fig is None:
        fig = plt.figure(figsize=(5.0, 8.4))
    ax = fig.add_subplot(111)

    if matrix is None or matrix.empty or len(matrix) < 2:
        ax.axis("off")
        ax.text(0.5, 0.5, "need >= 2 groups to cluster", ha="center", va="center", color=ASH)
        return fig

    profiles, labels = _profile_matrix(matrix)
    dist = _cosine_distance(profiles)
    condensed = dist[np.triu_indices_from(dist, k=1)]
    linkage_matrix = linkage(condensed, method="average")

    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="right",
        color_threshold=0,
        above_threshold_color=INK,
        ax=ax,
        leaf_font_size=8,
    )
    ax.set_xlabel("cosine distance  (1 - similarity)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0, pad=6)
    for label in ax.get_yticklabels():
        label.set_color(INK)

    fig.tight_layout()
    return fig
