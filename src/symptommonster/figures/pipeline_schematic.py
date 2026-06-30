"""Method schematic: the parallel signal and noise-floor pipeline.

A structural, data-free diagram of the pipeline. A shared corpus is masked, then
splits into two lanes (the real pre/post pairs and a scrambled-surrogate control)
that run the identical extraction and standardization before rejoining at the
statistical analysis. The surrogate lane is drawn in the comparator colour
so the highlight stays free for any downstream result callout. Carries no counts,
names, or values; only the stage structure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .primitives import connect, flow_box
from .theme import COMPARATOR, INK


def render(fig: plt.Figure | None = None) -> plt.Figure:
    """Render the pipeline schematic onto `fig` (created if not supplied)."""
    if fig is None:
        fig = plt.figure(figsize=(9.0, 3.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    box_w, box_h = 1.72, 1.05
    lane_gap = 1.95
    y_mid = 2.30
    y_real = y_mid + lane_gap / 2
    y_noise = y_mid - lane_gap / 2

    gap_seq = 0.60
    gap_branch = 0.95

    # Shared start: corpus, then masking.
    corpus_x = 0.30
    flow_box(
        ax, corpus_x, y_mid - box_h / 2, box_w, box_h,
        "Clinical note\ncorpus", "paired pre / post",
        stage_label="01",
    )
    mask_x = corpus_x + box_w + gap_seq
    flow_box(
        ax, mask_x, y_mid - box_h / 2, box_w, box_h,
        "Drug masking", "names replaced\nby a token",
        stage_label="02",
    )

    # Real extraction lane (top).
    real_extract_x = mask_x + box_w + gap_branch
    flow_box(
        ax, real_extract_x, y_real - box_h / 2, box_w, box_h,
        "Extraction", "on real\npre / post",
        stage_label="03",
    )
    real_norm_x = real_extract_x + box_w + gap_seq
    flow_box(
        ax, real_norm_x, y_real - box_h / 2, box_w, box_h,
        "Standardization", "ontology / rules\n/ model",
        stage_label="04",
    )

    # Noise lane (bottom, comparator-outlined). Stage numbers are omitted: the
    # lane label below already marks these as the parallel control variant.
    noise_extract_x = real_extract_x
    flow_box(
        ax, noise_extract_x, y_noise - box_h / 2, box_w, box_h,
        "Same extraction", "on scrambled\nsurrogate",
        edge_color=COMPARATOR, title_color=COMPARATOR,
    )
    noise_norm_x = real_norm_x
    flow_box(
        ax, noise_norm_x, y_noise - box_h / 2, box_w, box_h,
        "Same\nstandardization", "identical pipeline,\nshuffled inputs",
        edge_color=COMPARATOR, title_color=COMPARATOR,
    )

    # Shared end: statistics (terminus). The detail line is intentionally blank;
    # the box's role is named by its title alone.
    stats_x = real_norm_x + box_w + gap_branch
    flow_box(
        ax, stats_x, y_mid - box_h / 2, box_w, box_h,
        "Statistical\nanalysis", "",
        stage_label="05",
    )

    # Connectors. `shrink_b` keeps each arrowhead just outside its target box.
    pad = 2.0

    connect(ax, corpus_x + box_w, y_mid, mask_x, y_mid, shrink_b=pad)

    branch_x = mask_x + box_w + 0.36
    connect(ax, mask_x + box_w, y_mid, branch_x, y_mid, head=False)
    connect(ax, branch_x, y_mid, branch_x, y_real, head=False)
    connect(ax, branch_x, y_real, real_extract_x, y_real, shrink_b=pad)
    connect(ax, branch_x, y_mid, branch_x, y_noise, head=False, color=COMPARATOR)
    connect(ax, branch_x, y_noise, noise_extract_x, y_noise, color=COMPARATOR, shrink_b=pad)

    connect(ax, real_extract_x + box_w, y_real, real_norm_x, y_real, shrink_b=pad)
    connect(
        ax, noise_extract_x + box_w, y_noise, noise_norm_x, y_noise,
        color=COMPARATOR, shrink_b=pad,
    )

    rejoin_x = real_norm_x + box_w + 0.36
    connect(ax, real_norm_x + box_w, y_real, rejoin_x, y_real, head=False)
    connect(ax, rejoin_x, y_real, rejoin_x, y_mid, head=False)
    connect(ax, noise_norm_x + box_w, y_noise, rejoin_x, y_noise, head=False, color=COMPARATOR)
    connect(ax, rejoin_x, y_noise, rejoin_x, y_mid, head=False, color=COMPARATOR)
    connect(ax, rejoin_x, y_mid, stats_x, y_mid, shrink_b=pad)

    # Lane labels: italic, small, set well clear of the boxes.
    ax.text(
        (real_extract_x + real_norm_x + box_w) / 2,
        y_real + box_h / 2 + 0.30,
        "REAL  PRE / POST  PAIRS",
        ha="center", va="bottom",
        fontsize=7.0, color=INK, fontstyle="italic", fontweight="bold",
    )
    ax.text(
        (noise_extract_x + noise_norm_x + box_w) / 2,
        y_noise - box_h / 2 - 0.30,
        "SCRAMBLED-SURROGATE  NOISE  FLOOR",
        ha="center", va="top",
        fontsize=7.0, color=COMPARATOR, fontstyle="italic", fontweight="bold",
    )

    return fig
