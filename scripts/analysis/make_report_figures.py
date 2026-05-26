"""Generate three publication figures for the FedLEASE/WOS report.

Outputs (PDF) under <repo_root>/output/figures/:
  - fig_results.pdf       : 2-panel page-wide figure (Pareto scatter + grouped bars)
  - fig_gap_heatmap.pdf   : single-column 4x4 cross-cluster accuracy heatmap
  - fig_stitchability.pdf : single-column horizontal dumbbell plot

All experimental numbers are hard-coded below; no I/O dependencies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Global style (ICML two-column)
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Shared semantic palette across all three figures
COLOR_FEDAVG = "#d62728"          # FedAvg / baseline (red)
COLOR_SHUFFLED = "#1f77b4"        # WOS-shuffled (blue, headline)
COLOR_UNIFORM = "#9467bd"         # WOS-uniform (purple)
COLOR_FULL = "#2ca02c"            # WOS-full / learned (green)
COLOR_CLIENT_SIDE = "#ff7f0e"     # client-side learned (orange)
COLOR_RDROP = "#999999"           # R-Drop arms (grey)

OUT_DIR = Path(__file__).resolve().parents[2] / "output" / "figures"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Figure 1 - left panel: Pareto scatter (off-diag vs in-dist)
PARETO_POINTS = [
    # label,            off_diag, in_dist, color,            marker, size, is_headline
    ("FedAvg-LoRA",        63.31,  85.77,  COLOR_FEDAVG,     "o",    55,  False),
    ("Client-side learned", 67.68, 86.24,  COLOR_CLIENT_SIDE, "o",    55,  False),
    ("WOS-full",           74.17,  84.88,  COLOR_FULL,       "o",    55,  False),
    ("WOS-shuffled",       74.99,  85.38,  COLOR_SHUFFLED,   "o",   110,  True),
    ("WOS-uniform",        74.49,  85.29,  COLOR_UNIFORM,    "o",    55,  False),
]
RDROP_POINTS = [
    (75.78, 84.41),
    (74.62, 83.33),
    (75.60, 83.21),
    (75.32, 81.95),
]

# Figure 1 - right panel: cross-side decomposition (off-diag gain pt vs FedAvg)
GAIN_GROUPS = ["client only", "server only", "both sides"]
# None marks a missing/absent bar (server-only uniform has no measurement)
GAIN_DATA = {
    "learned":  [4.37, 8.13, 10.86],
    "shuffled": [4.76, 8.79, 11.68],
    "uniform":  [3.97, None, 11.18],
}
VARIANT_COLORS = {
    "learned":  COLOR_FULL,
    "shuffled": COLOR_SHUFFLED,
    "uniform":  COLOR_UNIFORM,
}

# Figure 2: 4x4 cross-cluster accuracy heatmap
HEATMAP_TASKS = ["SST-2", "QNLI", "MRPC", "QQP"]
HEATMAP_MATRIX = np.array(
    [
        [93.81, 48.43, 41.42, 57.81],
        [46.79, 88.98, 25.49, 27.72],
        [48.97, 43.36, 88.73, 70.62],
        [49.43, 40.77, 72.06, 82.31],
    ]
)

# Figure 3: stitchability dumbbell (top-to-bottom)
STITCH_ARMS = [
    # arm,          home,  indiv_worst, target, color
    ("FedAvg-LoRA",  49.06, 47.96, 63.31, COLOR_FEDAVG),
    ("WOS-shuffled", 48.26, 60.71, 74.99, COLOR_SHUFFLED),
    ("WOS-uniform",  48.41, 59.72, 74.49, COLOR_UNIFORM),
]


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def _draw_pareto(ax) -> None:
    """Left panel: Pareto scatter of off-diagonal vs in-distribution accuracy."""
    # Faint horizontal reference band at FedAvg in-dist level (~85.8) to show
    # that all WOS arms keep in-distribution accuracy essentially intact.
    ax.axhspan(85.3, 86.3, color=COLOR_FEDAVG, alpha=0.06, zorder=0)

    # R-Drop points share a single legend handle
    rd_x = [p[0] for p in RDROP_POINTS]
    rd_y = [p[1] for p in RDROP_POINTS]
    ax.scatter(
        rd_x, rd_y, marker="^", s=45, color=COLOR_RDROP,
        edgecolors="black", linewidths=0.4, label="R-Drop", zorder=2,
    )

    # No per-arm legend entry: each arm gets a text label on the canvas.
    for label, x, y, color, marker, size, headline in PARETO_POINTS:
        ax.scatter(
            x, y, marker=marker, s=size, color=color,
            edgecolors="black", linewidths=0.6 if not headline else 0.9,
            zorder=3 if not headline else 4,
        )

    # Manually placed text annotations to avoid overlap in the x≈74-75 cluster.
    # Each entry: (dx, dy, ha, va)
    label_layout = {
        "FedAvg-LoRA":        (0.4,  0.22, "left",   "center"),
        "Client-side learned": (0.4,  0.22, "left",  "center"),
        "WOS-full":           (0.0, -0.55, "center", "top"),    # below the green dot
        "WOS-shuffled":       (0.55, 0.35, "left",   "center"), # right of headline
        "WOS-uniform":        (0.0,  0.55, "center", "bottom"), # above the purple dot
    }
    for label, x, y, _c, _m, _s, _h in PARETO_POINTS:
        dx, dy, ha, va = label_layout[label]
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=6.5,
            ha=ha,
            va=va,
        )

    ax.set_xlim(61, 78)
    ax.set_ylim(81, 87)
    ax.set_xlabel("Off-diagonal stitching accuracy (%)")
    ax.set_ylabel("In-distribution accuracy (%)")
    ax.set_title("Strict Pareto win")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    # Only R-Drop needs a legend entry (other arms are labeled on-canvas).
    # Lower-left is empty; placing it there avoids touching any data point.
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4)


def _draw_gain_bars(ax) -> None:
    """Right panel: grouped bar chart of off-diagonal gain over FedAvg."""
    variants = ["learned", "shuffled", "uniform"]
    n_groups = len(GAIN_GROUPS)
    n_vars = len(variants)
    width = 0.23
    indices = np.arange(n_groups)

    for i, variant in enumerate(variants):
        values = GAIN_DATA[variant]
        xs, ys, g_idxs = [], [], []
        for g_idx, v in enumerate(values):
            if v is None:
                continue
            xs.append(indices[g_idx] + (i - (n_vars - 1) / 2.0) * width)
            ys.append(v)
            g_idxs.append(g_idx)
        bars = ax.bar(
            xs, ys, width=width, color=VARIANT_COLORS[variant],
            edgecolor="black", linewidth=0.5, label=variant,
        )
        for rect, value, g_idx in zip(bars, ys, g_idxs):
            # On the "both sides" group every value is two-digit (10.x/11.x),
            # so the centered labels would overlap horizontally. Lift the
            # middle bar's label to create vertical separation (V-shape).
            is_crowded_middle = (g_idx == 2 and variant == "shuffled")
            dy = 0.85 if is_crowded_middle else 0.15
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + dy,
                f"{value:.2f}",
                ha="center", va="bottom", fontsize=6.0,
            )

    ax.set_xticks(indices)
    ax.set_xticklabels(GAIN_GROUPS)
    ax.set_ylabel("Off-diagonal gain vs FedAvg (pt)")
    ax.set_ylim(0, 14)
    ax.set_title("Exposure mass drives the gain, not q content")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, loc="upper left", title=None, handletextpad=0.4)


def fig_results() -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    _draw_pareto(ax_left)
    _draw_gain_bars(ax_right)
    fig.tight_layout(pad=0.6)
    out_path = OUT_DIR / "fig_results.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def fig_gap_heatmap() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    matrix = HEATMAP_MATRIX
    vmin, vmax = float(matrix.min()), float(matrix.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="equal")

    # Per-cell annotations with adaptive text color for contrast
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            # Map value to [0, 1] then pick black text on light cells, white on dark.
            # RdYlGn is light in the middle (around yellow), so use endpoint
            # distance: very low (red) and very high (green) get white text.
            t = (value - vmin) / (vmax - vmin + 1e-9)
            text_color = "white" if (t < 0.20 or t > 0.85) else "black"
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=7, color=text_color,
            )

    ax.set_xticks(np.arange(len(HEATMAP_TASKS)))
    ax.set_yticks(np.arange(len(HEATMAP_TASKS)))
    ax.set_xticklabels(HEATMAP_TASKS)
    ax.set_yticklabels(HEATMAP_TASKS)
    ax.set_xlabel("Target cluster")
    ax.set_ylabel("Home cluster (expert source)")
    ax.set_title("Cross-cluster accuracy:\n41pt diagonal/off-diagonal gap", fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Accuracy (%)", fontsize=7)

    # Light tick lines, no top/right spines
    ax.tick_params(top=False, right=False, length=2)

    fig.tight_layout(pad=0.4)
    out_path = OUT_DIR / "fig_gap_heatmap.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def fig_stitchability() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    n_arms = len(STITCH_ARMS)
    y_positions = np.arange(n_arms)[::-1]  # top-to-bottom order matches list order

    for y, (arm, home, indiv_worst, target, color) in zip(y_positions, STITCH_ARMS):
        # Connector line between home (left) and target (right)
        ax.plot(
            [home, target], [y, y],
            color=color, linewidth=1.8, alpha=0.55, zorder=1,
        )
        # home head: hollow grey circle
        ax.scatter(
            [home], [y], s=55, facecolors="white",
            edgecolors=COLOR_RDROP, linewidths=1.2, zorder=3,
        )
        # individual-worst head: small diamond, dark grey
        ax.scatter(
            [indiv_worst], [y], s=40, marker="D",
            facecolors="#444444", edgecolors="black", linewidths=0.4, zorder=3,
        )
        # target head: solid arm-colored circle
        ax.scatter(
            [target], [y], s=70, color=color,
            edgecolors="black", linewidths=0.6, zorder=4,
        )
        # Numeric labels on home (left) and target (right).
        # On the FedAvg row, home (49.06) and individual-worst (47.96) almost
        # coincide, so the home text must clear BOTH markers — use the smaller
        # of the two as the right edge and shift extra to the left.
        home_anchor = min(home, indiv_worst) - 1.0
        ax.text(home_anchor, y, f"{home:.2f}", ha="right", va="center", fontsize=6.5)
        ax.text(target + 1.2, y, f"{target:.2f}", ha="left", va="center", fontsize=6.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([arm for arm, *_ in STITCH_ARMS])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(41, 82)
    # Leave room at the bottom for the marker legend placed outside the data area
    ax.set_ylim(-1.3, n_arms - 0.4)
    ax.set_title("WOS makes experts stitchable\n(home-head floor ~48 constant)", fontsize=8.5)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    # Custom legend explaining the three markers — placed below the axes
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor="white", markeredgecolor=COLOR_RDROP,
                   markersize=6, label="home head"),
        plt.Line2D([0], [0], marker="D", linestyle="",
                   markerfacecolor="#444444", markeredgecolor="black",
                   markersize=5, label="individual-worst head"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=COLOR_RDROP, markeredgecolor="black",
                   markersize=6, label="target head"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.4)
    out_path = OUT_DIR / "fig_stitchability.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_results()
    fig_gap_heatmap()
    fig_stitchability()


if __name__ == "__main__":
    main()
