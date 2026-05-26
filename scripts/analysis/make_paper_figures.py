"""Fig 2 — main result + mechanism composite, rendered with figures4papers house style.

Reads docs/result/canonical_numbers.json (D0 canonical source) and writes
output/figures/fig_main_composite.{pdf,png}.

Layout: 1x3 subplots
  (A) Pareto landscape — off-diag vs in-dist with seed error bars
  (B) Side decomposition — client / server / both, grouped bars
  (C) q-invariance — strip plot of three q variants
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

# ---------------------------------------------------------------------------
# Load figures4papers helper module
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = Path(
    "/storage/homefs/hw24w089/.claude/skills/figures4papers/scripts/scientific_figure_pro.py"
)
spec = importlib.util.spec_from_file_location("sfp", SKILL_PATH)
sfp = importlib.util.module_from_spec(spec)
sys.modules["sfp"] = sfp
spec.loader.exec_module(sfp)

P = sfp.PALETTE

# Apply house style (paper-friendly font + axes weight)
sfp.apply_publication_style(sfp.FigureStyle(font_size=15, axes_linewidth=2.0))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CANON_PATH = REPO_ROOT / "docs" / "result" / "canonical_numbers.json"
OUT_DIR = REPO_ROOT / "output" / "figures"
with CANON_PATH.open() as f:
    CANON = json.load(f)

# Semantic color mapping (drawn from figures4papers PALETTE)
COLOR = {
    "fedavg":       P["red_strong"],     # baseline (red)
    "client":       P["teal"],           # client-side learned (teal)
    "wos_full":     P["green_3"],        # WOS-full (green)
    "wos_shuffled": P["blue_main"],      # headline (deep blue)
    "wos_uniform":  P["violet"],         # data-independent q (violet)
    "rdrop":        P["neutral"],        # R-Drop arms (grey)
}


def _agg(key, section="main_arms"):
    a = CANON[section][key]["aggregate"]
    return a["in_dist_mean"], a["in_dist_std"], a["off_diag_mean"], a["off_diag_std"]


# ---------------------------------------------------------------------------
# Subplot (A) — Pareto scatter
# ---------------------------------------------------------------------------

def draw_pareto(ax):
    baseline_in, _, _, _ = _agg("fedavg_baseline")
    # Soft horizontal reference band at FedAvg in-dist
    ax.axhspan(baseline_in - 0.30, baseline_in + 0.30,
               color=COLOR["fedavg"], alpha=0.06, zorder=0)
    # Arrow + caption marking the headline lift
    ax.annotate("", xy=(74.6, 85.50), xytext=(64.2, 85.77),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.6, mutation_scale=18,
                                connectionstyle="arc3,rad=-0.18"),
                zorder=2)
    ax.text(69.4, 86.65, r"$+11.68\,$pt",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color="#222222")

    arms = [
        ("fedavg_baseline",       "FedAvg-LoRA baseline", COLOR["fedavg"],       "o", 140, False),
        ("trainonly_learned",     "Client-side learned",  COLOR["client"],       "s", 130, False),
        ("wos_full_learned",      "WOS-full (both learned)",     COLOR["wos_full"],     "D", 130, False),
        ("wos_uniform_both",      "WOS-uniform-both (data-indep.)", COLOR["wos_uniform"], "v", 150, False),
        ("wos_shuffled_headline", "WOS-shuffled (headline)",      COLOR["wos_shuffled"], "*", 460, True),
    ]

    # Main arms with error bars + scatter
    for key, label, color, marker, size, headline in arms:
        in_m, in_s, off_m, off_s = _agg(key)
        ax.errorbar(off_m, in_m, xerr=off_s, yerr=in_s,
                    fmt="none", ecolor=color, elinewidth=1.4,
                    capsize=3.0, capthick=1.4, alpha=0.6, zorder=2.5)
        ax.scatter(off_m, in_m, marker=marker, s=size, color=color,
                   edgecolors="white",
                   linewidths=1.6 if not headline else 2.2,
                   label=label,
                   zorder=3 if not headline else 4)

    # R-Drop arms — grey triangles, single legend handle (added last so it
    # appears at the bottom of the legend)
    rd_x, rd_y = [], []
    for key in ["rdrop_kl_0_5", "rdrop_kl_1_0", "rdrop_kl_2_0", "rdrop_kl_5_0"]:
        in_m, _, off_m, _ = _agg(key, "rdrop_arms")
        rd_x.append(off_m); rd_y.append(in_m)
    ax.scatter(rd_x, rd_y, marker="^", s=110, color=COLOR["rdrop"],
               edgecolors="white", linewidths=1.0, alpha=0.85,
               label=r"R-Drop (4$\lambda$$\times$3 seeds)", zorder=2)

    ax.set_xlim(60.0, 79.5)
    ax.set_ylim(80.5, 88.4)
    ax.set_xlabel("Off-diag stitching acc. (%)")
    ax.set_ylabel("In-dist accuracy (%)")
    ax.set_title("A. Pareto landscape", loc="left", fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    leg = ax.legend(loc="lower left", ncol=1, fontsize=9,
                    handletextpad=0.4, labelspacing=0.32,
                    borderaxespad=0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_edgecolor("none")


# ---------------------------------------------------------------------------
# Subplot (B) — Side-isolation grouped bars
# ---------------------------------------------------------------------------

def draw_side_isolation(ax):
    panel = CANON["side_isolated_decomposition"]
    group_labels = ["client\nonly", "server\nonly", "both\nsides"]
    group_keys = ["client_only", "server_only", "both_sides"]
    variants = [
        ("learned",         COLOR["wos_full"]),
        ("shuffled",        COLOR["wos_shuffled"]),
        ("uniform_nonhome", COLOR["wos_uniform"]),
    ]
    variant_labels = {
        "learned": "learned",
        "shuffled": "shuffled",
        "uniform_nonhome": "uniform-nh",
    }

    n_groups = len(group_keys)
    n_vars = len(variants)
    width = 0.25
    group_spread = 1.0
    indices = np.arange(n_groups) * group_spread

    for i, (vkey, vcolor) in enumerate(variants):
        xs, ys, ns = [], [], []
        for g_idx, gkey in enumerate(group_keys):
            cell = panel[gkey].get(vkey)
            if cell is None:
                continue
            xs.append(indices[g_idx] + (i - (n_vars - 1) / 2.0) * width)
            ys.append(cell["delta_off"])
            ns.append(cell["n"])
        ax.bar(xs, ys, width=width,
               color=vcolor, edgecolor="black", linewidth=1.0,
               label=variant_labels[vkey], zorder=3)
        for x, y, n_val in zip(xs, ys, ns):
            tag = f"{y:.1f}"
            ax.text(x, y + 0.4, tag,
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if n_val == 3 else "normal",
                    color="#222222")

    # Group-summary brackets above the bars
    top_y = 14.2
    group_means = []
    for gkey in group_keys:
        vals = [panel[gkey][v]["delta_off"]
                for v, _ in variants if panel[gkey].get(v) is not None]
        group_means.append(np.mean(vals))
    for idx, gm in enumerate(group_means):
        ax.text(indices[idx], top_y, f"+{gm:.1f}",
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", color="#222222")

    # Italic additive call-out
    ax.text((indices[0] + indices[2]) / 2.0, 16.3,
            "client (+4) + server (+8) ≈ both (+11.7)",
            ha="center", va="bottom", fontsize=10, style="italic",
            color="#555555")

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_xticks(indices)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(r"$\Delta$ off-diag vs FedAvg (pt)")
    ax.set_ylim(-1.5, 18.0)
    ax.set_title("B. Side decomposition", loc="left", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.25)
    ax.legend(loc="lower center", ncol=3, fontsize=10,
              handletextpad=0.4, handlelength=1.2, columnspacing=1.2,
              bbox_to_anchor=(0.5, -0.42), borderaxespad=0.0)
    # Footnote: place at the very top-right inside axes, above all bars


# ---------------------------------------------------------------------------
# Subplot (C) — q-invariance strip plot
# ---------------------------------------------------------------------------

def draw_q_invariance(ax):
    arms = [
        ("wos_full_learned",      "learned",     COLOR["wos_full"]),
        ("wos_shuffled_headline", "shuffled",    COLOR["wos_shuffled"]),
        ("wos_uniform_both",      "uniform\nnon-home", COLOR["wos_uniform"]),
    ]
    baseline_off = CANON["main_arms"]["fedavg_baseline"]["aggregate"]["off_diag_mean"]

    # Narrow grey band highlighting cluster of means
    ax.axhspan(73.8, 75.4, color="#cccccc", alpha=0.30, zorder=0)

    rng = np.random.default_rng(0)
    x_positions = np.arange(len(arms))
    for x, (akey, label, color) in zip(x_positions, arms):
        arm = CANON["main_arms"][akey]
        per_seed = arm["per_seed"]
        seed_offs = [v["off_diag"] for v in per_seed.values()
                     if v is not None and v.get("off_diag") is not None]
        in_m = arm["aggregate"]["off_diag_mean"]
        in_s = arm["aggregate"]["off_diag_std"]

        jitter = rng.uniform(-0.10, 0.10, size=len(seed_offs))
        ax.scatter(np.full(len(seed_offs), x) + jitter, seed_offs,
                   s=110, color=color, edgecolors="white",
                   linewidths=1.5, alpha=0.85, zorder=4)
        ax.errorbar(x, in_m, yerr=in_s,
                    fmt="_", color=color, ecolor="black",
                    elinewidth=1.5, capsize=7, markersize=24,
                    markeredgewidth=2.2, zorder=5)
        ax.text(x, 65.3, f"n={len(seed_offs)}",
                ha="center", va="bottom", fontsize=10, color="#555555")

    # Baseline reference line + label
    ax.axhline(baseline_off, color=COLOR["fedavg"], linewidth=1.6,
               linestyle="--", alpha=0.9, zorder=2)
    ax.text(2.40, baseline_off - 0.30,
            f"FedAvg = {baseline_off:.2f}",
            fontsize=10, color=COLOR["fedavg"], ha="right", va="top",
            fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label, _ in arms], fontsize=12)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_ylim(62.0, 78.0)
    ax.set_ylabel("Off-diag stitching acc. (%)")
    ax.set_title(r"C. $q$-invariance", loc="left", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.25)


# ---------------------------------------------------------------------------
# Fig 3 — Stitchability robustness
# ---------------------------------------------------------------------------

def draw_stitchability(ax):
    panel = CANON["stitching_matrix"]
    heads = ["home_head", "individual_worst", "individual_mean", "loco_mean", "cluster_average"]
    head_labels = [
        "home\n(un-stitched)",
        "individual\nworst",
        "individual\nmean",
        "LOCO\nmean",
        "cluster\naverage",
    ]
    arms = [
        ("fedavg_baseline_off", "FedAvg-LoRA baseline", COLOR["fedavg"],       "o", 150),
        ("wos_shuffled",        "WOS-shuffled (headline)",  COLOR["wos_shuffled"], "*", 380),
        ("wos_uniform_both",    "WOS-uniform-both (data-indep.)",  COLOR["wos_uniform"], "v", 170),
    ]

    x = np.arange(len(heads))

    # Shaded band marking the "un-stitched" floor zone (~48)
    ax.axhspan(47, 49.5, color="#dddddd", alpha=0.35, zorder=0)
    ax.text(4.45, 48.3,
            "un-stitched floor\n(home-head ≈ 48 across arms)",
            ha="right", va="center", fontsize=9.5,
            color="#555555", style="italic")

    for key, label, color, marker, size in arms:
        vals = [panel[key][h] for h in heads]
        ax.plot(x, vals, color=color, linewidth=2.5, alpha=0.85, zorder=2)
        ax.scatter(x, vals, marker=marker, s=size, color=color,
                   edgecolors="white", linewidths=1.8, label=label, zorder=3)

    # Only annotate the right-most column (cluster_average) — the headline
    # number. Other columns are covered by the lift call-outs above.
    cluster_idx = len(heads) - 1
    for key, _label, color, _marker, _size in arms:
        v = panel[key]["cluster_average"]
        # Push baseline label below the marker; WOS labels above to avoid
        # vertical collision between the two WOS lines.
        if key == "fedavg_baseline_off":
            dy, va = -2.4, "top"
        elif key == "wos_shuffled":
            dy, va = +1.4, "bottom"
        else:  # wos_uniform_both
            dy, va = -2.4, "top"
        ax.text(cluster_idx, v + dy, f"{v:.1f}",
                ha="center", va=va, fontsize=10,
                color=color, fontweight="bold", zorder=4)

    # +11.7 pt arrow on cluster_average column
    baseline_cluster = panel["fedavg_baseline_off"]["cluster_average"]
    wos_cluster      = panel["wos_shuffled"]["cluster_average"]
    ax.annotate(
        "", xy=(4.22, wos_cluster), xytext=(4.22, baseline_cluster),
        arrowprops=dict(arrowstyle="<|-|>", color="#222222",
                        lw=1.8, mutation_scale=18),
        zorder=3,
    )
    ax.text(4.35, (wos_cluster + baseline_cluster) / 2,
            f"+{wos_cluster - baseline_cluster:.1f}\npt",
            ha="left", va="center", fontsize=15, fontweight="bold",
            color="#222222")

    # Per-head WOS lift call-outs (small, on the upper line)
    for head_key, hx in zip(["individual_worst", "individual_mean", "loco_mean"], [1, 2, 3]):
        b = panel["fedavg_baseline_off"][head_key]
        w = panel["wos_shuffled"][head_key]
        ax.text(hx, max(w, b) + 4.0, f"+{w - b:.1f}",
                ha="center", va="bottom", fontsize=10,
                color="#1f3a5f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(head_labels)
    ax.set_xlim(-0.4, 4.95)
    ax.set_ylim(43, 82)
    ax.set_ylabel("Off-diag stitching accuracy (%)")
    ax.set_xlabel("Target-task classification head construction")
    ax.set_title(
        "Stitchability is robust to head construction (n=3)",
        loc="left", fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.25)
    leg = ax.legend(loc="upper left", fontsize=10, handletextpad=0.4,
                    labelspacing=0.32, borderaxespad=0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_edgecolor("none")


def fig_stitchability():
    fig, axes = sfp.create_subplots(1, 1, figsize=(12, 5.0))
    draw_stitchability(axes[0])
    fig.tight_layout(pad=1.2)
    out_base = OUT_DIR / "fig_stitchability"
    paths = sfp.finalize_figure(fig, out_base, formats=["pdf", "png"], dpi=300, pad=0.15)
    for p in paths:
        print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def fig_main_composite():
    fig, axes = sfp.create_subplots(
        1, 3, figsize=(17, 5.0),
        gridspec_kw={"width_ratios": [1.25, 1.0, 0.95]},
    )
    draw_pareto(axes[0])
    draw_side_isolation(axes[1])
    draw_q_invariance(axes[2])
    fig.tight_layout(pad=1.4, w_pad=3.6)
    out_base = OUT_DIR / "fig_main_composite"
    paths = sfp.finalize_figure(fig, out_base, formats=["pdf", "png"], dpi=300, pad=0.15)
    for p in paths:
        print(f"wrote {p}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_main_composite()
    fig_stitchability()


if __name__ == "__main__":
    main()
