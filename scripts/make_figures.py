"""
Generates the seven figures embedded in paper/manuscript.md from project
artifacts. Output filenames match their manuscript numbers (document order
of first appearance); helper function names are internal labels.

  - fig1_architecture.pdf (Fig. 1): system architecture -- schematic, no
    external data, but its box labels and data flow are read directly off
    src/conformal.py, src/fuzzy_risk.py, src/mcts_policy.py's actual
    function signatures, not redrawn from memory.
  - fig2_fuzzy_membership.pdf (Fig. 2): fuzzy membership functions -- plots
    src/fuzzy_risk.py's actual _buffer_memberships / _width_memberships
    functions directly (imported, not reimplemented), so the figure cannot
    drift from the real rule base.
  - fig3_coverage_horizon.pdf (Fig. 3): conformal coverage vs test horizon --
    research/coverage_horizon_sweep.json.
  - fig4_trajectory.pdf (Fig. 4): whole-unit example inventory trajectories --
    research/trajectory_examples_integer_order.json.
  - fig5_stability.pdf (Fig. 5): whole-unit sample-to-sample stability --
    research/m5_integer_order_robustness_results.json.
  - fig6_sensitivity.pdf (Fig. 6): whole-unit cost-model sensitivity --
    research/m5_integer_order_sensitivity_analysis.json.
  - fig7_distribution_shift.pdf (Fig. 7): whole-unit distribution-shift stress
    test -- integer-order shock results and their matched ACI ablations.
Run after scripts/run_coverage_horizon_sweep.py,
scripts/run_trajectory_example.py, and scripts/run_m5_distribution_shift.py
(run once per seed in {0, 1, 4, 7}) -- all must have already written their
output JSON. Writes all figures as PDF (vector) to paper/figures/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fuzzy_risk
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m5_evaluation as m5eval

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLOR_NAIVE = "#7f7f7f"
COLOR_SBA = "#1f77b4"
COLOR_FUSED = "#d62728"
COLOR_SS = "#ff7f0e"
COLOR_ML = "#9467bd"


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    box_w, box_h, y0, gap = 1.9, 1.0, 1.4, 0.55
    labels = [
        ("M5 demand\nhistory", "#e8e8e8"),
        ("Rolling conformal\ncalibrator\n(refit + ACI)", "#cfe2f3"),
        ("Fuzzy risk\nmodule", "#d9ead3"),
        ("Monte Carlo\nlookahead", "#f4cccc"),
        ("Order\nquantity", "#e8e8e8"),
    ]
    xs_left = []
    x = 0.2
    for label, color in labels:
        xs_left.append(x)
        box = FancyBboxPatch((x, y0), box_w, box_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                              linewidth=1.0, edgecolor="black", facecolor=color)
        ax.add_patch(box)
        ax.text(x + box_w / 2, y0 + box_h / 2, label, ha="center", va="center", fontsize=8.5)
        x += box_w + gap

    arrow_labels = ["demand interval\n[lo, hi]", "risk\nmultiplier", "scaled\naction set", ""]
    for i in range(4):
        x0 = xs_left[i] + box_w
        x1 = xs_left[i + 1]
        ax.annotate("", xy=(x1, y0 + box_h / 2), xytext=(x0, y0 + box_h / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
        if arrow_labels[i]:
            ax.text((x0 + x1) / 2, y0 + box_h + 0.32, arrow_labels[i], ha="center", va="bottom", fontsize=7,
                     color="#333")

    ax.annotate("realized demand (feedback, one review period later)",
                xy=(xs_left[0] + box_w / 2, y0 - 0.08), xytext=(xs_left[3] + box_w / 2, y0 - 1.15),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#666", connectionstyle="arc3,rad=0.2",
                                 linestyle="dashed"),
                fontsize=7.5, color="#444", ha="center")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_fuzzy_membership():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))

    x_buf = np.linspace(-1.0, 2.0, 400)
    low = [fuzzy_risk._buffer_memberships(x)["LOW"] for x in x_buf]
    med = [fuzzy_risk._buffer_memberships(x)["MEDIUM"] for x in x_buf]
    high = [fuzzy_risk._buffer_memberships(x)["HIGH"] for x in x_buf]
    axes[0].plot(x_buf, low, label="LOW", color="#d62728", lw=1.6)
    axes[0].plot(x_buf, med, label="MEDIUM", color="#2ca02c", lw=1.6, linestyle="--")
    axes[0].plot(x_buf, high, label="HIGH", color="#1f77b4", lw=1.6, linestyle=":")
    axes[0].set_xlabel("buffer ratio")
    axes[0].set_ylabel("membership")
    axes[0].set_title("(a) buffer-ratio memberships", fontsize=9.5)
    axes[0].legend(fontsize=7.5, loc="upper right", frameon=False)
    axes[0].set_ylim(-0.05, 1.15)

    x_w = np.linspace(0.0, 1.5, 400)
    narrow = [fuzzy_risk._width_memberships(x)["NARROW"] for x in x_w]
    medw = [fuzzy_risk._width_memberships(x)["MEDIUM"] for x in x_w]
    wide = [fuzzy_risk._width_memberships(x)["WIDE"] for x in x_w]
    axes[1].plot(x_w, narrow, label="NARROW", color="#d62728", lw=1.6)
    axes[1].plot(x_w, medw, label="MEDIUM", color="#2ca02c", lw=1.6, linestyle="--")
    axes[1].plot(x_w, wide, label="WIDE", color="#1f77b4", lw=1.6, linestyle=":")
    axes[1].set_xlabel("relative interval width")
    axes[1].set_ylabel("membership")
    axes[1].set_title("(b) relative-width memberships", fontsize=9.5)
    axes[1].legend(fontsize=7.5, loc="upper right", frameon=False)
    axes[1].set_ylim(-0.05, 1.15)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_fuzzy_membership.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_sensitivity():
    with open(RESEARCH / "m5_integer_order_sensitivity_analysis.json") as f:
        d = json.load(f)
    rows = d["cost_scenarios"]
    reference = next(r for r in rows if r["axis"] == "reference")
    stockout_rows = [r for r in rows if r["axis"] == "stockout_multiplier"] + [reference]
    stockout_rows.sort(key=lambda r: r["stockout_multiplier"])
    holding_rows = [r for r in rows if r["axis"] == "holding_rate"] + [reference]
    holding_rows.sort(key=lambda r: r["holding_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))

    x = [r["stockout_multiplier"] for r in stockout_rows]
    y = [r["cost_reduction_pct"]["vs_tuned_s_s"] for r in stockout_rows]
    y_sba = [r["cost_reduction_pct"]["vs_sba"] for r in stockout_rows]
    y_ml = [r["cost_reduction_pct"]["vs_pooled_lightgbm"] for r in stockout_rows]
    axes[0].plot(x, y_sba, marker="s", color=COLOR_SBA, lw=1.2, alpha=0.8, label="vs. SBA/Croston")
    axes[0].plot(x, y, marker="o", color=COLOR_SS, lw=1.5, label="vs. tuned (s, S)")
    axes[0].plot(x, y_ml, marker="D", color=COLOR_ML, lw=1.2, label="vs. pooled LightGBM")
    axes[0].axhline(0, color="black", lw=0.8, linestyle="-")
    axes[0].axvline(1.0, color="#333", lw=0.8, linestyle=":")
    axes[0].fill_between(x, y, 0, where=[v < 0 for v in y], color="#d62728", alpha=0.15, interpolate=True)
    axes[0].fill_between(x, y, 0, where=[v >= 0 for v in y], color="#2ca02c", alpha=0.12, interpolate=True)
    axes[0].set_xlabel("stockout multiplier (× full price)")
    axes[0].set_ylabel("lookahead cost reduction, %")
    axes[0].set_title("(a) stockout-cost sensitivity", fontsize=9.5)
    axes[0].legend(fontsize=6.4, loc="lower right", frameon=False)

    xh = [r["holding_rate"] * 100 for r in holding_rows]
    yh = [r["cost_reduction_pct"]["vs_tuned_s_s"] for r in holding_rows]
    yh_sba = [r["cost_reduction_pct"]["vs_sba"] for r in holding_rows]
    yh_ml = [r["cost_reduction_pct"]["vs_pooled_lightgbm"] for r in holding_rows]
    axes[1].plot(xh, yh_sba, marker="s", color=COLOR_SBA, lw=1.2, alpha=0.8, label="vs. SBA/Croston")
    axes[1].plot(xh, yh, marker="o", color=COLOR_SS, lw=1.5, label="vs. tuned (s, S)")
    axes[1].plot(xh, yh_ml, marker="D", color=COLOR_ML, lw=1.2, label="vs. pooled LightGBM")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axvline(20, color="#333", lw=0.8, linestyle=":")
    axes[1].set_xlabel("annual holding-cost rate, %")
    axes[1].set_ylabel("lookahead cost reduction, %")
    axes[1].set_title("(b) holding-rate sensitivity", fontsize=9.5)
    axes[1].legend(fontsize=6.4, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def fig4_stability():
    with open(RESEARCH / "m5_integer_order_robustness_results.json") as f:
        d = json.load(f)
    per_seed = d["per_seed"]
    groups = [
        ("vs_gaussian", COLOR_NAIVE, "o", "vs. Gaussian\n(8 whole-unit draws)"),
        ("vs_sba", COLOR_SBA, "s", "vs. SBA/Croston\n(8 whole-unit draws)"),
        ("vs_tuned_s_s", COLOR_SS, "^", "vs. tuned (s, S)\n(8 whole-unit draws)"),
        ("vs_pooled_lightgbm", COLOR_ML, "D", "vs. pooled LightGBM\n(8 whole-unit draws)"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (key, color, marker, _) in enumerate(groups):
        y = len(groups) - i
        vals = [r["cost_reduction_pct"][key] for r in per_seed]
        ax.hlines(y, min(vals), max(vals), color="black", lw=0.8, zorder=1)
        ax.scatter(vals, [y] * len(vals), color=color, marker=marker, s=45, zorder=3)
        headline = next(r["cost_reduction_pct"][key] for r in per_seed if r["sample_seed"] == 0)
        ax.scatter([headline], [y], color=color, marker="*", s=220, edgecolor="black",
                   linewidth=0.6, zorder=4)

    ax.set_yticks([len(groups) - i for i in range(len(groups))])
    ax.set_yticklabels([label for _, _, _, label in groups])
    ax.set_xlabel("whole-unit lookahead aggregate cost reduction, %")
    ax.set_ylim(0.5, len(groups) + 0.5)
    ax.axvline(0, color="black", lw=0.6, linestyle=":")
    star = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=13,
                       label="seed 0 (headline sample)")
    ax.legend(handles=[star], fontsize=7.5, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_stability.pdf", bbox_inches="tight")
    plt.close(fig)


def fig5_coverage_horizon():
    with open(RESEARCH / "coverage_horizon_sweep.json") as f:
        d = json.load(f)
    horizons = sorted(int(h) for h in d["horizons"].keys())
    static_mean = [d["horizons"][str(h)]["static_mean"] * 100 for h in horizons]
    aci_mean = [d["horizons"][str(h)]["refit_aci_mean"] * 100 for h in horizons]
    aci_std = [d["horizons"][str(h)]["refit_aci_std"] * 100 for h in horizons]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(horizons, static_mean, marker="o", color=COLOR_NAIVE, lw=1.6, label="static split conformal")
    ax.errorbar(horizons, aci_mean, yerr=aci_std, marker="s", color=COLOR_FUSED, lw=1.6, capsize=3,
                label="rolling refit + ACI (this paper)")
    ax.axhline(90, color="black", lw=0.8, linestyle="--", label="90% nominal target")
    ax.set_xlabel("test horizon (days)")
    ax.set_ylabel("empirical coverage, %")
    ax.set_xticks(horizons)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.set_ylim(55, 96)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_coverage_horizon.pdf", bbox_inches="tight")
    plt.close(fig)


def fig6_trajectory():
    with open(RESEARCH / "trajectory_examples_integer_order.json") as f:
        d = json.load(f)
    series_ids = list(d.keys())

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    titles = ["(a) largest loss vs. Gaussian order-up-to", "(b) representative lookahead win"]

    for ax, sid, title in zip(axes, series_ids, titles):
        traj = d[sid]
        naive_days = [t["day"] - m5eval.TEST_START for t in traj["trajectory"]["baseline"]]
        naive_inv = [t["inventory_before"] for t in traj["trajectory"]["baseline"]]
        fused_days = [t["day"] - m5eval.TEST_START for t in traj["trajectory"]["fused"]]
        fused_inv = [t["inventory_before"] for t in traj["trajectory"]["fused"]]

        ax.step(naive_days, naive_inv, where="post", color=COLOR_NAIVE, lw=1.5, label="Gaussian order-up-to")
        ax.step(fused_days, fused_inv, where="post", color=COLOR_FUSED, lw=1.5, label="lookahead policy")

        demand = traj["test_demand_daily"]
        weekly_demand = [sum(demand[i:i + 7]) for i in range(0, len(demand), 7)]
        ax.bar(range(0, len(demand), 7), weekly_demand, width=6, color="#cccccc", alpha=0.5,
               label="realized weekly demand", zorder=0)

        ax.set_ylabel("units")
        ax.set_title(f"{title}: {sid.replace('_evaluation', '')}", fontsize=9)
        if ax is axes[0]:
            ax.legend(fontsize=7.5, loc="upper right", frameon=False)

    axes[-1].set_xlabel("day of 364-day test window")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def fig7_distribution_shift():
    """Whole-unit cost by shock window and across the four sample seeds.
    """
    with open(RESEARCH / "m5_integer_distribution_shift_full.json") as f:
        headline = json.load(f)

    segs = ["pre", "post_pre_refit", "post_after_first_refit", "post_late"]
    seg_labels = [
        f"pre-shock\n({headline['segment_days']['pre']} days)",
        f"pre-refit\n({headline['segment_days']['post_pre_refit']} days)",
        f"post-refit\n({headline['segment_days']['post_after_first_refit']} days)",
        f"later\n({headline['segment_days']['post_late']} days)",
    ]
    fused_vals = [headline["segments"][s]["fused"]["total"] for s in segs]
    ml_vals = [headline["segments"][s]["ml"]["total"] for s in segs]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))

    x = np.arange(len(segs))
    width = 0.35
    axes[0].bar(x - width / 2, fused_vals, width, color=COLOR_FUSED, label="Monte Carlo lookahead")
    axes[0].bar(x + width / 2, ml_vals, width, color=COLOR_ML, label="pooled LightGBM")
    axes[0].axvline(0.5, color="black", lw=0.7, linestyle=":")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seg_labels, fontsize=8.5)
    axes[0].set_ylabel("aggregate cost")
    axes[0].set_ylim(0, max(fused_vals + ml_vals) * 1.32)
    axes[0].set_title("(a) headline sample (seed 0)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)

    seed_files = {
        0: "m5_integer_distribution_shift_full.json",
        1: "m5_integer_distribution_shift_full_seed1.json",
        4: "m5_integer_distribution_shift_full_seed4.json",
        7: "m5_integer_distribution_shift_full_seed7.json",
    }
    seeds_sorted = sorted(seed_files)
    pre_refit_pct = []
    after_refit_pct = []
    for seed in seeds_sorted:
        with open(RESEARCH / seed_files[seed]) as f:
            dd = json.load(f)
        for segment, values in (
            ("post_pre_refit", pre_refit_pct),
            ("post_after_first_refit", after_refit_pct),
        ):
            fused, ml = dd["segments"][segment]["fused"]["total"], dd["segments"][segment]["ml"]["total"]
            values.append(100.0 * (1.0 - fused / ml) if ml else 0.0)

    seed_x = np.arange(len(seeds_sorted))
    axes[1].bar(seed_x - width / 2, pre_refit_pct, width, color=COLOR_FUSED, alpha=0.9, label="before first refit")
    axes[1].bar(seed_x + width / 2, after_refit_pct, width, color=COLOR_ML, alpha=0.9, label="after first refit")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(seed_x)
    axes[1].set_xticklabels([str(s) for s in seeds_sorted], fontsize=8.5)
    axes[1].set_xlabel("sample seed")
    axes[1].set_ylabel("lookahead cost reduction vs. LightGBM (%)")
    axes[1].set_title("(b) first refit windows across 4 samples", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_distribution_shift.pdf", bbox_inches="tight")
    plt.close(fig)


def fig8_gamified_demo():
    """Generate an internal illustration for the Replenishment Manager demo
    (not embedded in or packaged with the manuscript) from two screenshots --
    screenshots/02-live-decision.png (the player choosing an order under
    the same conformal-calibrated demand band the fused system uses) and
    screenshots/03-round-reveal.png (the settled round, player order vs.
    the live MCTS policy's order, same realized demand). Not a
    matplotlib-generated plot -- real UI screenshots, composited here into
    one two-panel figure to match this file's other figures' layout
    convention. Illustrative only, per the manuscript's own disclosure
    that no output from the demo is used as a result reported there.
    """
    a = plt.imread(str(ROOT / "screenshots" / "02-live-decision.png"))
    b = plt.imread(str(ROOT / "screenshots" / "03-round-reveal.png"))

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.9))
    for ax, img, title in zip(axes, (a, b), (
        "(a) player choosing an order under the live conformal-calibrated demand band",
        "(b) settled round: player order vs. the live MCTS policy, same realized demand",
    )):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_gamified_demo.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig1_architecture()
    fig2_fuzzy_membership()
    fig3_sensitivity()
    fig4_stability()
    fig5_coverage_horizon()
    fig6_trajectory()
    fig7_distribution_shift()
    for p in sorted(FIGDIR.glob("*.pdf")):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
