"""
Computes and saves, as a proper artifact, two face-validity statistics
Section 4.3 of the manuscript reports. Both were originally computed
interactively in this session's own scratch history and never saved to a
file -- flagged as a reproducibility gap by an independent review (the same
class of issue scripts/run_coverage_horizon_sweep.py already fixed for the
coverage-vs-horizon comparison). Both statistics are in fact fully
reproducible from research/m5_evaluation_results.json's existing per-series
fields; this script makes that reproduction explicit and saved rather than
ad hoc.
"""

import json
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parent.parent / "research"


def order_demand_ratio_stats(rows):
    """Median order_qty_median / weekly_demand ratio across the sample.
    weekly_demand is mean_test_demand * 7 (both already stored per series);
    this needs no per-period data, only the two per-series summary fields
    already saved in m5_evaluation_results.json.
    """
    ratios = []
    for r in rows:
        weekly_demand = r["mean_test_demand"] * 7
        if weekly_demand > 0:
            ratios.append(r["order_qty_median"] / weekly_demand)
    ratios = np.array(ratios)
    return {
        "n": len(ratios),
        "n_total": len(rows),
        "min": float(ratios.min()),
        "p25": float(np.percentile(ratios, 25)),
        "median": float(np.median(ratios)),
        "p75": float(np.percentile(ratios, 75)),
        "max": float(ratios.max()),
        "n_over_10x": int((ratios > 10).sum()),
    }


def holding_dominated_loss_count(rows):
    """Among series where the fused policy's total cost exceeds the naive
    baseline's, how many have the fused policy's OWN cost breakdown
    holding-dominated (fused_holding > fused_stockout on that series)? This
    is the definition manuscript.md Section 4.3 uses ("the loss is driven
    predominantly by holding cost rather than stockout cost") -- stated
    explicitly here since an independent reviewer, trying several other
    plausible definitions (delta-vs-baseline comparisons), could not
    reproduce the reported count without knowing this was the intended one.
    """
    losers = [r for r in rows if r["fused_cost"] > r["baseline_cost"]]
    holding_dominated = [r for r in losers if r["fused_holding"] > r["fused_stockout"]]
    return {
        "n_losers_vs_naive": len(losers),
        "n_holding_dominated": len(holding_dominated),
        "definition": "fused_holding > fused_stockout on that series (not a delta vs. baseline)",
    }


def realized_service_levels(rows, policies=("baseline", "sba", "ss", "ml", "fused")):
    """Aggregate realized service level per policy: 1 - (total unmet units
    / total demand units), both summed across the sample. Unmet units are
    recovered from each policy's stored stockout cost divided by that
    series' price (stockout_cost_per_unit = price * stockout_multiplier,
    and stockout_multiplier = 1.0 at the headline setting used to produce
    m5_evaluation_results.json) -- not previously saved as an artifact, the
    round-1 84.3%/92.8%/96.0% figures were computed ad hoc and not
    reproducible from a saved file, the same class of gap this script was
    created to close for the other two face-validity statistics.
    """
    total_demand_units = sum(r["mean_test_demand"] * 364 for r in rows)
    out = {}
    for p in policies:
        unmet_units = sum(r[f"{p}_stockout"] / r["price"] for r in rows if r["price"] > 0)
        out[p] = {
            "unmet_units": unmet_units,
            "total_demand_units": total_demand_units,
            "realized_service_level": 1.0 - unmet_units / total_demand_units if total_demand_units else None,
        }
    return out


def main():
    with open(RESEARCH / "m5_evaluation_results.json") as f:
        d = json.load(f)
    rows = d["per_series"]

    output = {
        "order_demand_ratio": order_demand_ratio_stats(rows),
        "holding_dominated_loss": holding_dominated_loss_count(rows),
        "realized_service_levels": realized_service_levels(rows),
    }

    out_path = RESEARCH / "face_validity_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    r = output["order_demand_ratio"]
    print(f"order/weekly-demand ratio (n={r['n']}/{r['n_total']}): "
          f"min={r['min']:.2f} p25={r['p25']:.2f} median={r['median']:.2f} "
          f"p75={r['p75']:.2f} max={r['max']:.2f} n>10x={r['n_over_10x']}")
    h = output["holding_dominated_loss"]
    print(f"naive-baseline losers: {h['n_losers_vs_naive']}, "
          f"holding-dominated ({h['definition']}): {h['n_holding_dominated']}")
    for p, sl in output["realized_service_levels"].items():
        print(f"  {p}: realized service level = {sl['realized_service_level']*100:.1f}%")
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
