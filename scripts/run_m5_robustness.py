"""
Rechecks the M5 result under the corrected pre-test strata and shared
Gaussian opening state. It reports sample-to-sample variance, price-based
cost sensitivity, a paired zero-start sensitivity, and MCTS planner-seed
variability on one fixed sample.

Reuses `run_m5_evaluation.py`'s `evaluate_sample` (identical simulation code
to the single reported run -- same conformal fix, same cost-consistency fix
in mcts_policy.py, same SBA/Croston baseline) so results are directly
comparable to `research/m5_evaluation_results.json`'s headline numbers.

1. Sample-to-sample stability: draw 8 independent stratified samples
   (seeds 0-7; seed 0 is the one reported as the headline result) at the
   default cost model, report the distribution of the aggregate
   fused-vs-naive and fused-vs-SBA percent cost change across draws.
2. Cost-model sensitivity: fix the sample (seed 0, the headline sample) and
   sweep the stockout-cost multiplier (fraction of full retail price
   actually lost on a stockout -- full price is a conservative upper bound;
   real margin is typically lower) and the annual holding-cost rate across
   their documented plausible ranges, reporting how the fused-vs-SBA
   advantage moves. Coverage checking is skipped here (cost-model-
   independent, already established in the headline run) to keep the sweep
   fast.
3. Opening-inventory sensitivity: compare the selected common Gaussian
   opening target with a common zero start on the same sample.
4. Planner-seed variability: hold sample seed 0 fixed and vary only the
   deterministic per-series MCTS random stream.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_m5_evaluation as m5eval

N_STABILITY_SEEDS = 8
PLANNER_SEED_BASES = (0, 1, 2, 3, 4)
STOCKOUT_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
HOLDING_RATES = [0.15, 0.20, 0.25, 0.35]
SENSITIVITY_SAMPLE_SEED = 0  # the headline sample, per research/M5_EVALUATION.md


def run_stability(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id):
    rows = []
    for seed in range(N_STABILITY_SEEDS):
        t0 = time.time()
        out = m5eval.evaluate_sample(seed, meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id)
        rows.append({
            "sample_seed": seed,
            "n_series": out["n_series"],
            "coverage_at_90": out["conformal_coverage"]["0.9"]["mean_empirical_coverage"],
            "vs_naive_pct": out["mean_cost_change_pct_aggregate"],
            "vs_sba_pct": out["mean_cost_change_pct_aggregate_vs_sba"],
            "vs_ss_pct": out["mean_cost_change_pct_aggregate_vs_ss"],
            "vs_ml_pct": out["mean_cost_change_pct_aggregate_vs_ml"],
            "n_fused_better_vs_naive": out["n_fused_better"],
            "n_fused_better_vs_sba": out["n_fused_better_vs_sba"],
            "n_fused_better_vs_ss": out["n_fused_better_vs_ss"],
            "n_fused_better_vs_ml": out["n_fused_better_vs_ml"],
        })
        print(f"  seed={seed} coverage90={rows[-1]['coverage_at_90']:.3f} "
              f"vs_naive={rows[-1]['vs_naive_pct']:.1f}% vs_sba={rows[-1]['vs_sba_pct']:.1f}% "
              f"vs_ss={rows[-1]['vs_ss_pct']:.1f}% vs_ml={rows[-1]['vs_ml_pct']:.1f}% "
              f"({time.time()-t0:.1f}s)")
    vs_naive = np.array([r["vs_naive_pct"] for r in rows])
    vs_sba = np.array([r["vs_sba_pct"] for r in rows])
    vs_ss = np.array([r["vs_ss_pct"] for r in rows])
    vs_ml = np.array([r["vs_ml_pct"] for r in rows])

    def _stats(arr):
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    summary = {
        "n_seeds": N_STABILITY_SEEDS,
        "per_seed": rows,
        "vs_naive_pct": _stats(vs_naive),
        "vs_sba_pct": _stats(vs_sba),
        "vs_ss_pct": _stats(vs_ss),
        "vs_ml_pct": _stats(vs_ml),
    }
    return summary


def run_sensitivity(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id):
    rows = []
    base_holding_rate = m5eval.ANNUAL_HOLDING_RATE
    for mult in STOCKOUT_MULTIPLIERS:
        t0 = time.time()
        out = m5eval.evaluate_sample(
            SENSITIVITY_SAMPLE_SEED, meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id,
            holding_rate=base_holding_rate, stockout_multiplier=mult, check_coverage=False,
        )
        rows.append({
            "axis": "stockout_multiplier", "value": mult, "holding_rate": base_holding_rate,
            "vs_naive_pct": out["mean_cost_change_pct_aggregate"],
            "vs_sba_pct": out["mean_cost_change_pct_aggregate_vs_sba"],
            "vs_ss_pct": out["mean_cost_change_pct_aggregate_vs_ss"],
            "vs_ml_pct": out["mean_cost_change_pct_aggregate_vs_ml"],
            "n_fused_better_vs_sba": out["n_fused_better_vs_sba"], "n_series": out["n_series"],
        })
        print(f"  stockout_mult={mult} vs_naive={rows[-1]['vs_naive_pct']:.1f}% "
              f"vs_sba={rows[-1]['vs_sba_pct']:.1f}% vs_ss={rows[-1]['vs_ss_pct']:.1f}% "
              f"vs_ml={rows[-1]['vs_ml_pct']:.1f}% ({time.time()-t0:.1f}s)")

    for rate in HOLDING_RATES:
        t0 = time.time()
        out = m5eval.evaluate_sample(
            SENSITIVITY_SAMPLE_SEED, meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id,
            holding_rate=rate, stockout_multiplier=1.0, check_coverage=False,
        )
        rows.append({
            "axis": "holding_rate", "value": rate, "stockout_multiplier": 1.0,
            "vs_naive_pct": out["mean_cost_change_pct_aggregate"],
            "vs_sba_pct": out["mean_cost_change_pct_aggregate_vs_sba"],
            "vs_ss_pct": out["mean_cost_change_pct_aggregate_vs_ss"],
            "vs_ml_pct": out["mean_cost_change_pct_aggregate_vs_ml"],
            "n_fused_better_vs_sba": out["n_fused_better_vs_sba"], "n_series": out["n_series"],
        })
        print(f"  holding_rate={rate} vs_naive={rows[-1]['vs_naive_pct']:.1f}% "
              f"vs_sba={rows[-1]['vs_sba_pct']:.1f}% vs_ss={rows[-1]['vs_ss_pct']:.1f}% "
              f"vs_ml={rows[-1]['vs_ml_pct']:.1f}% ({time.time()-t0:.1f}s)")

    return rows


def run_opening_inventory_sensitivity(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id):
    """Pair the shared Gaussian opening state against a common zero start."""
    primary = m5eval.evaluate_sample(
        SENSITIVITY_SAMPLE_SEED, meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id,
        check_coverage=False, opening_inventory_mode="gaussian_target", planner_seed_base=0,
    )
    zero = m5eval.evaluate_sample(
        SENSITIVITY_SAMPLE_SEED, meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id,
        check_coverage=False, opening_inventory_mode="zero", planner_seed_base=0,
    )
    primary_by_id = {r["series_id"]: r for r in primary["per_series"]}
    zero_by_id = {r["series_id"]: r for r in zero["per_series"]}
    policy_fields = {
        "naive": "baseline_cost",
        "sba": "sba_cost",
        "ss": "ss_cost",
        "ml": "ml_cost",
        "fused": "fused_cost",
    }
    comparisons = {}
    for policy, field in policy_fields.items():
        primary_total = float(sum(r[field] for r in primary["per_series"]))
        zero_total = float(sum(r[field] for r in zero["per_series"]))
        paired_deltas = [zero_by_id[sid][field] - primary_by_id[sid][field] for sid in primary_by_id]
        comparisons[policy] = {
            "gaussian_target_total": primary_total,
            "zero_start_total": zero_total,
            "zero_start_change_pct_vs_gaussian": float(100.0 * (zero_total / primary_total - 1.0)) if primary_total else 0.0,
            "n_series_zero_start_lower_cost": int(sum(delta < 0 for delta in paired_deltas)),
            "n_series_paired": len(paired_deltas),
        }
    return {
        "sample_seed": SENSITIVITY_SAMPLE_SEED,
        "planner_seed_base": 0,
        "n_series": zero["n_series"],
        "opening_inventory_modes": ["gaussian_target", "zero"],
        "paired_policy_costs": comparisons,
    }


def run_planner_seed_variability(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id):
    """Vary MCTS randomness while holding sample membership fixed at seed 0."""
    rows = []
    for seed_base in PLANNER_SEED_BASES:
        t0 = time.time()
        out = m5eval.evaluate_sample(
            SENSITIVITY_SAMPLE_SEED, meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id,
            check_coverage=False, opening_inventory_mode="gaussian_target", planner_seed_base=seed_base,
        )
        rows.append({
            "planner_seed_base": seed_base,
            "sample_seed": SENSITIVITY_SAMPLE_SEED,
            "n_series": out["n_series"],
            "fused_cost_total": out["fused_cost_total"],
            "vs_naive_pct": out["mean_cost_change_pct_aggregate"],
            "vs_sba_pct": out["mean_cost_change_pct_aggregate_vs_sba"],
            "vs_ss_pct": out["mean_cost_change_pct_aggregate_vs_ss"],
            "vs_ml_pct": out["mean_cost_change_pct_aggregate_vs_ml"],
            "n_fused_better_vs_ml": out["n_fused_better_vs_ml"],
        })
        print(f"  planner_seed_base={seed_base} fused={rows[-1]['fused_cost_total']:.1f} "
              f"vs_ml={rows[-1]['vs_ml_pct']:.1f}% ({time.time()-t0:.1f}s)")

    def _stats(key):
        arr = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

    return {
        "sample_seed_held_fixed": SENSITIVITY_SAMPLE_SEED,
        "n_planner_seed_bases": len(PLANNER_SEED_BASES),
        "per_seed": rows,
        "vs_naive_pct": _stats("vs_naive_pct"),
        "vs_sba_pct": _stats("vs_sba_pct"),
        "vs_ss_pct": _stats("vs_ss_pct"),
        "vs_ml_pct": _stats("vs_ml_pct"),
        "fused_cost_total": _stats("fused_cost_total"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run M5 sample, cost, opening-state, and planner-seed sensitivity analyses")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id = m5eval.load_shared_data()

    print("=== Sample-to-sample stability (8 stratified draws) ===")
    stability = run_stability(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id)
    print(f"vs_naive: mean={stability['vs_naive_pct']['mean']:.1f}% "
          f"std={stability['vs_naive_pct']['std']:.1f}% "
          f"range=[{stability['vs_naive_pct']['min']:.1f}%, {stability['vs_naive_pct']['max']:.1f}%]")
    print(f"vs_sba:   mean={stability['vs_sba_pct']['mean']:.1f}% "
          f"std={stability['vs_sba_pct']['std']:.1f}% "
          f"range=[{stability['vs_sba_pct']['min']:.1f}%, {stability['vs_sba_pct']['max']:.1f}%]")
    print(f"vs_ss:    mean={stability['vs_ss_pct']['mean']:.1f}% "
          f"std={stability['vs_ss_pct']['std']:.1f}% "
          f"range=[{stability['vs_ss_pct']['min']:.1f}%, {stability['vs_ss_pct']['max']:.1f}%]")
    print(f"vs_ml:    mean={stability['vs_ml_pct']['mean']:.1f}% "
          f"std={stability['vs_ml_pct']['std']:.1f}% "
          f"range=[{stability['vs_ml_pct']['min']:.1f}%, {stability['vs_ml_pct']['max']:.1f}%]")

    print("\n=== Opening-inventory sensitivity (same sample, seed=0) ===")
    opening_inventory = run_opening_inventory_sensitivity(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id)
    for policy, values in opening_inventory["paired_policy_costs"].items():
        print(f"  {policy}: zero-start change={values['zero_start_change_pct_vs_gaussian']:.1f}% "
              f"({values['n_series_zero_start_lower_cost']}/{values['n_series_paired']} series lower)")

    print("\n=== Planner-seed variability (sample seed fixed at 0) ===")
    planner_seeds = run_planner_seed_variability(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id)
    print(f"  fused vs ML: mean={planner_seeds['vs_ml_pct']['mean']:.1f}% "
          f"std={planner_seeds['vs_ml_pct']['std']:.1f}% "
          f"range=[{planner_seeds['vs_ml_pct']['min']:.1f}%, {planner_seeds['vs_ml_pct']['max']:.1f}%]")

    print("\n=== Cost-model sensitivity (headline sample, seed=0) ===")
    sensitivity = run_sensitivity(meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id)

    output = {
        "stability": stability,
        "sensitivity": sensitivity,
        "opening_inventory_sensitivity": opening_inventory,
        "planner_seed_variability": planner_seeds,
        "stratification_zero_fraction_days_end_exclusive": m5eval.TEST_START,
        "primary_opening_inventory_mode": "gaussian_target",
    }
    out_path = args.output or (Path(__file__).resolve().parent.parent / "research" / "m5_robustness_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
