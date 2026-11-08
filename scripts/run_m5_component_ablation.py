"""Run matched component ablations for the corrected M5 evaluation.

Each variant changes one policy component while retaining the same sample,
opening inventory, realized demand, costs, and per-series planner seed as the
headline run. The ACI ablation sets gamma to zero while retaining periodic
forecast and calibration refits. The horizon ablation retains root-action
MCTS but removes future-period rollout costs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_m5_evaluation as evaluation


VARIANTS = {
    "fuzzy_disabled": {"fuzzy_enabled": False},
    "aci_disabled": {"aci_gamma": 0.0},
    "one_period_horizon": {"mcts_horizon_periods": 1},
}


def _read_baseline(path: Path, sample_seed: int, n_per_stratum: int, integer_order_units: bool) -> dict:
    with path.open() as source:
        baseline = json.load(source)
    config = baseline.get("config", {})
    expected = {
        "sample_seed": sample_seed,
        "n_per_stratum": n_per_stratum,
        "opening_inventory_mode": "gaussian_target",
        "planner_seed_base": 0,
        "integer_order_units": integer_order_units,
    }
    actual = {
        "sample_seed": baseline.get("sample_seed"),
        "n_per_stratum": config.get("n_per_stratum"),
        "opening_inventory_mode": baseline.get("opening_inventory_mode"),
        "planner_seed_base": baseline.get("planner_seed_base"),
        "integer_order_units": config.get("integer_order_units"),
    }
    if actual != expected:
        raise ValueError(f"Headline baseline does not match the ablation design: {actual} != {expected}")
    return baseline


def _paired_cost_change(variant: dict, baseline: dict) -> dict:
    baseline_by_id = {row["series_id"]: row for row in baseline["per_series"]}
    variant_by_id = {row["series_id"]: row for row in variant["per_series"]}
    if baseline_by_id.keys() != variant_by_id.keys():
        raise ValueError("Ablation and headline baseline have different evaluated series")
    base_costs = np.array([baseline_by_id[sid]["fused_cost"] for sid in baseline_by_id], dtype=float)
    variant_costs = np.array([variant_by_id[sid]["fused_cost"] for sid in baseline_by_id], dtype=float)
    cost_reduction = 100.0 * (1.0 - sum(variant_costs) / sum(base_costs))
    paired = evaluation.significance_test(
        variant_costs, base_costs, "ablated fused policy vs full fused policy"
    )
    return {
        "headline_fused_cost_total": float(sum(base_costs)),
        "ablated_fused_cost_total": float(sum(variant_costs)),
        "aggregate_cost_reduction_pct_vs_full": float(cost_reduction),
        "n_series_ablated_policy_lower_cost": int(sum(a < b for a, b in zip(variant_costs, base_costs))),
        "paired_comparison": paired,
        "coverage": variant["conformal_coverage"] or baseline["conformal_coverage"],
        "coverage_source": "recomputed" if variant["conformal_coverage"] else "unchanged from headline calibrator",
        "cost_totals": {
            "gaussian_order_up_to": variant["baseline_cost_total"],
            "sba_croston": variant["sba_cost_total"],
            "tuned_s_s": variant["ss_cost_total"],
            "pooled_lightgbm": variant["ml_cost_total"],
            "fused": variant["fused_cost_total"],
        },
        "per_series": variant["per_series"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched M5 policy-component ablations")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--n-per-stratum", type=int, default=evaluation.N_PER_STRATUM)
    parser.add_argument("--integer-order-units", action="store_true")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline_path = args.baseline or PROJECT_ROOT / "research" / (
        "m5_integer_order_sensitivity.json" if args.integer_order_units else "m5_evaluation_results.json"
    )
    output_path = args.output or PROJECT_ROOT / "research" / (
        "m5_integer_component_ablation_results.json" if args.integer_order_units
        else "m5_component_ablation_results.json"
    )
    baseline = _read_baseline(
        baseline_path, args.sample_seed, args.n_per_stratum, args.integer_order_units
    )
    try:
        baseline_label = baseline_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        baseline_label = baseline_path.name
    meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id = evaluation.load_shared_data()

    output = {
        "design": {
            "sample_seed": args.sample_seed,
            "n_per_stratum": args.n_per_stratum,
            "n_strata": 27,
            "opening_inventory_mode": "gaussian_target",
            "planner_seed_base": 0,
            "integer_order_units": args.integer_order_units,
            "holding_rate": evaluation.ANNUAL_HOLDING_RATE,
            "stockout_multiplier": 1.0,
            "test_days": evaluation.TEST_DAYS,
            "review_period_days": evaluation.REVIEW_PERIOD_DAYS,
            "baseline_result": baseline_label,
            "paired_unit": "item-store series",
            "variants": {
                "fuzzy_disabled": "Set fuzzy risk multiplier to 1.0; retain conformal intervals and 3-period root-action MCTS.",
                "aci_disabled": "Set ACI gamma to zero; retain rolling forecast/calibration refits and fuzzy risk adjustment.",
                "one_period_horizon": "Use a one-period root-action MCTS horizon; retain fuzzy risk adjustment and rolling ACI.",
            },
        },
        "headline_reference": {
            "fused_cost_total": baseline["fused_cost_total"],
            "conformal_coverage": baseline["conformal_coverage"],
        },
        "ablations": {},
    }

    for name, parameters in VARIANTS.items():
        started = time.time()
        variant = evaluation.evaluate_sample(
            args.sample_seed,
            meta,
            zfracs,
            prices,
            prices_series,
            wm_yr_wk,
            event_days,
            meta_by_id,
            n_per_stratum=args.n_per_stratum,
            check_coverage=(name == "aci_disabled"),
            opening_inventory_mode="gaussian_target",
            planner_seed_base=0,
            integer_order_units=args.integer_order_units,
            **parameters,
        )
        output["ablations"][name] = _paired_cost_change(variant, baseline)
        output["ablations"][name]["runtime_seconds"] = time.time() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as destination:
            json.dump(output, destination, indent=2)
        print(
            f"{name}: fused={variant['fused_cost_total']:.1f}; "
            f"cost reduction vs full={output['ablations'][name]['aggregate_cost_reduction_pct_vs_full']:.2f}%; "
            f"coverage@90={output['ablations'][name]['coverage']['0.9']['mean_empirical_coverage']:.3f}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as destination:
        json.dump(output, destination, indent=2)
    print(f"Saved matched ablations to {output_path}")


if __name__ == "__main__":
    main()
