"""Run whole-unit cost, opening-state, and planner-seed sensitivity analyses.

The continuous-order robustness sweep does not establish robustness for the
whole-unit policy used as the primary specification. This script repeats its
cost scenarios under the same seed-0 stratified sample and common Gaussian
opening target, then checks a zero start and planner-seed variation. It reuses
the already stored whole-unit seed-0 headline for the default-cost reference.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas
import scipy
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_m5_evaluation as evaluation


SAMPLE_SEED = 0
N_PER_STRATUM = evaluation.N_PER_STRATUM
PLANNER_SEED_BASES = (0, 1, 2, 3, 4)
STOCKOUT_MULTIPLIERS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
HOLDING_RATES = (0.15, 0.20, 0.25, 0.35)
BASELINE_PATH = PROJECT_ROOT / "research" / "m5_integer_order_sensitivity.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "research" / "m5_integer_order_sensitivity_analysis.json"

POLICY_FIELDS = {
    "gaussian_order_up_to": "baseline_cost",
    "sba_croston": "sba_cost",
    "tuned_s_s": "ss_cost",
    "pooled_lightgbm": "ml_cost",
    "monte_carlo_lookahead": "fused_cost",
}


def read_baseline() -> dict:
    result = json.loads(BASELINE_PATH.read_text())
    config = result.get("config", {})
    expected = {
        "sample_seed": SAMPLE_SEED,
        "n_per_stratum": N_PER_STRATUM,
        "opening_inventory_mode": "gaussian_target",
        "planner_seed_base": 0,
        "integer_order_units": True,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected or result.get("n_evaluated") != 324:
        raise ValueError(f"Whole-unit seed-0 reference does not match sensitivity design: {actual}")
    return result


def compact_result(result: dict) -> dict:
    return {
        "n_evaluated": result["n_evaluated"],
        "cost_totals": {
            "gaussian_order_up_to": result["baseline_cost_total"],
            "sba_croston": result["sba_cost_total"],
            "tuned_s_s": result["ss_cost_total"],
            "pooled_lightgbm": result["ml_cost_total"],
            "monte_carlo_lookahead": result["fused_cost_total"],
        },
        "cost_reduction_pct": {
            "vs_gaussian": result["mean_cost_change_pct_aggregate"],
            "vs_sba": result["mean_cost_change_pct_aggregate_vs_sba"],
            "vs_tuned_s_s": result["mean_cost_change_pct_aggregate_vs_ss"],
            "vs_pooled_lightgbm": result["mean_cost_change_pct_aggregate_vs_ml"],
        },
        "n_lookahead_lower_cost": {
            "vs_gaussian": result["n_fused_better"],
            "vs_sba": result["n_fused_better_vs_sba"],
            "vs_tuned_s_s": result["n_fused_better_vs_ss"],
            "vs_pooled_lightgbm": result["n_fused_better_vs_ml"],
        },
        "paired_comparisons": result["significance"][:4],
        "series_costs": [
            {
                "series_id": row["series_id"],
                **{name: row[field] for name, field in POLICY_FIELDS.items()},
            }
            for row in result["per_series"]
        ],
    }


def save_progress(path: Path, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(path)


def run_integer_sensitivity(output_path: Path) -> dict:
    baseline = read_baseline()
    meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id = evaluation.load_shared_data()

    output = {
        "status": "in_progress",
        "design": {
            "sample_seed": SAMPLE_SEED,
            "n_per_stratum": N_PER_STRATUM,
            "n_strata": 27,
            "opening_inventory_mode": "gaussian_target",
            "planner_seed_base": 0,
            "integer_order_units": True,
            "software_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pandas": pandas.__version__,
                "lightgbm": lightgbm.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "baseline_result": "research/m5_integer_order_sensitivity.json",
            "default_holding_rate": evaluation.ANNUAL_HOLDING_RATE,
            "default_stockout_multiplier": 1.0,
            "stockout_multipliers": list(STOCKOUT_MULTIPLIERS),
            "holding_rates": list(HOLDING_RATES),
            "paired_unit": "item-store series",
            "notes": [
                "Each cost scenario re-estimates the pooled quantile baseline at that scenario's critical ratio.",
                "The default-cost run reuses the matched seed-0 whole-unit headline result.",
                "Bootstrap intervals and paired tests are within this 324-series sample, not full-catalog inference.",
            ],
        },
        "cost_scenarios": [],
        "opening_inventory_sensitivity": None,
        "planner_seed_variability": None,
        "completed_runs": 0,
        "expected_new_runs": 13,
    }
    default_row = {
        "axis": "reference",
        "value": "headline",
        "holding_rate": evaluation.ANNUAL_HOLDING_RATE,
        "stockout_multiplier": 1.0,
        **compact_result(baseline),
        "source": "research/m5_integer_order_sensitivity.json",
    }
    output["cost_scenarios"].append(default_row)
    save_progress(output_path, output)

    scenarios = []
    for multiplier in STOCKOUT_MULTIPLIERS:
        if multiplier == 1.0:
            continue
        scenarios.append(("stockout_multiplier", multiplier, evaluation.ANNUAL_HOLDING_RATE, multiplier))
    for rate in HOLDING_RATES:
        if rate == evaluation.ANNUAL_HOLDING_RATE:
            continue
        scenarios.append(("holding_rate", rate, rate, 1.0))

    output["expected_new_runs"] = len(scenarios) + 1 + (len(PLANNER_SEED_BASES) - 1)
    for axis, value, holding_rate, multiplier in scenarios:
        started = time.time()
        result = evaluation.evaluate_sample(
            SAMPLE_SEED,
            meta,
            zfracs,
            prices,
            prices_series,
            wm_yr_wk,
            event_days,
            meta_by_id,
            n_per_stratum=N_PER_STRATUM,
            holding_rate=holding_rate,
            stockout_multiplier=multiplier,
            check_coverage=False,
            opening_inventory_mode="gaussian_target",
            planner_seed_base=0,
            integer_order_units=True,
        )
        row = {
            "axis": axis,
            "value": value,
            "holding_rate": holding_rate,
            "stockout_multiplier": multiplier,
            **compact_result(result),
            "runtime_seconds": time.time() - started,
        }
        output["cost_scenarios"].append(row)
        output["completed_runs"] += 1
        save_progress(output_path, output)
        print(
            f"{axis}={value}: lookahead vs Gaussian={row['cost_reduction_pct']['vs_gaussian']:.1f}%, "
            f"vs ML={row['cost_reduction_pct']['vs_pooled_lightgbm']:.1f}% "
            f"({row['runtime_seconds']:.1f}s)",
            flush=True,
        )

    # The zero-start comparison holds all other settings fixed to the shared
    # Gaussian-target seed-0 reference.
    started = time.time()
    zero = evaluation.evaluate_sample(
        SAMPLE_SEED,
        meta,
        zfracs,
        prices,
        prices_series,
        wm_yr_wk,
        event_days,
        meta_by_id,
        n_per_stratum=N_PER_STRATUM,
        check_coverage=False,
        opening_inventory_mode="zero",
        planner_seed_base=0,
        integer_order_units=True,
    )
    primary_by_id = {row["series_id"]: row for row in baseline["per_series"]}
    zero_by_id = {row["series_id"]: row for row in zero["per_series"]}
    if primary_by_id.keys() != zero_by_id.keys():
        raise ValueError("Zero-start and Gaussian-target runs do not share the same series")
    opening_comparison = {}
    for name, field in POLICY_FIELDS.items():
        primary_total = float(sum(row[field] for row in primary_by_id.values()))
        zero_total = float(sum(row[field] for row in zero_by_id.values()))
        deltas = [zero_by_id[sid][field] - primary_by_id[sid][field] for sid in primary_by_id]
        opening_comparison[name] = {
            "gaussian_target_total": primary_total,
            "zero_start_total": zero_total,
            "zero_start_change_pct_vs_gaussian": 100.0 * (zero_total / primary_total - 1.0) if primary_total else 0.0,
            "n_series_zero_start_lower_cost": int(sum(delta < 0 for delta in deltas)),
        }
    output["opening_inventory_sensitivity"] = {
        "sample_seed": SAMPLE_SEED,
        "planner_seed_base": 0,
        "integer_order_units": True,
        "paired_policy_costs": opening_comparison,
        "runtime_seconds": time.time() - started,
    }
    output["completed_runs"] += 1
    save_progress(output_path, output)
    print("zero-start sensitivity complete", flush=True)

    # Reuse planner seed 0 from the headline; vary only the seeded search
    # stream for the remaining runs.
    planner_rows = [{
        "planner_seed_base": 0,
        "fused_cost_total": baseline["fused_cost_total"],
        "vs_gaussian_pct": baseline["mean_cost_change_pct_aggregate"],
        "vs_sba_pct": baseline["mean_cost_change_pct_aggregate_vs_sba"],
        "vs_tuned_s_s_pct": baseline["mean_cost_change_pct_aggregate_vs_ss"],
        "vs_pooled_lightgbm_pct": baseline["mean_cost_change_pct_aggregate_vs_ml"],
        "n_lookahead_lower_cost_vs_ml": baseline["n_fused_better_vs_ml"],
        "source": "research/m5_integer_order_sensitivity.json",
    }]
    for seed_base in PLANNER_SEED_BASES[1:]:
        started = time.time()
        result = evaluation.evaluate_sample(
            SAMPLE_SEED,
            meta,
            zfracs,
            prices,
            prices_series,
            wm_yr_wk,
            event_days,
            meta_by_id,
            n_per_stratum=N_PER_STRATUM,
            check_coverage=False,
            opening_inventory_mode="gaussian_target",
            planner_seed_base=seed_base,
            integer_order_units=True,
        )
        row = {
            "planner_seed_base": seed_base,
            "fused_cost_total": result["fused_cost_total"],
            "vs_gaussian_pct": result["mean_cost_change_pct_aggregate"],
            "vs_sba_pct": result["mean_cost_change_pct_aggregate_vs_sba"],
            "vs_tuned_s_s_pct": result["mean_cost_change_pct_aggregate_vs_ss"],
            "vs_pooled_lightgbm_pct": result["mean_cost_change_pct_aggregate_vs_ml"],
            "n_lookahead_lower_cost_vs_ml": result["n_fused_better_vs_ml"],
            "runtime_seconds": time.time() - started,
        }
        planner_rows.append(row)
        output["completed_runs"] += 1
        output["planner_seed_variability"] = {"per_seed": planner_rows}
        save_progress(output_path, output)
        print(
            f"planner_seed_base={seed_base}: fused={row['fused_cost_total']:.1f}, "
            f"vs ML={row['vs_pooled_lightgbm_pct']:.1f}% ({row['runtime_seconds']:.1f}s)",
            flush=True,
        )

    planner_summary = {}
    for field in (
        "fused_cost_total",
        "vs_gaussian_pct",
        "vs_sba_pct",
        "vs_tuned_s_s_pct",
        "vs_pooled_lightgbm_pct",
    ):
        values = np.array([row[field] for row in planner_rows], dtype=float)
        planner_summary[field] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    output["planner_seed_variability"] = {
        "sample_seed_held_fixed": SAMPLE_SEED,
        "n_planner_seed_bases": len(PLANNER_SEED_BASES),
        "per_seed": planner_rows,
        "summary": planner_summary,
    }
    output["status"] = "complete"
    save_progress(output_path, output)
    print(f"Saved whole-unit sensitivity results to {output_path}")
    return output


def rerun_holding_rate_scenarios(output_path: Path) -> dict:
    """Replace the three invalid holding-rate rows without rerunning other checks."""
    if not output_path.exists():
        raise FileNotFoundError(f"Sensitivity results to update do not exist: {output_path}")
    output = json.loads(output_path.read_text())
    if output.get("status") != "complete" or not output.get("planner_seed_variability"):
        raise ValueError("The existing sensitivity file is not a completed full run")
    read_baseline()
    output["cost_scenarios"] = [
        row for row in output["cost_scenarios"] if row["axis"] != "holding_rate"
    ]
    output["status"] = "in_progress"
    output["completed_runs"] = 0
    output["expected_new_runs"] = len([rate for rate in HOLDING_RATES if rate != evaluation.ANNUAL_HOLDING_RATE])
    save_progress(output_path, output)

    meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id = evaluation.load_shared_data()
    for rate in HOLDING_RATES:
        if rate == evaluation.ANNUAL_HOLDING_RATE:
            continue
        started = time.time()
        result = evaluation.evaluate_sample(
            SAMPLE_SEED,
            meta,
            zfracs,
            prices,
            prices_series,
            wm_yr_wk,
            event_days,
            meta_by_id,
            n_per_stratum=N_PER_STRATUM,
            holding_rate=rate,
            stockout_multiplier=1.0,
            check_coverage=False,
            opening_inventory_mode="gaussian_target",
            planner_seed_base=0,
            integer_order_units=True,
        )
        row = {
            "axis": "holding_rate",
            "value": rate,
            "holding_rate": rate,
            "stockout_multiplier": 1.0,
            **compact_result(result),
            "runtime_seconds": time.time() - started,
        }
        output["cost_scenarios"].append(row)
        output["completed_runs"] += 1
        save_progress(output_path, output)
        print(
            f"holding_rate={rate}: lookahead vs Gaussian={row['cost_reduction_pct']['vs_gaussian']:.1f}%, "
            f"vs ML={row['cost_reduction_pct']['vs_pooled_lightgbm']:.1f}% "
            f"({row['runtime_seconds']:.1f}s)",
            flush=True,
        )

    output["status"] = "complete"
    output["completed_runs"] = 13
    output["expected_new_runs"] = 13
    output["correction_reruns"] = 3
    save_progress(output_path, output)
    print(f"Corrected holding-rate rows in {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run whole-unit-order M5 sensitivity analyses")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holding-rates-only", action="store_true")
    args = parser.parse_args()
    if args.holding_rates_only:
        rerun_holding_rate_scenarios(args.output)
    else:
        run_integer_sensitivity(args.output)


if __name__ == "__main__":
    main()
