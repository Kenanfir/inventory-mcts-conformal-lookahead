"""Summarize matched whole-unit-order M5 evaluation runs for sample seeds 0–7."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"


def read_run(seed: int) -> dict:
    path = RESEARCH / (
        "m5_integer_order_sensitivity.json" if seed == 0
        else f"m5_integer_order_seed{seed}.json"
    )
    result = json.loads(path.read_text())
    config = result.get("config", {})
    expected = {
        "sample_seed": seed,
        "opening_inventory_mode": "gaussian_target",
        "planner_seed_base": 0,
        "integer_order_units": True,
        "n_per_stratum": 12,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected or result.get("n_evaluated") != 324:
        raise ValueError(f"Unexpected integer-order run settings in {path}: {actual}")
    return result


def main() -> None:
    continuous = json.loads((RESEARCH / "m5_robustness_results.json").read_text())
    continuous_by_seed = {
        row["sample_seed"]: row for row in continuous["stability"]["per_seed"]
    }
    rows = []
    for seed in range(8):
        result = read_run(seed)
        reference = continuous_by_seed[seed]
        row = {
            "sample_seed": seed,
            "n_evaluated": result["n_evaluated"],
            "conformal_coverage_90": result["conformal_coverage"]["0.9"]["mean_empirical_coverage"],
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
            "n_series_lookahead_lower_cost": {
                "vs_gaussian": result["n_fused_better"],
                "vs_sba": result["n_fused_better_vs_sba"],
                "vs_tuned_s_s": result["n_fused_better_vs_ss"],
                "vs_pooled_lightgbm": result["n_fused_better_vs_ml"],
            },
            "continuous_order_cost_reduction_pct": {
                "vs_gaussian": reference["vs_naive_pct"],
                "vs_sba": reference["vs_sba_pct"],
                "vs_tuned_s_s": reference["vs_ss_pct"],
                "vs_pooled_lightgbm": reference["vs_ml_pct"],
            },
        }
        row["integer_minus_continuous_percentage_points"] = {
            key: row["cost_reduction_pct"][key] - row["continuous_order_cost_reduction_pct"][key]
            for key in row["cost_reduction_pct"]
        }
        rows.append(row)

    metrics = list(rows[0]["cost_reduction_pct"])
    summary = {}
    for metric in metrics:
        values = [row["cost_reduction_pct"][metric] for row in rows]
        summary[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "n_positive": sum(value > 0 for value in values),
            "n_negative": sum(value < 0 for value in values),
        }

    output = {
        "design": {
            "sample_seeds": list(range(8)),
            "n_per_stratum": 12,
            "opening_inventory_mode": "gaussian_target",
            "planner_seed_base": 0,
            "integer_order_units": True,
            "comparison_reference": "research/m5_robustness_results.json",
            "interpretation": "Descriptive variation across eight balanced sample draws; no pooled catalog-level inference.",
        },
        "summary": summary,
        "per_seed": rows,
    }
    destination = RESEARCH / "m5_integer_order_robustness_results.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
