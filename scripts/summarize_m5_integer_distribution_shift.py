"""Summarize matched whole-unit shock runs and the incremental ACI effect."""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_m5_evaluation as evaluation

RESEARCH = PROJECT_ROOT / "research"
SEEDS = (0, 1, 4, 7)
SEGMENTS = ("pre", "post_pre_refit", "post_after_first_refit", "post_late")


def result_path(label: str, seed: int) -> Path:
    suffix = "" if seed == 0 else f"_seed{seed}"
    return RESEARCH / f"m5_integer_distribution_shift_{label}{suffix}.json"


def read_result(label: str, seed: int) -> dict:
    result = json.loads(result_path(label, seed).read_text())
    expected_gamma = 0.0 if label == "aci_disabled" else evaluation.ACI_GAMMA
    if (
        result.get("sample_seed") != seed
        or result.get("opening_inventory_mode") != "gaussian_target"
        or result.get("planner_seed_base") != 0
        or result.get("integer_order_units") is not True
        or result.get("aci_gamma") != expected_gamma
        or result.get("n_series") != 324
        or "per_series_segment_costs" not in result
    ):
        raise ValueError(f"Unexpected matched whole-unit shock result: {result_path(label, seed)}")
    return result


def main() -> None:
    rows = []
    for seed in SEEDS:
        full = read_result("full", seed)
        no_aci = read_result("aci_disabled", seed)
        if full["evaluated_series_ids"] != no_aci["evaluated_series_ids"]:
            raise ValueError(f"Full ACI and no-ACI runs differ in evaluated series for seed {seed}")

        for segment in SEGMENTS:
            full_fused = np.asarray(full["per_series_segment_costs"][segment]["fused"], dtype=float)
            no_aci_fused = np.asarray(no_aci["per_series_segment_costs"][segment]["fused"], dtype=float)
            full_ml_total = full["segments"][segment]["ml"]["total"]
            no_aci_ml_total = no_aci["segments"][segment]["ml"]["total"]
            full_fused_total = full["segments"][segment]["fused"]["total"]
            no_aci_fused_total = no_aci["segments"][segment]["fused"]["total"]
            full_reduction = 100.0 * (1.0 - full_fused_total / full_ml_total) if full_ml_total else 0.0
            no_aci_reduction = 100.0 * (1.0 - no_aci_fused_total / no_aci_ml_total) if no_aci_ml_total else 0.0
            rows.append({
                "sample_seed": seed,
                "segment": segment,
                "segment_days": full["segment_days"][segment],
                "full_aci_fused_vs_ml_cost_reduction_pct": full_reduction,
                "aci_disabled_fused_vs_ml_cost_reduction_pct": no_aci_reduction,
                "incremental_aci_effect_on_reduction_percentage_points": full_reduction - no_aci_reduction,
                "full_aci_fused_cost": full_fused_total,
                "aci_disabled_fused_cost": no_aci_fused_total,
                "aci_disabled_cost_reduction_vs_full_pct": 100.0 * (1.0 - no_aci_fused_total / full_fused_total) if full_fused_total else 0.0,
                "paired_no_aci_vs_full_aci": evaluation.significance_test(
                    no_aci_fused, full_fused, f"no ACI vs full ACI, {segment}, seed {seed}"
                ),
            })

    summaries = {}
    for segment in SEGMENTS:
        segment_rows = [row for row in rows if row["segment"] == segment]
        summaries[segment] = {}
        for field in (
            "full_aci_fused_vs_ml_cost_reduction_pct",
            "aci_disabled_fused_vs_ml_cost_reduction_pct",
            "incremental_aci_effect_on_reduction_percentage_points",
        ):
            values = np.asarray([row[field] for row in segment_rows], dtype=float)
            summaries[segment][field] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    output = {
        "design": {
            "sample_seeds": list(SEEDS),
            "integer_order_units": True,
            "opening_inventory_mode": "gaussian_target",
            "planner_seed_base": 0,
            "shock_multiplier": 1.8,
            "shock_day_offset": 189,
            "aci_disabled_definition": "gamma=0 with the same rolling refits retained",
            "inference_scope": "paired item-store comparisons within each 324-series sample; not full-catalog inference",
        },
        "per_sample_segment": rows,
        "summary_by_segment": summaries,
    }
    destination = RESEARCH / "m5_integer_distribution_shift_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
