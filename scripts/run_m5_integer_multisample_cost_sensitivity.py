"""Evaluate selected whole-unit cost scenarios across the eight M5 samples.

The primary cost sweep was first run on sample seed 0. This follow-up checks
whether the same policy rankings recur across seeds 0-7. Seed-0 results are
reused from the completed whole-unit sweep; all other sample/scenario pairs
are recomputed with the same opening-inventory target and planner seed.
"""

import concurrent.futures
import json
import os
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


SAMPLE_SEEDS = tuple(range(8))
SCENARIOS = (
    ("stockout_multiplier", 0.25, evaluation.ANNUAL_HOLDING_RATE, 0.25),
    ("stockout_multiplier", 0.50, evaluation.ANNUAL_HOLDING_RATE, 0.50),
    ("stockout_multiplier", 0.75, evaluation.ANNUAL_HOLDING_RATE, 0.75),
    ("stockout_multiplier", 2.00, evaluation.ANNUAL_HOLDING_RATE, 2.00),
    ("holding_rate", 0.15, 0.15, 1.00),
    ("holding_rate", 0.25, 0.25, 1.00),
    ("holding_rate", 0.35, 0.35, 1.00),
)
SENSITIVITY_PATH = PROJECT_ROOT / "research" / "m5_integer_order_sensitivity_analysis.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "research" / "m5_integer_multisample_cost_sensitivity.json"
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
_WORKER_DATA = None


def baseline_path(seed: int) -> Path:
    if seed == 0:
        return PROJECT_ROOT / "research" / "m5_integer_order_sensitivity.json"
    return PROJECT_ROOT / "research" / f"m5_integer_order_seed{seed}.json"


def validate_baseline(result: dict, seed: int) -> None:
    config = result.get("config", {})
    expected = {
        "sample_seed": seed,
        "n_per_stratum": evaluation.N_PER_STRATUM,
        "opening_inventory_mode": "gaussian_target",
        "planner_seed_base": 0,
        "integer_order_units": True,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected or result.get("n_evaluated") != 324:
        raise ValueError(f"Seed-{seed} baseline does not match this design: {actual}")


def compact(result: dict, seed: int, axis: str, value: float,
            holding_rate: float, stockout_multiplier: float,
            runtime_seconds: float, source: str) -> dict:
    return {
        "sample_seed": seed,
        "axis": axis,
        "value": value,
        "holding_rate": holding_rate,
        "stockout_multiplier": stockout_multiplier,
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
        "runtime_seconds": runtime_seconds,
        "source": source,
    }


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _init_worker() -> None:
    global _WORKER_DATA
    _WORKER_DATA = evaluation.load_shared_data()


def _evaluate_job(job: tuple) -> dict:
    axis, value, holding_rate, multiplier, seed = job
    if _WORKER_DATA is None:
        _init_worker()
    started = time.time()
    result = evaluation.evaluate_sample(
        seed,
        *_WORKER_DATA,
        n_per_stratum=evaluation.N_PER_STRATUM,
        holding_rate=holding_rate,
        stockout_multiplier=multiplier,
        check_coverage=False,
        opening_inventory_mode="gaussian_target",
        planner_seed_base=0,
        integer_order_units=True,
    )
    return compact(
        result, seed, axis, value, holding_rate, multiplier,
        time.time() - started, "rerun",
    )


def _row_key(row: dict) -> tuple:
    return row["axis"], float(row["value"]), int(row["sample_seed"])


def _ordered_rows(rows: list[dict]) -> list[dict]:
    lookup = {_row_key(row): row for row in rows}
    return [
        lookup[(axis, float(value), seed)]
        for axis, value, _, _ in SCENARIOS
        for seed in SAMPLE_SEEDS
        if (axis, float(value), seed) in lookup
    ]


def run(output_path: Path, max_workers: int = DEFAULT_WORKERS) -> dict:
    sensitivity = json.loads(SENSITIVITY_PATH.read_text())
    if sensitivity.get("status") != "complete":
        raise ValueError("The seed-0 whole-unit sensitivity file is not complete")
    seed0_rows = {
        (row["axis"], float(row["value"])): row
        for row in sensitivity["cost_scenarios"]
        if row["axis"] != "reference"
    }
    for seed in SAMPLE_SEEDS:
        path = baseline_path(seed)
        result = json.loads(path.read_text())
        validate_baseline(result, seed)

    design = {
        "sample_seeds": list(SAMPLE_SEEDS),
        "n_per_stratum": evaluation.N_PER_STRATUM,
        "opening_inventory_mode": "gaussian_target",
        "planner_seed_base": 0,
        "integer_order_units": True,
        "paired_unit": "item-store series within each sample",
        "cross_sample_summary": "descriptive mean and range over eight draws; no cross-draw inferential claim",
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "reused_seed0_sensitivity": "research/m5_integer_order_sensitivity_analysis.json",
        "scenarios": [
            {"axis": axis, "value": value, "holding_rate": rate,
             "stockout_multiplier": multiplier}
            for axis, value, rate, multiplier in SCENARIOS
        ],
    }
    expected_new_runs = (len(SAMPLE_SEEDS) - 1) * len(SCENARIOS)
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload.get("status") == "complete":
            return payload
        if payload.get("status") != "in_progress" or payload.get("design") != design:
            raise ValueError("Existing output is not an in-progress run of this design")
        rows = payload.get("rows", [])
    else:
        payload = {
            "status": "in_progress",
            "design": design,
            "rows": [],
            "completed_new_runs": 0,
            "expected_new_runs": expected_new_runs,
        }
        rows = payload["rows"]

    expected_keys = {
        (axis, float(value), seed)
        for axis, value, _, _ in SCENARIOS
        for seed in SAMPLE_SEEDS
    }
    existing_keys = [_row_key(row) for row in rows]
    if len(existing_keys) != len(set(existing_keys)) or not set(existing_keys) <= expected_keys:
        raise ValueError("Existing output has duplicate or unexpected sample-scenario rows")
    payload["rows"] = _ordered_rows(rows)
    payload["completed_new_runs"] = sum(row.get("source") == "rerun" for row in rows)
    payload["expected_new_runs"] = expected_new_runs
    payload["summary"] = []
    save(output_path, payload)

    for axis, value, holding_rate, multiplier in SCENARIOS:
        if (axis, float(value), 0) in {_row_key(row) for row in payload["rows"]}:
            continue
        seed0_row = seed0_rows[(axis, float(value))]
        payload["rows"].append({
            "sample_seed": 0,
            "axis": axis,
            "value": value,
            "holding_rate": holding_rate,
            "stockout_multiplier": multiplier,
            "n_evaluated": seed0_row["n_evaluated"],
            "cost_totals": seed0_row["cost_totals"],
            "cost_reduction_pct": seed0_row["cost_reduction_pct"],
            "n_lookahead_lower_cost": seed0_row["n_lookahead_lower_cost"],
            "paired_comparisons": seed0_row["paired_comparisons"],
            "runtime_seconds": 0.0,
            "source": "research/m5_integer_order_sensitivity_analysis.json",
        })
        payload["rows"] = _ordered_rows(payload["rows"])
        save(output_path, payload)
    completed_keys = {_row_key(row) for row in payload["rows"]}
    jobs = [
        (axis, value, holding_rate, multiplier, seed)
        for axis, value, holding_rate, multiplier in SCENARIOS
        for seed in SAMPLE_SEEDS[1:]
        if (axis, float(value), seed) not in completed_keys
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
    ) as executor:
        futures = {executor.submit(_evaluate_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            payload["rows"].append(row)
            payload["rows"] = _ordered_rows(payload["rows"])
            payload["completed_new_runs"] += 1
            save(output_path, payload)
            print(
                f"{row['axis']}={row['value']:g}, seed={row['sample_seed']}: "
                f"lookahead vs LightGBM "
                f"{row['cost_reduction_pct']['vs_pooled_lightgbm']:+.1f}% "
                f"({row['runtime_seconds']:.1f}s; "
                f"{payload['completed_new_runs']}/{expected_new_runs})",
                flush=True,
            )

    summaries = []
    for axis, value, _, _ in SCENARIOS:
        group = [row for row in payload["rows"]
                 if row["axis"] == axis and float(row["value"]) == float(value)]
        if len(group) != len(SAMPLE_SEEDS):
            raise ValueError(f"Incomplete scenario rows for {axis}={value:g}")
        comparisons = {}
        for key in ("vs_gaussian", "vs_sba", "vs_tuned_s_s", "vs_pooled_lightgbm"):
            values = [row["cost_reduction_pct"][key] for row in group]
            comparisons[key] = {
                "mean_pct": float(np.mean(values)),
                "min_pct": float(np.min(values)),
                "max_pct": float(np.max(values)),
                "n_draws_lookahead_lower_cost": int(sum(x > 0 for x in values)),
            }
        summaries.append({"axis": axis, "value": value, "n_draws": len(group),
                          "aggregate_cost_reduction": comparisons})
    payload["summary"] = summaries
    payload["status"] = "complete"
    save(output_path, payload)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    completed = run(args.output, args.workers)
    print(f"completed {completed['completed_new_runs']} new sample-scenario runs")
