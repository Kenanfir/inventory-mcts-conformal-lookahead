#!/usr/bin/env python3
"""Validate and summarize a complete M5 inventory-policy evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "m5_full_catalog_evaluation.json"
DEFAULT_OUTPUT = ROOT / "research" / "m5_full_catalog_summary.json"
POLICIES = {
    "gaussian_order_up_to": "baseline_cost",
    "sba_croston": "sba_cost",
    "tuned_s_S": "ss_cost",
    "pooled_lightgbm": "ml_cost",
    "mcts_lookahead": "fused_cost",
}


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    """Return census-level descriptive statistics after completeness checks."""
    rows = payload.get("per_series")
    strata = payload.get("strata_report")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation must contain non-empty per_series records")
    if not isinstance(strata, list) or not strata:
        raise ValueError("evaluation must contain a non-empty strata_report")
    if any(int(s["n_available"]) != int(s["n_picked"]) for s in strata):
        raise ValueError("not a full-catalog census: at least one stratum was sampled")
    if sum(int(s["n_picked"]) for s in strata) != len(rows):
        raise ValueError("stratum counts do not match the number of per-series records")
    if int(payload.get("n_evaluated", -1)) != len(rows):
        raise ValueError("n_evaluated does not match the per-series records")
    if int(payload.get("n_missing_price", -1)) != 0:
        raise ValueError("full-catalog comparison requires a price for every series")
    ids = [row.get("series_id") for row in rows]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("series_id values must be present and unique")

    totals = {
        name: float(sum(float(row[key]) for row in rows))
        for name, key in POLICIES.items()
    }
    lookahead = totals["mcts_lookahead"]
    comparisons = {}
    for name, total in totals.items():
        if name == "mcts_lookahead":
            continue
        wins = sum(float(row["fused_cost"]) < float(row[POLICIES[name]]) for row in rows)
        comparisons[name] = {
            "aggregate_cost_reduction_pct": 100.0 * (1.0 - lookahead / total) if total else None,
            "series_won": wins,
            "series_count": len(rows),
            "series_win_share": wins / len(rows),
        }

    def grouped(field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result: dict[str, dict[str, Any]] = {}
        for label, group_rows in sorted(groups.items()):
            mcts_total = sum(float(row["fused_cost"]) for row in group_rows)
            ml_total = sum(float(row["ml_cost"]) for row in group_rows)
            wins = sum(float(row["fused_cost"]) < float(row["ml_cost"]) for row in group_rows)
            result[label] = {
                "series_count": len(group_rows),
                "mcts_cost": mcts_total,
                "lightgbm_cost": ml_total,
                "mcts_cost_reduction_vs_lightgbm_pct": 100.0 * (1.0 - mcts_total / ml_total) if ml_total else None,
                "mcts_series_won_vs_lightgbm": wins,
                "mcts_series_win_share_vs_lightgbm": wins / len(group_rows),
            }
        return result

    coverage_levels = sorted({
        level for row in rows for level in row.get("coverage_by_level", {})
    })
    coverage = {}
    for level in coverage_levels:
        values = [float(row["coverage_by_level"][level]["empirical_coverage"]) for row in rows]
        coverage[level] = {
            "nominal": float(level),
            "mean_series_coverage": float(np.mean(values)),
            "sd_across_series": float(np.std(values, ddof=0)),
        }

    config = payload.get("config", {})
    ratios = [float(row["fused_cost"]) / float(row["ml_cost"]) for row in rows if float(row["ml_cost"]) > 0]
    return {
        "analysis": "complete M5 benchmark census; descriptive finite-benchmark results",
        "series_count": len(rows),
        "strata_count": len(strata),
        "all_available_series_selected": True,
        "missing_price_series": 0,
        "config": config,
        "aggregate_costs": totals,
        "mcts_comparisons": comparisons,
        "coverage": coverage,
        "mcts_to_lightgbm_per_series_cost_ratio_percentiles": {
            str(q): float(np.percentile(ratios, q)) for q in (10, 25, 50, 75, 90)
        },
        "by_category": grouped("cat_id"),
        "by_state": grouped("state_id"),
        "by_store": grouped("store_id"),
        "inference_note": (
            "No series-level p-values or confidence intervals are interpreted: "
            "this is a census of the released M5 benchmark, whose series share "
            "stores, calendar shocks, and pooled model fits. Results remain "
            "conditional on one planner-seed base and the stated price-proxy simulator."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ml = summary["mcts_comparisons"]["pooled_lightgbm"]
    print(
        f"series={summary['series_count']} strata={summary['strata_count']} "
        f"MCTS-vs-LightGBM-cost-change={-ml['aggregate_cost_reduction_pct']:.1f}% "
        f"MCTS-wins={ml['series_won']}/{ml['series_count']}"
    )
    print(f"summary={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
