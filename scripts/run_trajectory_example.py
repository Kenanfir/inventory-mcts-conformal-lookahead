"""
Instrumented re-run of two spot-checked series from the headline evaluation
(scripts/run_m5_evaluation.py, seed 0 sample), logging per-review-period
inventory level and demand rather than only the aggregate cost, so Section
4.3's face-validity discussion can be shown, not just described.

Rewritten 2026-09-23 (round 3, alongside the new (s,S)/ML baselines and the
MCTS action-space fix -- research/M5_EVALUATION.md) to call
run_m5_evaluation.run_one_series(..., record_trajectory=True) directly
instead of a separate, hand-duplicated simulation loop. The old duplicated
loop had silently drifted out of sync with the real evaluation (missing
the margin-anchored `extra_candidate_levels` MCTS candidates added this
round), which the code review discipline this project follows would flag
as exactly the kind of gap that makes a manuscript figure non-reproducible
from its own claimed source. Calling the same function the headline
evaluation calls is the fix, not just re-fixing the copy.

The worst-case and typical-win series ids are recomputed fresh from the
current research/m5_evaluation_results.json each run (not hardcoded), since
which series is worst-case for fused depends on the current MCTS tuning and
already changed once this round.
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import run_m5_evaluation as m5eval
import m5_loader


def _pick_series(rows):
    """Worst case: the series with the largest fused_cost - baseline_cost
    among series where fused loses. Typical win: the series whose
    fused/baseline cost ratio is closest to the median ratio among series
    fused wins on."""
    losers = [r for r in rows if r["fused_cost"] > r["baseline_cost"]]
    worst = max(losers, key=lambda r: r["fused_cost"] - r["baseline_cost"])

    winners = [r for r in rows if r["fused_cost"] < r["baseline_cost"] and r["baseline_cost"] > 0]
    win_ratios = [r["fused_cost"] / r["baseline_cost"] for r in winners]
    median_ratio = float(np.median(win_ratios))
    typical = min(winners, key=lambda r: abs((r["fused_cost"] / r["baseline_cost"]) - median_ratio))

    return worst["series_id"], typical["series_id"]


def main():
    parser = argparse.ArgumentParser(description="Reproduce representative policy trajectories")
    parser.add_argument("--integer-order-units", action="store_true")
    args = parser.parse_args()
    meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id = m5eval.load_shared_data()
    ids, _ = m5_loader.stratified_sample(meta, m5eval.N_PER_STRATUM, zfracs, seed=m5eval.SAMPLE_SEED)
    series_map = m5_loader.load_series_by_ids(ids)

    project_root = Path(__file__).resolve().parent.parent
    baseline_path = project_root / "research" / (
        "m5_integer_order_sensitivity.json" if args.integer_order_units else "m5_evaluation_results.json"
    )
    with baseline_path.open() as f:
        headline = json.load(f)
    rows = headline["per_series"]

    worst_case_id, typical_id = _pick_series(rows)
    print(f"Worst case: {worst_case_id}")
    print(f"Typical win: {typical_id}")

    # Pooled ML models, trained once across the whole sample exactly as
    # evaluate_sample does, so the ML baseline's trajectory here is the same
    # policy (same trained models) as in the headline run.
    ml_models, _ = m5eval.train_ml_models_by_epoch(series_map, meta_by_id, prices_series, wm_yr_wk, event_days)

    out = {}
    for sid in [worst_case_id, typical_id]:
        m = meta_by_id[sid]
        key = (m["store_id"], m["item_id"])
        price = prices[key]
        series = series_map[sid]
        r = m5eval.run_one_series(
            sid, series, event_days, price, seed=0,
            cat_id=m["cat_id"], state_id=m["state_id"], ml_models=ml_models,
            prices_series=prices_series, wm_yr_wk=wm_yr_wk, series_key=key,
            check_coverage=False, record_trajectory=True,
            planner_seed_base=0, integer_order_units=args.integer_order_units,
        )
        test = series[m5eval.TEST_START:m5eval.TEST_START + m5eval.TEST_DAYS]
        out[sid] = {
            "series_id": sid,
            "price": price,
            "naive_cost": r["baseline_cost"],
            "fused_cost": r["fused_cost"],
            "ml_cost": r["ml_cost"],
            "trajectory": r["trajectory"],
            "test_demand_daily": test.tolist(),
        }
        print(f"  {sid}: naive={r['baseline_cost']:.1f} fused={r['fused_cost']:.1f} ml={r['ml_cost']:.1f}")

    out_path = project_root / "research" / (
        "trajectory_examples_integer_order.json" if args.integer_order_units else "trajectory_examples.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
