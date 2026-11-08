"""
Distribution-shift stress test: added 2026-09-23 after (1) the impact-
benchmark loop found the manuscript's baseline set too shallow and added a
tuned (s,S) and a pooled-ML-quantile baseline (research/
STRENGTHENING_OPTIONS.md), and (2) the fused method, having caught up to
and beaten those two on the STATIC 324-series M5 window, was still beaten
on raw average cost by the pooled ML baseline (research/M5_EVALUATION.md).
A follow-up literature pass found no inventory-specific precedent for
"loses average cost to a strong ML baseline, wins on other grounds," but
did find two independently-documented facts this test is built on:
  (a) M5's own 2011-2016 window is documented in the forecasting literature
      (Kolassa et al.; Hewamalage & Montero-Manso) as NOT representing a
      distribution shift/regime change -- so "evaluated on real M5 data"
      does not by itself support any claim about shift robustness, and
  (b) a genuine distribution-shift test is therefore new evidence, not
      redundant with the existing static-window comparison.

Design: for each series in the SAME stratified sample (seed=0) used by the
headline evaluation, inject a single demand-level shock partway through the
test window (SHOCK_DAY_OFFSET days into the 364-day test window) by
multiplying that series' remaining demand by SHOCK_MULTIPLIER. This is not
hidden from any policy -- every policy observes the shocked demand as it
happens, exactly as a real deployment would. This tests the complete
policy's response to a constructed change. The shock comparison alone does
not isolate ACI: the fused system's calibrator updates online, while its
point forecaster and the comparison baselines refit on a 28-day cadence,
and the policy also includes fuzzy risk adjustment and Monte Carlo
lookahead. A matched gamma=0 run assesses ACI's incremental contribution
while retaining periodic refits. Refits are aligned to absolute
dataset-day epochs, while decisions are made weekly. The shock is placed at
the nearest weekly review to the test midpoint that begins a full 28-day
interval before the next refit. The analysis reports this pre-refit
interval and the first 28 days after that refit separately.

Sample membership uses only pre-test zero-demand fractions through d_1577,
and every policy receives the same Gaussian-target opening inventory before
the test window. Planner streams use deterministic series-ID seeds so seed
comparisons do not depend on iteration order.

Only an upward shock is tested (SHOCK_MULTIPLIER > 1): M5's cost ratio
(~260:1, stockout costs dominate) makes an upward demand shift the more
consequential direction to get wrong quickly. A downward-shock/demand-
collapse scenario is a natural extension not run here -- noted as a scope
limitation, not silently assumed away.
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
import m5_loader

SAMPLE_SEED = 0
N_PER_STRATUM = m5eval.N_PER_STRATUM
SHOCK_DAY_OFFSET = 189  # nearest weekly review to midpoint with a full 28-day pre-refit window
SHOCK_MULTIPLIER = 1.8  # sustained +80% demand level from the shock day onward
POST_SHOCK_WINDOW_DAYS = m5eval.REFIT_PERIOD_DAYS
POLICIES = ("baseline", "sba", "ss", "ml", "fused")
SEGMENTS = ("pre", "post_pre_refit", "post_after_first_refit", "post_late")


def _shift_series_map(series_map, shock_abs_day, multiplier):
    """Return a new series_map where every series' demand from
    `shock_abs_day` onward is scaled by `multiplier` and rounded to the
    nearest non-negative integer count (M5 demand is unit sales). Days
    before `shock_abs_day` -- including all fit/calibration history -- are
    byte-for-byte unchanged, so the shock is genuinely unanticipated by any
    policy's pre-shock fitting."""
    shifted = {}
    for sid, series in series_map.items():
        s = series.copy()
        s[shock_abs_day:] = np.maximum(0.0, np.round(s[shock_abs_day:] * multiplier))
        shifted[sid] = s
    return shifted


def _segment_costs(period_entries, day_lo, day_hi):
    """Sum (holding, stockout) over trajectory dict entries with
    day_lo <= day < day_hi (see run_m5_evaluation.run_one_series's
    `record_trajectory` docstring for the entry format)."""
    hold = sum(e["holding"] for e in period_entries if day_lo <= e["day"] < day_hi)
    stock = sum(e["stockout"] for e in period_entries if day_lo <= e["day"] < day_hi)
    return hold, stock


def _first_refit_review_day_after_shock(shock_abs_day, test_end):
    """Return the first weekly review start after the shock that enters a
    new absolute-day refit epoch. The refit code uses `day // 28`, not days
    elapsed from the shock or from TEST_START.
    """
    shock_epoch = shock_abs_day // m5eval.REFIT_PERIOD_DAYS
    day = shock_abs_day + m5eval.REVIEW_PERIOD_DAYS
    while day < test_end:
        if day // m5eval.REFIT_PERIOD_DAYS != shock_epoch:
            return day
        day += m5eval.REVIEW_PERIOD_DAYS
    return test_end


def run(
    sample_seed=SAMPLE_SEED,
    n_per_stratum=N_PER_STRATUM,
    opening_inventory_mode="gaussian_target",
    planner_seed_base=0,
    aci_gamma=m5eval.ACI_GAMMA,
    integer_order_units=False,
    output_path=None,
):
    t0 = time.time()
    meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id = m5eval.load_shared_data()
    ids, strata_report = m5_loader.stratified_sample(meta, n_per_stratum, zfracs, seed=sample_seed)
    series_map = m5_loader.load_series_by_ids(ids)

    shock_abs_day = m5eval.TEST_START + SHOCK_DAY_OFFSET
    shifted_map = _shift_series_map(series_map, shock_abs_day, SHOCK_MULTIPLIER)
    test_end = m5eval.TEST_START + m5eval.TEST_DAYS
    if (shock_abs_day - m5eval.TEST_START) % m5eval.REVIEW_PERIOD_DAYS:
        raise ValueError("Shock day must align with a weekly review start")
    first_refit_day = _first_refit_review_day_after_shock(shock_abs_day, test_end)
    if first_refit_day - shock_abs_day != POST_SHOCK_WINDOW_DAYS:
        raise ValueError("Shock timing must leave one complete refit period before the next weekly refit")
    post_refit_window_end = min(first_refit_day + POST_SHOCK_WINDOW_DAYS, test_end)

    # Pooled ML models retrained on the SHIFTED data, on the identical
    # refit-epoch cadence as the headline evaluation -- a real deployment
    # would retrain on whatever demand actually happened, shock included.
    ml_models, ml_critical_ratio = m5eval.train_ml_models_by_epoch(
        shifted_map, meta_by_id, prices_series, wm_yr_wk, event_days,
    )

    bounds = {
        "pre": (m5eval.TEST_START, shock_abs_day),
        "post_pre_refit": (shock_abs_day, first_refit_day),
        "post_after_first_refit": (first_refit_day, post_refit_window_end),
        "post_late": (post_refit_window_end, test_end),
    }

    totals = {seg: {p: {"holding": 0.0, "stockout": 0.0} for p in POLICIES} for seg in SEGMENTS}
    per_series_seg_cost = {seg: {p: [] for p in POLICIES} for seg in SEGMENTS}
    evaluated_series_ids = []
    missing_price = []
    n_evaluated = 0

    for sid in ids:
        m = meta_by_id[sid]
        key = (m["store_id"], m["item_id"])
        price = prices.get(key)
        if price is None:
            missing_price.append(sid)
            continue
        series = shifted_map[sid]
        r = m5eval.run_one_series(
            sid, series, event_days, price, seed=0,
            cat_id=m["cat_id"], state_id=m["state_id"], ml_models=ml_models,
            prices_series=prices_series, wm_yr_wk=wm_yr_wk, series_key=key,
            check_coverage=False, record_trajectory=True,
            opening_inventory_mode=opening_inventory_mode, planner_seed_base=planner_seed_base,
            aci_gamma=aci_gamma,
            integer_order_units=integer_order_units,
        )
        n_evaluated += 1
        evaluated_series_ids.append(sid)
        for policy in POLICIES:
            traj = r["trajectory"][policy]
            for seg, (lo, hi) in bounds.items():
                hold, stock = _segment_costs(traj, lo, hi)
                totals[seg][policy]["holding"] += hold
                totals[seg][policy]["stockout"] += stock
                per_series_seg_cost[seg][policy].append(hold + stock)

    elapsed = time.time() - t0

    significance = {}
    for seg in SEGMENTS:
        fused_arr = np.array(per_series_seg_cost[seg]["fused"])
        ml_arr = np.array(per_series_seg_cost[seg]["ml"])
        significance[seg] = m5eval.significance_test(fused_arr, ml_arr, f"fused vs ml, {seg}")

    out = {
        "sample_seed": sample_seed,
        "n_per_stratum": n_per_stratum,
        "opening_inventory_mode": opening_inventory_mode,
        "planner_seed_base": planner_seed_base,
        "aci_gamma": aci_gamma,
        "integer_order_units": integer_order_units,
        "stratification_zero_fraction_days_end_exclusive": m5eval.TEST_START,
        "n_series": n_evaluated,
        "evaluated_series_ids": evaluated_series_ids,
        "n_missing_price": len(missing_price),
        "shock_day_offset": SHOCK_DAY_OFFSET,
        "shock_abs_day": shock_abs_day,
        "shock_multiplier": SHOCK_MULTIPLIER,
        "first_refit_review_day_after_shock": first_refit_day,
        "days_to_first_refit": first_refit_day - shock_abs_day,
        "post_shock_window_days": POST_SHOCK_WINDOW_DAYS,
        "segment_days": {seg: hi - lo for seg, (lo, hi) in bounds.items()},
        "segments": {
            seg: {
                policy: {
                    "holding": totals[seg][policy]["holding"],
                    "stockout": totals[seg][policy]["stockout"],
                    "total": totals[seg][policy]["holding"] + totals[seg][policy]["stockout"],
                }
                for policy in POLICIES
            }
            for seg in totals
        },
        "significance_fused_vs_ml": significance,
        "per_series_segment_costs": per_series_seg_cost,
        "runtime_seconds": elapsed,
    }

    print(f"n_series={n_evaluated} shock_abs_day={shock_abs_day} multiplier={SHOCK_MULTIPLIER} runtime={elapsed:.1f}s")
    for seg in SEGMENTS:
        row = out["segments"][seg]
        sig = significance[seg]
        print(f"  {seg}: " + " ".join(f"{p}={row[p]['total']:.1f}" for p in POLICIES)
              + f"   [wilcoxon p={sig['wilcoxon_signed_rank_p']:.4g} boot95ci={sig['bootstrap_95ci_pct_change']}]")
    fused_vs_ml_pre_refit = 100.0 * (1.0 - out["segments"]["post_pre_refit"]["fused"]["total"] / out["segments"]["post_pre_refit"]["ml"]["total"]) if out["segments"]["post_pre_refit"]["ml"]["total"] else 0.0
    fused_vs_ml_after_refit = 100.0 * (1.0 - out["segments"]["post_after_first_refit"]["fused"]["total"] / out["segments"]["post_after_first_refit"]["ml"]["total"]) if out["segments"]["post_after_first_refit"]["ml"]["total"] else 0.0
    fused_vs_ml_late = 100.0 * (1.0 - out["segments"]["post_late"]["fused"]["total"] / out["segments"]["post_late"]["ml"]["total"]) if out["segments"]["post_late"]["ml"]["total"] else 0.0
    print(f"  days to first refit: {first_refit_day - shock_abs_day}")
    print(f"  fused vs ml: pre-refit={fused_vs_ml_pre_refit:.1f}% after-first-refit={fused_vs_ml_after_refit:.1f}% post-late={fused_vs_ml_late:.1f}%")

    suffix = "" if sample_seed == SAMPLE_SEED else f"_seed{sample_seed}"
    out_path = Path(output_path) if output_path is not None else (
        Path(__file__).resolve().parent.parent / "research" / f"m5_distribution_shift_results{suffix}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Full results written to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the constructed M5 demand-shock evaluation")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--n-per-stratum", type=int, default=N_PER_STRATUM)
    parser.add_argument("--opening-inventory-mode", choices=("gaussian_target", "zero"), default="gaussian_target")
    parser.add_argument("--planner-seed-base", type=int, default=0)
    parser.add_argument("--aci-gamma", type=float, default=m5eval.ACI_GAMMA)
    parser.add_argument("--integer-order-units", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(
        sample_seed=args.sample_seed,
        n_per_stratum=args.n_per_stratum,
        opening_inventory_mode=args.opening_inventory_mode,
        planner_seed_base=args.planner_seed_base,
        aci_gamma=args.aci_gamma,
        integer_order_units=args.integer_order_units,
        output_path=args.output,
    )
    sys.exit(0)
