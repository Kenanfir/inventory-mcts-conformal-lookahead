"""
Re-run of the synthetic fail-fast check (workspace/tmp/inventory-mcts-fuzzy-
conformal-checkfast/run_checkfast.py) with the Part 1 fix applied: static
split conformal is replaced by conformal.RollingConformalCalibrator
(periodic forecaster/calibration refit + Adaptive Conformal Inference).

Gate before touching real M5 data (contract 23, Part 1 step 2): confirm
empirical coverage is close to nominal with the fix, on the same synthetic
setup that originally showed the 84.8%-vs-90%-nominal undercoverage. See
research/ADAPTIVE_CONFORMAL.md for the gamma/refit-period selection.

Everything else (fuzzy risk module, MCTS planner, baseline policy, series
generator, simulation parameters) is UNCHANGED from the original fail-fast
check, so any coverage/cost difference is attributable to the conformal fix
being tested, not a confound from also changing something else.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import synthetic_demand
import conformal
import fuzzy_risk
import mcts_policy
import baseline_policy

N_SERIES = 64
N_DAYS = 150
FIT_DAYS = 60
CAL_DAYS = 45
TEST_DAYS = 45
REVIEW_PERIOD_DAYS = 3
MCTS_HORIZON_PERIODS = 3
MCTS_SIMULATIONS = 15
ACTION_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]
NOMINAL_LEVELS = [0.8, 0.9]
OPERATIONAL_LEVEL = 0.9

# Rolling-conformal parameters, selected in research/ADAPTIVE_CONFORMAL.md's
# gamma/refit-period sweep on this same synthetic setup (chosen for coverage
# closeness/stability; the fused-vs-baseline cost result below is not
# sensitive to this choice within the sweep -- see ADAPTIVE_CONFORMAL.md).
REFIT_PERIOD_DAYS = 30
ACI_GAMMA = 0.03


def _daily_coverage_check(series, calib_level):
    """Walk the full post-warmup span of `series` day by day, refit-and-ACI
    as the rolling calibrator prescribes, and report realized coverage at
    `calib_level`. Used only for the coverage diagnostic, separately from
    the fused-policy cost simulation below (which uses the same calibrator
    class but at REVIEW_PERIOD_DAYS granularity, matching how the policy
    actually consumes it).
    """
    calib = conformal.RollingConformalCalibrator(
        series, FIT_DAYS, CAL_DAYS, REFIT_PERIOD_DAYS, calib_level, ACI_GAMMA
    )
    start = FIT_DAYS + CAL_DAYS
    n = len(series) - start
    covered = 0
    widths = []
    for i in range(n):
        day = start + i
        lo, hi, pf, q = calib.step(day, realized_value=series[day])
        widths.append(hi - lo)
        if lo - 1e-9 <= series[day] <= hi + 1e-9:
            covered += 1
    return covered / n, float(np.mean(widths))


def run_one_series(seed: int):
    series = synthetic_demand.generate_series(seed, n_days=N_DAYS)
    fit = series[0:FIT_DAYS]
    test = series[FIT_DAYS + CAL_DAYS:FIT_DAYS + CAL_DAYS + TEST_DAYS]
    assert len(test) == TEST_DAYS

    coverage_by_level = {}
    for level in NOMINAL_LEVELS:
        emp_cov, mean_width = _daily_coverage_check(series, level)
        coverage_by_level[level] = {"empirical_coverage": emp_cov, "mean_interval_width": mean_width}

    s_level = baseline_policy.order_up_to_level(fit, REVIEW_PERIOD_DAYS)
    initial_inventory = s_level

    rng = np.random.default_rng(10_000 + seed)

    # --- baseline policy over the test window (unchanged) ---
    inv_b = initial_inventory
    cost_b = 0.0
    t = 0
    while t < TEST_DAYS:
        period_len = min(REVIEW_PERIOD_DAYS, TEST_DAYS - t)
        order_qty = baseline_policy.order_quantity(inv_b, s_level)
        demand = float(test[t:t + period_len].sum())
        inv_after = inv_b + order_qty
        sold = min(inv_after, demand)
        leftover = inv_after - sold
        unmet = demand - sold
        cost_b += mcts_policy.HOLDING_COST * leftover + mcts_policy.STOCKOUT_COST * unmet
        inv_b = leftover
        t += period_len

    # --- fused MCTS + fuzzy + rolling-conformal(ACI) policy over the test window ---
    calib = conformal.RollingConformalCalibrator(
        series, FIT_DAYS, CAL_DAYS, REFIT_PERIOD_DAYS, OPERATIONAL_LEVEL, ACI_GAMMA
    )
    inv_f = initial_inventory
    cost_f = 0.0
    t = 0
    abs_day = FIT_DAYS + CAL_DAYS
    while t < TEST_DAYS:
        period_len = min(REVIEW_PERIOD_DAYS, TEST_DAYS - t)
        horizon_days = MCTS_HORIZON_PERIODS * REVIEW_PERIOD_DAYS

        # Refit (if due) using only data strictly before abs_day, then take a
        # read-only snapshot (current weekday_avg / q) for the MCTS horizon --
        # same "uniform q across the horizon" simplification the original
        # script used, just from a live-tracked q instead of a static one.
        calib.refit_if_due(abs_day)
        forecasts_per_day = calib.forecast_days(abs_day, horizon_days)
        cal_q_operational = calib.current_q()

        lo0 = max(0.0, forecasts_per_day[:REVIEW_PERIOD_DAYS].sum() - cal_q_operational * REVIEW_PERIOD_DAYS ** 0.5)
        hi0 = forecasts_per_day[:REVIEW_PERIOD_DAYS].sum() + cal_q_operational * REVIEW_PERIOD_DAYS ** 0.5
        period_forecast = float(forecasts_per_day[:REVIEW_PERIOD_DAYS].sum())

        buffer_ratio = (inv_f - lo0) / max(hi0 - lo0, 1e-6)
        rel_width = (hi0 - lo0) / max(period_forecast, 1e-6)
        risk_mult = fuzzy_risk.risk_multiplier(buffer_ratio, rel_width)

        order_qty = mcts_policy.plan_action(
            rng,
            inv_f,
            forecasts_per_day,
            cal_q_operational,
            risk_mult,
            REVIEW_PERIOD_DAYS,
            MCTS_HORIZON_PERIODS,
            ACTION_MULTIPLIERS,
            n_simulations=MCTS_SIMULATIONS,
        )

        demand = float(test[t:t + period_len].sum())
        inv_after = inv_f + order_qty
        sold = min(inv_after, demand)
        leftover = inv_after - sold
        unmet = demand - sold
        cost_f += mcts_policy.HOLDING_COST * leftover + mcts_policy.STOCKOUT_COST * unmet
        inv_f = leftover

        # Feed each realized day in this period back into ACI, in order --
        # this is the only place actual values reach the calibrator, and only
        # for days that have just happened.
        for i in range(period_len):
            day = abs_day + i
            if day < FIT_DAYS + CAL_DAYS + TEST_DAYS:
                calib.observe(day, series[day])

        t += period_len
        abs_day += period_len

    return {
        "coverage_by_level": coverage_by_level,
        "baseline_cost": cost_b,
        "fused_cost": cost_f,
        "mean_series_demand": float(series.mean()),
        "zero_frac": float((series == 0).mean()),
    }


def main():
    t0 = time.time()
    results = [run_one_series(i) for i in range(N_SERIES)]

    agg_coverage = {}
    for level in NOMINAL_LEVELS:
        covs = [r["coverage_by_level"][level]["empirical_coverage"] for r in results]
        widths = [r["coverage_by_level"][level]["mean_interval_width"] for r in results]
        agg_coverage[str(level)] = {
            "nominal": level,
            "mean_empirical_coverage": float(np.mean(covs)),
            "std_empirical_coverage": float(np.std(covs)),
            "mean_interval_width": float(np.mean(widths)),
        }

    baseline_costs = np.array([r["baseline_cost"] for r in results])
    fused_costs = np.array([r["fused_cost"] for r in results])
    improvement_pct = 100.0 * (baseline_costs - fused_costs) / np.where(baseline_costs == 0, 1, baseline_costs)

    elapsed = time.time() - t0

    cov90 = agg_coverage["0.9"]["mean_empirical_coverage"]
    cov_ok = abs(cov90 - 0.9) <= 0.05
    fused_better = float(np.mean(fused_costs)) < float(np.mean(baseline_costs))
    n_series_fused_better = int((fused_costs < baseline_costs).sum())

    verdict_lines = [
        f"[FIXED conformal: rolling refit every {REFIT_PERIOD_DAYS}d + ACI gamma={ACI_GAMMA}]",
        f"Conformal calibration: nominal 90% -> empirical {cov90:.3f} across {N_SERIES} series "
        f"({'CLOSE to nominal' if cov_ok else 'OFF nominal, needs investigation'}); "
        f"static split conformal on this same setup gave 84.8% (research/FAIL_FAST_FINDINGS.md).",
        f"Fused (MCTS+fuzzy+rolling-conformal) mean cost {fused_costs.mean():.2f} vs baseline "
        f"(order-up-to) mean cost {baseline_costs.mean():.2f} "
        f"({'fused beats baseline on average' if fused_better else 'fused does NOT beat baseline on average'}, "
        f"mean cost change {np.mean(improvement_pct):.1f}%).",
        f"Fused beat baseline on {n_series_fused_better}/{N_SERIES} individual series.",
        f"Runtime: {elapsed:.1f}s for {N_SERIES} series.",
    ]
    verdict = " ".join(verdict_lines)

    output = {
        "fix": {
            "method": "rolling refit + Adaptive Conformal Inference (Gibbs & Candes 2021, arXiv:2106.00170)",
            "refit_period_days": REFIT_PERIOD_DAYS,
            "aci_gamma": ACI_GAMMA,
        },
        "n_series": N_SERIES,
        "runtime_seconds": elapsed,
        "conformal_coverage": agg_coverage,
        "baseline_cost": {
            "mean": float(baseline_costs.mean()),
            "std": float(baseline_costs.std()),
            "per_series": baseline_costs.tolist(),
        },
        "fused_cost": {
            "mean": float(fused_costs.mean()),
            "std": float(fused_costs.std()),
            "per_series": fused_costs.tolist(),
        },
        "mean_cost_change_pct": float(np.mean(improvement_pct)),
        "n_series_fused_better": n_series_fused_better,
        "coverage_ok_at_90": cov_ok,
        "fused_beats_baseline_on_average": fused_better,
        "verdict": verdict,
    }

    out_path = Path(__file__).resolve().parent.parent / "research" / "checkfast_fixed_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(verdict)
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
