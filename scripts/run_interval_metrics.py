"""Interval-quality metrics for the demand intervals used by the policy.

The main study reports coverage and mean width. A forecasting reader also
needs a proper scoring rule, so this script scores the same one-step daily
intervals (same sample, same 364-day test window, same fit/calibration
windows and refit cadence) with the interval (Winkler) score and the average
pinball loss at the two interval end points.

Methods, all at a 90% nominal level and all using the study's weekday-seasonal
event-adjusted point forecaster:
  rolling_aci    refit every 28 days + Adaptive Conformal Inference (the study's calibrator)
  rolling        refit every 28 days, ACI off (gamma = 0)
  static         fitted once at the start of the test window, never refit, ACI off
  gaussian       point forecast +/- 1.645 * RMS of the current calibration residuals, refit every 28 days

Writes research/interval_metrics.json. Scores are reported pooled over
series-days (units of demand) and scale-free (score divided by the series'
mean test-window demand, then averaged over series with positive mean demand).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import conformal  # noqa: E402
import m5_loader  # noqa: E402
import run_m5_evaluation as ev  # noqa: E402

ALPHA = 0.10
Z90 = 1.6448536269514722
SEEDS = list(range(8))
METHODS = ("rolling_aci", "rolling", "static", "gaussian")


def interval_score(y, lo, hi, alpha=ALPHA):
    return (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def run_series(series, event_days, method):
    if method == "static":
        cal = conformal.RollingConformalCalibrator(
            series, ev.FIT_WINDOW, ev.CAL_WINDOW, 10**9, 1.0 - ALPHA, 0.0, event_days=event_days
        )
    elif method == "rolling":
        cal = conformal.RollingConformalCalibrator(
            series, ev.FIT_WINDOW, ev.CAL_WINDOW, ev.REFIT_PERIOD_DAYS, 1.0 - ALPHA, 0.0, event_days=event_days
        )
    elif method == "rolling_aci":
        cal = conformal.RollingConformalCalibrator(
            series, ev.FIT_WINDOW, ev.CAL_WINDOW, ev.REFIT_PERIOD_DAYS, 1.0 - ALPHA, ev.ACI_GAMMA,
            event_days=event_days,
        )
    else:  # gaussian: same forecaster and refit cadence, normal interval from calibration residual RMS
        cal = conformal.RollingConformalCalibrator(
            series, ev.FIT_WINDOW, ev.CAL_WINDOW, ev.REFIT_PERIOD_DAYS, 1.0 - ALPHA, 0.0, event_days=event_days
        )
    los, his, ys = [], [], []
    for i in range(ev.TEST_DAYS):
        day = ev.TEST_START + i
        cal.refit_if_due(day)
        pf = cal._forecast_one(day)
        if method == "gaussian":
            q = Z90 * float(np.sqrt(np.mean(cal.scores ** 2)))
        else:
            q = cal.current_q()
        lo, hi = conformal.interval(pf, q)
        y = float(series[day])
        if method in ("rolling_aci", "rolling", "static"):
            cal.observe(day, y)
        los.append(lo)
        his.append(hi)
        ys.append(y)
    return np.array(ys), np.array(los), np.array(his)


def summarize_series(y, lo, hi):
    score = interval_score(y, lo, hi)
    pin = 0.5 * (pinball(y, lo, ALPHA / 2) + pinball(y, hi, 1 - ALPHA / 2))
    return {
        "coverage": float(np.mean((y >= lo - 1e-9) & (y <= hi + 1e-9))),
        "width": float(np.mean(hi - lo)),
        "interval_score": float(np.mean(score)),
        "pinball": float(np.mean(pin)),
        "mean_demand": float(np.mean(y)),
    }


def main():
    meta = m5_loader.load_series_meta()
    zfracs = m5_loader.compute_zero_fracs(end_day_exclusive=ev.TEST_START)
    prices = m5_loader.load_prices_index()
    calendar = m5_loader.load_calendar()
    event_days = calendar["is_event"]
    meta_by_id = {m["id"]: m for m in meta}
    out = {"alpha": ALPHA, "methods": list(METHODS), "per_seed": {}}
    for seed in SEEDS:
        ids, _ = m5_loader.stratified_sample(meta, ev.N_PER_STRATUM, zfracs, seed=seed)
        ids = [i for i in ids if prices.get((meta_by_id[i]["store_id"], meta_by_id[i]["item_id"])) is not None]
        series_map = m5_loader.load_series_by_ids(ids)
        res = {m: [] for m in METHODS}
        for sid in ids:
            for m in METHODS:
                y, lo, hi = run_series(series_map[sid], event_days, m)
                res[m].append(summarize_series(y, lo, hi))
        seed_out = {"n_series": len(ids)}
        for m in METHODS:
            rows = res[m]
            pos = [r for r in rows if r["mean_demand"] > 0]
            seed_out[m] = {
                "coverage": float(np.mean([r["coverage"] for r in rows])),
                "width": float(np.mean([r["width"] for r in rows])),
                "interval_score": float(np.mean([r["interval_score"] for r in rows])),
                "pinball": float(np.mean([r["pinball"] for r in rows])),
                "scaled_interval_score": float(np.mean([r["interval_score"] / r["mean_demand"] for r in pos])),
                "n_positive_demand": len(pos),
            }
        out["per_seed"][str(seed)] = seed_out
        print(seed, {m: round(seed_out[m]["interval_score"], 3) for m in METHODS}, flush=True)
    summ = {}
    for m in METHODS:
        for k in ("coverage", "width", "interval_score", "pinball", "scaled_interval_score"):
            vals = [out["per_seed"][s][m][k] for s in out["per_seed"]]
            summ.setdefault(m, {})[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    out["summary_over_draws"] = summ
    # draws in which each method has the lowest interval score
    wins = {m: 0 for m in METHODS}
    for s in out["per_seed"]:
        best = min(METHODS, key=lambda m: out["per_seed"][s][m]["interval_score"])
        wins[best] += 1
    out["lowest_interval_score_draws"] = wins
    path = ROOT / "research" / "interval_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
