"""
Regenerates, as a saved artifact, the static-vs-(refit+ACI) conformal
coverage comparison across test horizons that research/ADAPTIVE_CONFORMAL.md
reports in prose. That comparison was originally computed interactively in a
scratch session and never saved to a file -- this script reproduces it
properly, using the same synthetic generator, parameters (refit_period=30,
gamma=0.03, 64 series, FIT_DAYS=60, CAL_DAYS=45), and src/conformal.py
functions, and writes the result to research/coverage_horizon_sweep.json so a
figure can be generated from a real, checked-in artifact rather than from
memory of a printed number.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import synthetic_demand
import conformal

N_SERIES = 64
FIT_DAYS = 60
CAL_DAYS = 45
LEVEL = 0.9
REFIT_PERIOD = 30
GAMMA = 0.03
HORIZONS = [45, 90, 180, 365]


def static_coverage(seed, test_days, n_days):
    series = synthetic_demand.generate_series(seed, n_days=n_days)
    fit = series[0:FIT_DAYS]
    cal = series[FIT_DAYS:FIT_DAYS + CAL_DAYS]
    test = series[FIT_DAYS + CAL_DAYS:FIT_DAYS + CAL_DAYS + test_days]
    weekday_avg = conformal.fit_seasonal_forecaster(fit)
    scores = conformal.calibrate(weekday_avg, cal, cal_start_day=FIT_DAYS)
    q = conformal.conformal_quantile(scores, LEVEL)
    cov, _ = conformal.evaluate_coverage(weekday_avg, q, test, test_start_day=FIT_DAYS + CAL_DAYS)
    return cov


def rolling_aci_coverage(seed, test_days, n_days):
    series = synthetic_demand.generate_series(seed, n_days=n_days)
    start = n_days - test_days
    calib = conformal.RollingConformalCalibrator(series, FIT_DAYS, CAL_DAYS, REFIT_PERIOD, LEVEL, GAMMA)
    covered = 0
    for i in range(test_days):
        day = start + i
        lo, hi, pf, q = calib.step(day, realized_value=series[day])
        if lo - 1e-9 <= series[day] <= hi + 1e-9:
            covered += 1
    return covered / test_days


def main():
    results = {"n_series": N_SERIES, "level": LEVEL, "refit_period": REFIT_PERIOD, "gamma": GAMMA, "horizons": {}}
    for horizon in HORIZONS:
        n_days = FIT_DAYS + CAL_DAYS + horizon
        static_covs = [static_coverage(i, horizon, n_days) for i in range(N_SERIES)]
        aci_covs = [rolling_aci_coverage(i, horizon, n_days) for i in range(N_SERIES)]
        results["horizons"][str(horizon)] = {
            "static_mean": float(np.mean(static_covs)),
            "static_std": float(np.std(static_covs)),
            "refit_aci_mean": float(np.mean(aci_covs)),
            "refit_aci_std": float(np.std(aci_covs)),
        }
        print(f"horizon={horizon}: static={np.mean(static_covs):.4f} "
              f"refit+aci={np.mean(aci_covs):.4f} (std {np.std(aci_covs):.4f})")

    out_path = Path(__file__).resolve().parent.parent / "research" / "coverage_horizon_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
