"""Split conformal prediction for one-step-ahead demand.

Point forecast: weekday-seasonal average fit on a training window (a
deliberately simple baseline — the thing being checked here is the conformal
calibration and downstream policy, not forecast sophistication).
"""

import numpy as np


def fit_seasonal_forecaster(train: np.ndarray):
    weekday_avg = np.zeros(7)
    for w in range(7):
        vals = train[w::7]
        weekday_avg[w] = vals.mean() if len(vals) else train.mean()
    return weekday_avg


def forecast(weekday_avg: np.ndarray, day_index: int) -> float:
    return float(weekday_avg[day_index % 7])


def fit_event_multiplier(train: np.ndarray, train_is_event: np.ndarray, clip=(0.5, 3.0)) -> float:
    """Ratio of mean demand on calendar event days to mean demand on
    non-event days within `train`, clipped to `clip` to avoid an unstable
    estimate from a training window with very few event days. Returns 1.0
    (no adjustment) if there are no event days in `train` or non-event mean
    demand is non-positive. This is the "event flags" half of what
    `research/DATA.md`/the build contract asked the real M5 forecaster to
    actually use, instead of the synthetic check's plain weekday index.
    """
    if train_is_event is None or not np.any(train_is_event):
        return 1.0
    event_vals = train[train_is_event]
    normal_vals = train[~train_is_event]
    if len(event_vals) == 0 or len(normal_vals) == 0 or normal_vals.mean() <= 0:
        return 1.0
    mult = event_vals.mean() / normal_vals.mean()
    return float(np.clip(mult, clip[0], clip[1]))


def forecast_event_adjusted(weekday_avg: np.ndarray, event_mult: float, day_index: int, is_event_day: bool) -> float:
    base = forecast(weekday_avg, day_index)
    return base * event_mult if is_event_day else base


def calibrate(weekday_avg: np.ndarray, cal: np.ndarray, cal_start_day: int):
    """Return nonconformity scores (abs residual) for the calibration window."""
    n = len(cal)
    preds = np.array([forecast(weekday_avg, cal_start_day + i) for i in range(n)])
    scores = np.abs(cal - preds)
    return scores


def calibrate_event_adjusted(
    weekday_avg: np.ndarray, event_mult: float, cal: np.ndarray, cal_start_day: int, cal_is_event: np.ndarray
):
    """Same as `calibrate`, but residuals are computed against the
    event-adjusted forecast, so nonconformity scores reflect what the
    predictor actually outputs (including on event days) -- calibrating
    against the plain weekday forecast while predicting with the
    event-adjusted one would itself violate exchangeability between
    calibration and prediction residuals.
    """
    n = len(cal)
    preds = np.array(
        [forecast_event_adjusted(weekday_avg, event_mult, cal_start_day + i, cal_is_event[i]) for i in range(n)]
    )
    return np.abs(cal - preds)


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile (Lei et al. correction)."""
    n = len(scores)
    idx = int(np.ceil((n + 1) * level)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(np.sort(scores)[idx])


def interval(point_forecast: float, q: float):
    lower = max(0.0, point_forecast - q)
    upper = point_forecast + q
    return lower, upper


def evaluate_coverage(weekday_avg, q, test: np.ndarray, test_start_day: int):
    n = len(test)
    covered = 0
    widths = []
    for i in range(n):
        pf = forecast(weekday_avg, test_start_day + i)
        lo, hi = interval(pf, q)
        widths.append(hi - lo)
        if lo - 1e-9 <= test[i] <= hi + 1e-9:
            covered += 1
    return covered / n, float(np.mean(widths))


def aci_step(weekday_avg, scores: np.ndarray, alpha_t: float, day_index: int):
    """One step of Adaptive Conformal Inference (Gibbs & Candes 2021): build
    the interval for `day_index` using the calibration scores' (1 - alpha_t)
    quantile, where alpha_t is the online-adjusted miscoverage level (not the
    fixed target alpha). alpha_t is clipped to (eps, 1-eps) since the
    finite-sample quantile in `conformal_quantile` is undefined outside it.
    """
    alpha_t = min(max(alpha_t, 1e-6), 1 - 1e-6)
    q = conformal_quantile(scores, 1.0 - alpha_t)
    pf = forecast(weekday_avg, day_index)
    lo, hi = interval(pf, q)
    return lo, hi, pf, q


def evaluate_coverage_aci(
    weekday_avg,
    scores: np.ndarray,
    test: np.ndarray,
    test_start_day: int,
    target_level: float,
    gamma: float,
):
    """Adaptive Conformal Inference (ACI), Gibbs & Candes (2021), "Adaptive
    Conformal Inference Under Distribution Shift" (arXiv:2106.00170). Update
    rule: alpha_{t+1} = alpha_t + gamma * (alpha - err_t), where alpha is the
    fixed target miscoverage (1 - target_level) and err_t is 1 if the
    realized value fell outside the interval built from alpha_t, else 0.

    This directly targets the mechanism diagnosed in FAIL_FAST_FINDINGS.md:
    a static split-conformal quantile undercovers once demand drifts away
    from the calibration-window residual distribution (exchangeability
    violation under trend). ACI recalibrates online using only information
    available at each step (the realized coverage error), so it adapts
    without needing to re-fit the forecaster or peek ahead.

    Found limitation (see ADAPTIVE_CONFORMAL.md): ACI alone is bounded by
    the calibration set's own residual distribution — `conformal_quantile`
    can only return a value already present in `scores`, so however low
    `alpha_t` goes, the interval can never widen past the single widest
    residual seen at calibration time. Under sustained/accelerating trend
    (as in this project's synthetic generator, where the drift term compounds
    with `day_index`), new residuals keep exceeding that historical max, so
    ACI-alone coverage does not converge to nominal and can even *degrade*
    on longer test horizons. `RollingConformalCalibrator` below combines this
    with periodic refitting, which is what actually fixes it.

    Returns (empirical_coverage, mean_interval_width, alpha_trace) — the
    alpha_trace is returned for diagnostics (e.g. confirming it visibly
    tightens/widens in response to drift rather than sitting flat).
    """
    target_alpha = 1.0 - target_level
    alpha_t = target_alpha
    n = len(test)
    covered = 0
    widths = []
    alpha_trace = []
    for i in range(n):
        lo, hi, _, _ = aci_step(weekday_avg, scores, alpha_t, test_start_day + i)
        widths.append(hi - lo)
        err = 0 if (lo - 1e-9 <= test[i] <= hi + 1e-9) else 1
        if err == 0:
            covered += 1
        alpha_trace.append(alpha_t)
        alpha_t = alpha_t + gamma * (target_alpha - err)
    return covered / n, float(np.mean(widths)), alpha_trace


class RollingConformalCalibrator:
    """Online demand-interval calibrator: periodic forecaster/calibration
    refit + Adaptive Conformal Inference (ACI), combined.

    Diagnosed fix for FAIL_FAST_FINDINGS.md's coverage-drift problem, which
    named two candidate fixes -- "(a) a forecaster that gets refit/updated
    ... or (b) an adaptive/online conformal method ... or (c) both" -- and
    predicted (c) might be needed. Empirically it was: on the synthetic
    fail-fast setup at 90% nominal, static split conformal gave 84.8%
    coverage; ACI alone recovered only to ~88-89% and got *worse* on longer
    horizons (see `evaluate_coverage_aci`'s docstring for why); rolling
    refit alone reached ~89-92% with still-elevated variance; refit + ACI
    together converged tightly to 89.8-90.1% coverage across test horizons
    of 45-365 days, with variance shrinking as the horizon grows. See
    ADAPTIVE_CONFORMAL.md for the full comparison and exact numbers.

    Refit uses only data strictly before the current day (`series[:day_index]`
    windows) -- never a lookahead into data the policy hasn't seen yet.

    Refit cadence is keyed to `day_index // refit_period` (a "refit epoch"),
    not to how many times a method has been called -- so `current_q()` /
    `forecast_days()` can be used freely to preview multiple days ahead (for
    a multi-day planning horizon) without disturbing state or requiring one
    call per calendar day, while `refit_if_due` + `observe` (the only two
    state-mutating methods) are meant to be called exactly once per calendar
    day, in day order, as the simulation actually advances through time --
    using only data strictly before that day, never a peek ahead.
    """

    def __init__(
        self,
        series: np.ndarray,
        fit_window: int,
        cal_window: int,
        refit_period: int,
        target_level: float,
        gamma: float,
        event_days: np.ndarray = None,
    ):
        self.series = series
        self.fit_window = fit_window
        self.cal_window = cal_window
        self.refit_period = refit_period
        self.target_level = target_level
        self.target_alpha = 1.0 - target_level
        self.gamma = gamma
        self.alpha_t = self.target_alpha
        self.event_days = event_days  # bool array aligned with `series`, or None
        self.weekday_avg = None
        self.event_mult = 1.0
        self.scores = None
        self._last_refit_epoch = None

    def refit_if_due(self, day_index: int):
        """Refit the forecaster + calibration scores using only
        `series[:day_index]`, if this day starts a new refit epoch. Call
        once per calendar day as the simulation advances; a no-op most days.
        """
        epoch = day_index // self.refit_period
        if self._last_refit_epoch == epoch:
            return
        fit_start = day_index - self.fit_window - self.cal_window
        cal_start = day_index - self.cal_window
        if fit_start < 0:
            raise ValueError(
                f"day_index={day_index} has fewer than fit_window+cal_window="
                f"{self.fit_window + self.cal_window} days of history before it"
            )
        fit = self.series[fit_start:cal_start]
        cal = self.series[cal_start:day_index]
        self.weekday_avg = fit_seasonal_forecaster(fit)
        if self.event_days is None:
            self.scores = calibrate(self.weekday_avg, cal, cal_start_day=cal_start)
        else:
            fit_is_event = self.event_days[fit_start:cal_start]
            cal_is_event = self.event_days[cal_start:day_index]
            self.event_mult = fit_event_multiplier(fit, fit_is_event)
            self.scores = calibrate_event_adjusted(
                self.weekday_avg, self.event_mult, cal, cal_start_day=cal_start, cal_is_event=cal_is_event
            )
        self._last_refit_epoch = epoch

    def current_q(self) -> float:
        """Current conformal quantile magnitude from the live calibration
        scores and the current (ACI-adjusted) alpha_t. Pure/read-only -- safe
        to call for any day using the current state, including days in a
        planning horizon that haven't happened yet.
        """
        alpha_t = min(max(self.alpha_t, 1e-6), 1 - 1e-6)
        return conformal_quantile(self.scores, 1.0 - alpha_t)

    def _forecast_one(self, day_index: int) -> float:
        if self.event_days is None:
            return forecast(self.weekday_avg, day_index)
        return forecast_event_adjusted(self.weekday_avg, self.event_mult, day_index, bool(self.event_days[day_index]))

    def forecast_days(self, start_day: int, n_days: int) -> np.ndarray:
        """Point forecast for `n_days` starting at `start_day`, from the
        current weekday_avg (and event multiplier, if event_days was given).
        Pure/read-only."""
        return np.array([self._forecast_one(start_day + i) for i in range(n_days)])

    def observe(self, day_index: int, realized_value: float):
        """Update alpha_t via ACI: build the interval for `day_index` with
        the CURRENT state (before this observation), check whether
        `realized_value` fell inside it, and adjust alpha_t accordingly.
        Call once per calendar day, in day order, only after that day's
        value is actually known -- this is the only place a future value
        could leak in if misused, so never call it with a day that hasn't
        happened yet in the simulation.
        """
        q = self.current_q()
        pf = self._forecast_one(day_index)
        lo, hi = interval(pf, q)
        err = 0 if (lo - 1e-9 <= realized_value <= hi + 1e-9) else 1
        self.alpha_t = self.alpha_t + self.gamma * (self.target_alpha - err)
        return lo, hi, pf, q, err

    def step(self, day_index: int, realized_value: float = None):
        """Convenience wrapper for daily use: refit-if-due, build today's
        interval, and (if `realized_value` given) observe it. Returns
        (lo, hi, point_forecast, q). Do not use this alongside
        `forecast_days`/`current_q` previews inside the same day -- it
        mutates state via `refit_if_due` and (if given a value) `observe`.
        """
        self.refit_if_due(day_index)
        if realized_value is None:
            q = self.current_q()
            pf = self._forecast_one(day_index)
            lo, hi = interval(pf, q)
            return lo, hi, pf, q
        lo, hi, pf, q, _ = self.observe(day_index, realized_value)
        return lo, hi, pf, q
