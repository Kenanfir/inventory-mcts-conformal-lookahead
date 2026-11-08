"""
Part 2 of docs/handoff/contracts/23-inventory-mcts-fuzzy-conformal-build.md:
real-M5 evaluation of the root-action Monte Carlo lookahead policy with
fuzzy risk adjustment and rolling conformal calibration (ACI) vs
a naive order-up-to baseline, on a stratified sample of real M5 series, with
the Part 1 conformal fix already applied (see research/ADAPTIVE_CONFORMAL.md
-- do not compare these numbers to the pre-fix 28.3% synthetic figure, which
was a since-corrected artifact).

Sample: 324 series (12 per cell x 27 cells: cat_id{FOODS,HOBBIES,HOUSEHOLD}
x state_id{CA,TX,WI} x intermittency-tertile{low,medium,high}, tertiles cut
within each (cat_id, state_id) group on each series' zero-demand fraction
from pre-test days d_1..d_1577 -- see src/m5_loader.py:stratified_sample).
Fixed seed (0) for reproducibility.
Not all ~30,490 series in sales_train_evaluation.csv: a stratified sample
sized for a meaningful, explicitly-reasoned real-data check within a
reasonable runtime, per contract Part 2 step 3.

Test window: last 364 days (52 weeks) of the 1941-day M5 history
(2011-01-29 to 2016-06-19), weekly review (REVIEW_PERIOD_DAYS=7, more
realistic for retail replenishment than the synthetic check's 3-day period).
Fit/calibration windows: 180/90 days immediately preceding the test window,
refit every 28 days (4 weeks) -- the ~30-day cadence validated in
ADAPTIVE_CONFORMAL.md's sweep, adjusted to a whole number of review periods.

Cost model: holding cost = sell_price * ANNUAL_HOLDING_RATE / 365 *
REVIEW_PERIOD_DAYS per unit left over each period; stockout cost =
sell_price per unit of unmet demand (lost-sale value; M5 has no COGS/margin
data, so full retail price is used as the standard newsvendor-style
approximation when margin is unknown -- see the cost-model note in
research/DATA.md this script's docstring points to). ANNUAL_HOLDING_RATE =
0.20 (20%/year), a commonly-cited midpoint in inventory holding-cost
literature (typically quoted 15-35% of item value/year).

Opening state: all five policies start from the Gaussian order-up-to target
estimated on the pre-test fit window; a zero-start sensitivity is reported
separately by scripts/run_m5_robustness.py.
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import m5_loader
import conformal
import fuzzy_risk
import mcts_policy
import baseline_policy
import sba_baseline
import ss_baseline
import ml_baseline

N_PER_STRATUM = 12
SAMPLE_SEED = 0

TOTAL_DAYS = 1941  # sales_train_evaluation.csv day columns, d_1..d_1941
TEST_DAYS = 364
REVIEW_PERIOD_DAYS = 7
FIT_WINDOW = 180
CAL_WINDOW = 90
REFIT_PERIOD_DAYS = 28
MCTS_HORIZON_PERIODS = 3
# MCTS_SIMULATIONS=15 and a 0-2.0x action grid were the synthetic fail-fast
# check's cheap defaults, tuned for that check's fixed 1:5 holding:stockout
# ratio (critical ratio 0.833, ~2x mean forecast is generous headroom).
# Adding the (s,S)/ML baselines (2026-09-23, after the impact-benchmark
# found this project's baseline set too shallow -- research/
# STRENGTHENING_OPTIONS.md) exposed that these defaults were never re-tuned
# for M5's real ~260:1 ratio (critical ratio 0.996): with only a 3.4x
# ceiling (2.0 action multiplier x 1.7 max fuzzy risk multiplier) and 15
# noisy Monte Carlo rollouts to estimate expected cost under a heavily
# right-skewed loss, MCTS could not reach, and reliably could not tell it
# should reach, the near-max-demand order levels this cost ratio calls for
# -- verified empirically: realized order_qty_median/mean_weekly_demand
# never exceeded ~1.5x under the old defaults (research/m5_evaluation_
# results.json, pre-fix), while both new baselines search/predict well
# beyond that. Widening the grid and simulation budget (this is the real
# evaluation, not the fail-fast check the old defaults were sized for, so
# the extra compute is legitimate) -- a first pass (sims=40, grid to 6.0x)
# beat naive/SBA/(s,S) at full 324-series scale but still lost to the
# pooled ML baseline (fused_total=12857.8 vs ml_total=6755.4); a second
# pass (sims=60, grid to 10.0x) closed that gap too at n_per_stratum=3
# (fused 2170.6 vs ml 2913.8, ss 3180.4) -- see research/M5_EVALUATION.md
# for the full before/after progression.
MCTS_SIMULATIONS = 60
ACTION_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
NOMINAL_LEVELS = [0.8, 0.9]
OPERATIONAL_LEVEL = 0.9
ACI_GAMMA = 0.03

ANNUAL_HOLDING_RATE = 0.20
DEFAULT_PRICE = None  # set per-series below; series with no price are skipped and reported

TEST_START = TOTAL_DAYS - TEST_DAYS


def planner_seed_for_series(series_id, planner_seed_base: int = 0) -> int:
    """Stable MCTS seed for one series, independent of sample ordering."""
    return int(zlib.crc32(str(series_id).encode("utf-8"), planner_seed_base & 0xFFFFFFFF))


def _validate_opening_inventory_mode(mode: str) -> None:
    if mode not in {"gaussian_target", "zero"}:
        raise ValueError("opening_inventory_mode must be 'gaussian_target' or 'zero'")


def _daily_coverage_check(series, event_days, calib_level, aci_gamma: float = ACI_GAMMA):
    calib = conformal.RollingConformalCalibrator(
        series, FIT_WINDOW, CAL_WINDOW, REFIT_PERIOD_DAYS, calib_level, aci_gamma, event_days=event_days
    )
    covered = 0
    widths = []
    for i in range(TEST_DAYS):
        day = TEST_START + i
        lo, hi, pf, q = calib.step(day, realized_value=series[day])
        widths.append(hi - lo)
        if lo - 1e-9 <= series[day] <= hi + 1e-9:
            covered += 1
    return covered / TEST_DAYS, float(np.mean(widths))


def _quantize_order_quantity(order_qty: float, integer_order_units: bool) -> float:
    """Round positive orders up to whole retail units when requested."""
    if not integer_order_units:
        return float(order_qty)
    return float(np.ceil(max(0.0, float(order_qty)) - 1e-12))


def _ml_epoch_anchor_days(test_start, test_days, review_period_days, refit_period_days):
    """The set of {epoch: first abs_day the test loop enters it} pairs the
    pooled ML baseline needs a model for, mirroring the SBA/conformal
    refit-epoch cadence exactly (`abs_day // refit_period_days`) so it
    refits on the identical schedule as everything else it's compared
    against."""
    anchors = {}
    abs_day = test_start
    t = 0
    while t < test_days:
        epoch = abs_day // refit_period_days
        if epoch not in anchors:
            anchors[epoch] = abs_day
        period_len = min(review_period_days, test_days - t)
        t += period_len
        abs_day += period_len
    return anchors


def train_ml_models_by_epoch(series_map, meta_by_id, prices_series, wm_yr_wk, event_days,
                              holding_rate: float = ANNUAL_HOLDING_RATE,
                              stockout_multiplier: float = 1.0,
                              ml_training_window_days: int = FIT_WINDOW + CAL_WINDOW):
    """One pooled LightGBM quantile model per refit epoch encountered during
    the test window, trained on data pooled across every series in
    `series_map` (src/ml_baseline.py's design: a single global model, not
    one per series -- the only practical choice at typical per-series M5
    history lengths, and what the field's dataset-scale ceiling paper does
    too). By default, the fit window is FIT_WINDOW+CAL_WINDOW days, matching
    the recency horizon used by the other policies. The optional shorter
    window supports a conservative sensitivity check against the conformal
    model's 180-day fit window. Returns (models: dict[epoch] -> model or None,
    critical_ratio: float).
    """
    if ml_training_window_days < 1:
        raise ValueError("ml_training_window_days must be a positive integer")
    critical_ratio = stockout_multiplier / (stockout_multiplier + holding_rate / 365.0 * REVIEW_PERIOD_DAYS)
    anchors = _ml_epoch_anchor_days(TEST_START, TEST_DAYS, REVIEW_PERIOD_DAYS, REFIT_PERIOD_DAYS)
    models = {}
    for epoch, anchor_day in anchors.items():
        start_day = max(0, anchor_day - ml_training_window_days)
        X, y = ml_baseline.build_training_table(
            series_map, meta_by_id, prices_series, wm_yr_wk, event_days, REVIEW_PERIOD_DAYS,
            train_end_day=anchor_day, start_day=start_day,
        )
        models[epoch] = ml_baseline.train_quantile_model(X, y, critical_ratio) if X is not None and len(X) >= 50 else None
    return models, critical_ratio


def run_one_series(
    series_id,
    series,
    event_days,
    price,
    seed,
    cat_id,
    state_id,
    ml_models: dict,
    prices_series: dict,
    wm_yr_wk,
    series_key,
    holding_rate: float = ANNUAL_HOLDING_RATE,
    stockout_multiplier: float = 1.0,
    check_coverage: bool = True,
    record_trajectory: bool = False,
    opening_inventory_mode: str = "gaussian_target",
    planner_seed_base: int | None = None,
    aci_gamma: float = ACI_GAMMA,
    fuzzy_enabled: bool = True,
    mcts_horizon_periods: int = MCTS_HORIZON_PERIODS,
    integer_order_units: bool = False,
):
    """`holding_rate`/`stockout_multiplier` let the cost model be varied
    (research/M5_EVALUATION.md's sensitivity sweep) without touching the
    default single-run path. `check_coverage=False` skips the (cost-model-
    independent) daily coverage re-check, since it doesn't change across
    cost-model scenarios and is the majority of this function's runtime --
    used by the sensitivity sweep, which only cares about cost, not coverage.
    `cat_id`/`state_id` and the pre-trained `ml_models` (one pooled model
    per refit epoch, see `train_ml_models_by_epoch`) drive the ML baseline
    below; it is the only baseline that does not fit its own model here,
    since fitting per-series would contradict the pooled design.
    `prices_series`/`wm_yr_wk`/`series_key` give the ML baseline's
    predictions a walk-forward price-as-of-day feature instead of the
    whole-history mean `price` (which remains the shared cost-model
    coefficient for every policy, unaffected by this) -- see
    m5_loader.load_prices_series's docstring.
    `record_trajectory` (default False, no behavior/output change to any
    existing key) additionally records each policy's per-period state
    (day, holding cost, stockout cost, inventory_before, order, demand;
    fused also logs its fuzzy risk_multiplier) as a list of dicts under a
    `trajectory` key. Added for two consumers: the distribution-shift
    stress test (scripts/run_m5_distribution_shift.py), which needs cost
    isolated to the periods immediately following an injected demand
    shock, and scripts/run_trajectory_example.py (Figure 4), which needs
    the full inventory trajectory -- both now call this same function
    instead of each hand-duplicating the simulation loop, so a change to
    the loop here can't silently drift out of sync with what a figure or
    stress test claims to reproduce.
    """
    _validate_opening_inventory_mode(opening_inventory_mode)
    if mcts_horizon_periods < 1:
        raise ValueError("mcts_horizon_periods must be at least 1")
    holding_cost_per_unit = price * holding_rate / 365.0 * REVIEW_PERIOD_DAYS
    stockout_cost_per_unit = price * stockout_multiplier
    trajectory = {"baseline": [], "sba": [], "ss": [], "ml": [], "fused": []} if record_trajectory else None

    fit = series[TEST_START - FIT_WINDOW - CAL_WINDOW: TEST_START - CAL_WINDOW]
    test = series[TEST_START:TEST_START + TEST_DAYS]
    assert len(test) == TEST_DAYS

    coverage_by_level = {}
    if check_coverage:
        for level in NOMINAL_LEVELS:
            emp_cov, mean_width = _daily_coverage_check(series, event_days, level, aci_gamma)
            coverage_by_level[level] = {"empirical_coverage": emp_cov, "mean_interval_width": mean_width}

    s_level = baseline_policy.order_up_to_level(fit, REVIEW_PERIOD_DAYS)
    initial_inventory = s_level if opening_inventory_mode == "gaussian_target" else 0.0
    if integer_order_units:
        initial_inventory = float(np.ceil(max(0.0, initial_inventory) - 1e-12))
    rng_seed = 10_000 + seed if planner_seed_base is None else planner_seed_for_series(series_id, planner_seed_base)
    rng = np.random.default_rng(rng_seed)

    # --- naive baseline policy (Gaussian order-up-to) ---
    inv_b = initial_inventory
    hold_b = stock_b = 0.0
    t = 0
    while t < TEST_DAYS:
        period_len = min(REVIEW_PERIOD_DAYS, TEST_DAYS - t)
        order_qty = baseline_policy.order_quantity(inv_b, s_level)
        order_qty = _quantize_order_quantity(order_qty, integer_order_units)
        demand = float(test[t:t + period_len].sum())
        inv_after = inv_b + order_qty
        sold = min(inv_after, demand)
        leftover = inv_after - sold
        unmet = demand - sold
        hold_b += holding_cost_per_unit * leftover
        stock_b += stockout_cost_per_unit * unmet
        if record_trajectory:
            trajectory["baseline"].append({"day": TEST_START + t, "holding": holding_cost_per_unit * leftover,
                                            "stockout": stockout_cost_per_unit * unmet, "inventory_before": inv_b,
                                            "order": order_qty, "demand": demand})
        inv_b = leftover
        t += period_len
    cost_b = hold_b + stock_b

    # --- SBA (intermittent-demand-aware) baseline, refit on the same
    # refit-epoch cadence as the fused policy's conformal calibrator
    # (`day_index // REFIT_PERIOD_DAYS`, matching
    # `RollingConformalCalibrator.refit_if_due` exactly, not a
    # TEST_START-relative modulo that drifts out of phase with it -- both
    # bugs found in independent review, 2026-09-22, see
    # research/M5_EVALUATION.md). Fit window uses the freshest
    # FIT_WINDOW+CAL_WINDOW days of history ending at abs_day: SBA has no
    # calibration set (unlike the conformal calibrator, which needs a
    # held-out CAL_WINDOW to compute nonconformity residuals), so excluding
    # the most recent CAL_WINDOW days -- which the first version of this
    # code did, by copying the conformal calibrator's window convention
    # without adapting it -- served no purpose and only made SBA stale by
    # up to 90 days relative to what a real deployment would use. ---
    sba_s_level = sba_baseline.sba_order_up_to_level(fit, REVIEW_PERIOD_DAYS)
    inv_s = initial_inventory
    hold_s = stock_s = 0.0
    t = 0
    abs_day = TEST_START
    sba_last_refit_epoch = None
    while t < TEST_DAYS:
        period_len = min(REVIEW_PERIOD_DAYS, TEST_DAYS - t)
        sba_epoch = abs_day // REFIT_PERIOD_DAYS
        if sba_epoch != sba_last_refit_epoch:
            recent_fit = series[abs_day - FIT_WINDOW - CAL_WINDOW: abs_day]
            sba_s_level = sba_baseline.sba_order_up_to_level(recent_fit, REVIEW_PERIOD_DAYS)
            sba_last_refit_epoch = sba_epoch
        order_qty = sba_baseline.order_quantity(inv_s, sba_s_level)
        order_qty = _quantize_order_quantity(order_qty, integer_order_units)
        demand = float(test[t:t + period_len].sum())
        inv_after = inv_s + order_qty
        sold = min(inv_after, demand)
        leftover = inv_after - sold
        unmet = demand - sold
        hold_s += holding_cost_per_unit * leftover
        stock_s += stockout_cost_per_unit * unmet
        if record_trajectory:
            trajectory["sba"].append({"day": abs_day, "holding": holding_cost_per_unit * leftover,
                                       "stockout": stockout_cost_per_unit * unmet, "inventory_before": inv_s,
                                       "order": order_qty, "demand": demand})
        inv_s = leftover
        t += period_len
        abs_day += period_len
    cost_s = hold_s + stock_s

    # --- tuned (s, S) baseline: simulation-optimized on the lost-sales cost
    # function itself (src/ss_baseline.py), same refit-epoch cadence and
    # fit-window convention as SBA above. ---
    ss_s, ss_S, _ = ss_baseline.fit_ss(fit, REVIEW_PERIOD_DAYS, holding_cost_per_unit, stockout_cost_per_unit)
    inv_ss = initial_inventory
    hold_ss = stock_ss = 0.0
    t = 0
    abs_day = TEST_START
    ss_last_refit_epoch = None
    while t < TEST_DAYS:
        period_len = min(REVIEW_PERIOD_DAYS, TEST_DAYS - t)
        ss_epoch = abs_day // REFIT_PERIOD_DAYS
        if ss_epoch != ss_last_refit_epoch:
            recent_fit = series[abs_day - FIT_WINDOW - CAL_WINDOW: abs_day]
            ss_s, ss_S, _ = ss_baseline.fit_ss(recent_fit, REVIEW_PERIOD_DAYS, holding_cost_per_unit, stockout_cost_per_unit)
            ss_last_refit_epoch = ss_epoch
        order_qty = ss_baseline.order_quantity(inv_ss, ss_s, ss_S)
        order_qty = _quantize_order_quantity(order_qty, integer_order_units)
        demand = float(test[t:t + period_len].sum())
        inv_after = inv_ss + order_qty
        sold = min(inv_after, demand)
        leftover = inv_after - sold
        unmet = demand - sold
        hold_ss += holding_cost_per_unit * leftover
        stock_ss += stockout_cost_per_unit * unmet
        if record_trajectory:
            trajectory["ss"].append({"day": abs_day, "holding": holding_cost_per_unit * leftover,
                                      "stockout": stockout_cost_per_unit * unmet, "inventory_before": inv_ss,
                                      "order": order_qty, "demand": demand})
        inv_ss = leftover
        t += period_len
        abs_day += period_len
    cost_ss = hold_ss + stock_ss

    # --- pooled ML (LightGBM quantile) baseline: uses the pre-trained
    # per-epoch pooled models (src/ml_baseline.py, train_ml_models_by_epoch
    # above) -- the only baseline here that does not fit itself, since the
    # model is shared, global, and trained once per evaluate_sample() call
    # across every series in the sample. ---
    ml_epoch0 = TEST_START // REFIT_PERIOD_DAYS
    model0 = ml_models.get(ml_epoch0)
    price_as_of_start = m5_loader.mean_price_as_of(prices_series, series_key, int(wm_yr_wk[TEST_START]))
    ml_level = ml_baseline.predict_order_up_to_level(
        model0, series, event_days, TEST_START, REVIEW_PERIOD_DAYS, price_as_of_start, cat_id, state_id
    ) if model0 is not None else 0.0
    inv_m = initial_inventory
    hold_m = stock_m = 0.0
    t = 0
    abs_day = TEST_START
    ml_last_refit_epoch = None
    while t < TEST_DAYS:
        period_len = min(REVIEW_PERIOD_DAYS, TEST_DAYS - t)
        ml_epoch = abs_day // REFIT_PERIOD_DAYS
        if ml_epoch != ml_last_refit_epoch:
            model = ml_models.get(ml_epoch)
            if model is not None:
                price_as_of = m5_loader.mean_price_as_of(prices_series, series_key, int(wm_yr_wk[abs_day]))
                ml_level = ml_baseline.predict_order_up_to_level(
                    model, series, event_days, abs_day, REVIEW_PERIOD_DAYS, price_as_of, cat_id, state_id
                )
            ml_last_refit_epoch = ml_epoch
        order_qty = ml_baseline.order_quantity(inv_m, ml_level)
        order_qty = _quantize_order_quantity(order_qty, integer_order_units)
        demand = float(test[t:t + period_len].sum())
        inv_after = inv_m + order_qty
        sold = min(inv_after, demand)
        leftover = inv_after - sold
        unmet = demand - sold
        hold_m += holding_cost_per_unit * leftover
        stock_m += stockout_cost_per_unit * unmet
        if record_trajectory:
            trajectory["ml"].append({"day": abs_day, "holding": holding_cost_per_unit * leftover,
                                      "stockout": stockout_cost_per_unit * unmet, "inventory_before": inv_m,
                                      "order": order_qty, "demand": demand})
        inv_m = leftover
        t += period_len
        abs_day += period_len
    cost_m = hold_m + stock_m

    # --- fused MCTS + fuzzy + rolling-conformal(ACI) policy ---
    calib = conformal.RollingConformalCalibrator(
        series, FIT_WINDOW, CAL_WINDOW, REFIT_PERIOD_DAYS, OPERATIONAL_LEVEL, aci_gamma, event_days=event_days
    )
    inv_f = initial_inventory
    hold_f = stock_f = 0.0
    t = 0
    abs_day = TEST_START
    order_qtys = []
    while t < TEST_DAYS:
        period_len = min(REVIEW_PERIOD_DAYS, TEST_DAYS - t)
        horizon_days = mcts_horizon_periods * REVIEW_PERIOD_DAYS

        calib.refit_if_due(abs_day)
        forecasts_per_day = calib.forecast_days(abs_day, horizon_days)
        cal_q_operational = calib.current_q()

        lo0 = max(0.0, forecasts_per_day[:REVIEW_PERIOD_DAYS].sum() - cal_q_operational * REVIEW_PERIOD_DAYS ** 0.5)
        hi0 = forecasts_per_day[:REVIEW_PERIOD_DAYS].sum() + cal_q_operational * REVIEW_PERIOD_DAYS ** 0.5
        period_forecast = float(forecasts_per_day[:REVIEW_PERIOD_DAYS].sum())

        buffer_ratio = (inv_f - lo0) / max(hi0 - lo0, 1e-6)
        rel_width = (hi0 - lo0) / max(period_forecast, 1e-6)
        risk_mult = fuzzy_risk.risk_multiplier(buffer_ratio, rel_width) if fuzzy_enabled else 1.0

        # Distribution-aware candidates anchored on the conformal interval's
        # own margin (hi0 - lo0, already computed above for the fuzzy
        # inputs) rather than only multiples of the point forecast: the
        # margin approximates the calibrated OPERATIONAL_LEVEL quantile's
        # spread, so extrapolating it by a few multiples reaches further
        # into the tail the way a higher-quantile bound would, without a
        # new distributional assumption beyond what the calibrator already
        # encodes. Needed because M5's real cost ratio (critical ratio
        # ~0.996) puts the true cost-minimizing order level far past what
        # ACTION_MULTIPLIERS x point-forecast alone reaches for
        # high-variance series -- see mcts_policy.plan_action's docstring
        # and research/M5_EVALUATION.md.
        margin = hi0 - lo0
        extra_candidates = [period_forecast + margin * k for k in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)]

        order_qty = mcts_policy.plan_action(
            rng,
            inv_f,
            forecasts_per_day,
            cal_q_operational,
            risk_mult,
            REVIEW_PERIOD_DAYS,
            mcts_horizon_periods,
            ACTION_MULTIPLIERS,
            n_simulations=MCTS_SIMULATIONS,
            holding_cost=holding_cost_per_unit,
            stockout_cost=stockout_cost_per_unit,
            extra_candidate_levels=extra_candidates,
            integer_order_units=integer_order_units,
        )
        order_qtys.append(order_qty)

        demand = float(test[t:t + period_len].sum())
        inv_after = inv_f + order_qty
        sold = min(inv_after, demand)
        leftover = inv_after - sold
        unmet = demand - sold
        hold_f += holding_cost_per_unit * leftover
        stock_f += stockout_cost_per_unit * unmet
        if record_trajectory:
            trajectory["fused"].append({"day": abs_day, "holding": holding_cost_per_unit * leftover,
                                         "stockout": stockout_cost_per_unit * unmet, "inventory_before": inv_f,
                                         "order": order_qty, "demand": demand,
                                         "risk_multiplier": risk_mult})
        inv_f = leftover

        for i in range(period_len):
            day = abs_day + i
            if day < TOTAL_DAYS:
                calib.observe(day, series[day])

        t += period_len
        abs_day += period_len
    cost_f = hold_f + stock_f

    return {
        "series_id": series_id,
        "price": price,
        "opening_inventory_mode": opening_inventory_mode,
        "opening_inventory": float(initial_inventory),
        "coverage_by_level": coverage_by_level,
        "baseline_cost": cost_b,
        "baseline_holding": hold_b,
        "baseline_stockout": stock_b,
        "sba_cost": cost_s,
        "sba_holding": hold_s,
        "sba_stockout": stock_s,
        "ss_cost": cost_ss,
        "ss_holding": hold_ss,
        "ss_stockout": stock_ss,
        "ml_cost": cost_m,
        "ml_holding": hold_m,
        "ml_stockout": stock_m,
        "fused_cost": cost_f,
        "fused_holding": hold_f,
        "fused_stockout": stock_f,
        "mean_test_demand": float(test.mean()),
        "zero_frac_test": float((test == 0).mean()),
        "order_qty_median": float(np.median(order_qtys)),
        "trajectory": trajectory,
    }


def load_shared_data():
    """Everything that doesn't depend on the sample seed or cost-model
    scenario -- loaded once and reused across repeated-sampling /
    sensitivity-sweep runs, not just the single default run."""
    meta = m5_loader.load_series_meta()
    zfracs = m5_loader.compute_zero_fracs(end_day_exclusive=TEST_START)
    series_map_all = None  # loaded per-sample below (only for the ids drawn)
    prices = m5_loader.load_prices_index()
    # Walk-forward price-as-of-day lookup, used only for the ML baseline's
    # predictive feature (`load_prices_index`'s whole-history mean is still
    # used for the shared cost-model coefficient above, which is fine since
    # every policy sees the identical value -- see m5_loader.load_prices_series
    # docstring for why the feature use needed a fix and the coefficient use
    # didn't).
    prices_series = m5_loader.load_prices_series()
    calendar = m5_loader.load_calendar()
    event_days = calendar["is_event"]
    wm_yr_wk = calendar["wm_yr_wk"]
    assert len(event_days) >= TOTAL_DAYS + MCTS_HORIZON_PERIODS * REVIEW_PERIOD_DAYS
    meta_by_id = {m["id"]: m for m in meta}
    return meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id


def evaluate_sample(
    sample_seed,
    meta,
    zfracs,
    prices,
    prices_series,
    wm_yr_wk,
    event_days,
    meta_by_id,
    n_per_stratum: int = N_PER_STRATUM,
    holding_rate: float = ANNUAL_HOLDING_RATE,
    stockout_multiplier: float = 1.0,
    check_coverage: bool = True,
    opening_inventory_mode: str = "gaussian_target",
    planner_seed_base: int = 0,
    aci_gamma: float = ACI_GAMMA,
    fuzzy_enabled: bool = True,
    mcts_horizon_periods: int = MCTS_HORIZON_PERIODS,
    integer_order_units: bool = False,
    ml_training_window_days: int = FIT_WINDOW + CAL_WINDOW,
):
    """Draw a stratified sample with `sample_seed` and run the full
    naive/SBA/fused evaluation + significance testing on it. Used both by
    `main()` (the single reported run, seed=SAMPLE_SEED, defaults) and by
    the repeated-sampling stability check and cost-model sensitivity sweep
    in `run_m5_robustness.py`, which vary `sample_seed` or
    `holding_rate`/`stockout_multiplier` respectively.
    """
    _validate_opening_inventory_mode(opening_inventory_mode)
    if mcts_horizon_periods < 1:
        raise ValueError("mcts_horizon_periods must be at least 1")
    ids, strata_report = m5_loader.stratified_sample(meta, n_per_stratum, zfracs, seed=sample_seed)
    series_map = m5_loader.load_series_by_ids(ids)

    # One pooled ML model per refit epoch, trained once across the whole
    # sample before the per-series loop -- see train_ml_models_by_epoch's
    # docstring for why this can't be fit per-series like the other two.
    ml_models, ml_critical_ratio = train_ml_models_by_epoch(
        series_map, meta_by_id, prices_series, wm_yr_wk, event_days,
        holding_rate=holding_rate, stockout_multiplier=stockout_multiplier,
        ml_training_window_days=ml_training_window_days,
    )

    missing_price = []
    results = []
    for sid in ids:
        m = meta_by_id[sid]
        key = (m["store_id"], m["item_id"])
        price = prices.get(key)
        if price is None:
            missing_price.append(sid)
            continue
        series = series_map[sid]
        r = run_one_series(
            sid, series, event_days, price, seed=0,
            cat_id=m["cat_id"], state_id=m["state_id"], ml_models=ml_models,
            prices_series=prices_series, wm_yr_wk=wm_yr_wk, series_key=key,
            holding_rate=holding_rate, stockout_multiplier=stockout_multiplier, check_coverage=check_coverage,
            opening_inventory_mode=opening_inventory_mode, planner_seed_base=planner_seed_base,
            aci_gamma=aci_gamma, fuzzy_enabled=fuzzy_enabled, mcts_horizon_periods=mcts_horizon_periods,
            integer_order_units=integer_order_units,
        )
        r.update({"cat_id": m["cat_id"], "state_id": m["state_id"], "store_id": m["store_id"]})
        results.append(r)

    agg_coverage = {}
    if check_coverage:
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
    sba_costs = np.array([r["sba_cost"] for r in results])
    ss_costs = np.array([r["ss_cost"] for r in results])
    ml_costs = np.array([r["ml_cost"] for r in results])
    fused_costs = np.array([r["fused_cost"] for r in results])

    def _pct_and_n_better(a, b):
        n_better = int((a < b).sum())
        pct = 100.0 * (1.0 - a.sum() / b.sum()) if b.sum() else 0.0
        return pct, n_better

    mean_cost_ratio_pct, n_fused_better = _pct_and_n_better(fused_costs, baseline_costs)
    mean_cost_ratio_pct_vs_sba, n_fused_better_vs_sba = _pct_and_n_better(fused_costs, sba_costs)
    mean_cost_ratio_pct_vs_ss, n_fused_better_vs_ss = _pct_and_n_better(fused_costs, ss_costs)
    mean_cost_ratio_pct_vs_ml, n_fused_better_vs_ml = _pct_and_n_better(fused_costs, ml_costs)

    significance = [
        significance_test(fused_costs, baseline_costs, "fused vs naive Gaussian order-up-to"),
        significance_test(fused_costs, sba_costs, "fused vs SBA/Croston order-up-to"),
        significance_test(fused_costs, ss_costs, "fused vs tuned (s,S) [simulation-optimized]"),
        significance_test(fused_costs, ml_costs, "fused vs pooled ML quantile (LightGBM newsvendor)"),
        significance_test(sba_costs, baseline_costs, "SBA/Croston vs naive Gaussian order-up-to"),
        significance_test(ss_costs, baseline_costs, "tuned (s,S) vs naive Gaussian order-up-to"),
        significance_test(ml_costs, baseline_costs, "pooled ML quantile vs naive Gaussian order-up-to"),
    ]

    return {
        "sample_seed": sample_seed,
        "planner_seed_base": planner_seed_base,
        "aci_gamma": aci_gamma,
        "fuzzy_enabled": fuzzy_enabled,
        "mcts_horizon_periods": mcts_horizon_periods,
        "integer_order_units": integer_order_units,
        "opening_inventory_mode": opening_inventory_mode,
        "stratification_zero_fraction_days_end_exclusive": TEST_START,
        "holding_rate": holding_rate,
        "stockout_multiplier": stockout_multiplier,
        "ml_training_window_days": ml_training_window_days,
        "ml_critical_ratio": ml_critical_ratio,
        "n_sampled": len(ids),
        "n_evaluated": len(results),
        "n_missing_price": len(missing_price),
        "missing_price_ids": missing_price,
        "strata_report": strata_report,
        "conformal_coverage": agg_coverage,
        "baseline_cost_total": float(baseline_costs.sum()),
        "sba_cost_total": float(sba_costs.sum()),
        "ss_cost_total": float(ss_costs.sum()),
        "ml_cost_total": float(ml_costs.sum()),
        "fused_cost_total": float(fused_costs.sum()),
        "n_fused_better": n_fused_better,
        "n_fused_better_vs_sba": n_fused_better_vs_sba,
        "n_fused_better_vs_ss": n_fused_better_vs_ss,
        "n_fused_better_vs_ml": n_fused_better_vs_ml,
        "n_series": len(results),
        "mean_cost_change_pct_aggregate": mean_cost_ratio_pct,
        "mean_cost_change_pct_aggregate_vs_sba": mean_cost_ratio_pct_vs_sba,
        "mean_cost_change_pct_aggregate_vs_ss": mean_cost_ratio_pct_vs_ss,
        "mean_cost_change_pct_aggregate_vs_ml": mean_cost_ratio_pct_vs_ml,
        "significance": significance,
        "per_series": results,
    }


def significance_test(a, b, label):
    """Paired Wilcoxon signed-rank test on per-series cost (a=fused,
    b=comparator) + a percentile bootstrap CI on the aggregate
    (sum-based) percent cost change, both computed on the SAME sample --
    not a claim about the full M5 catalog, but a statement of whether this
    sample's result is distinguishable from noise at this sample size."""
    diff = a - b
    if np.allclose(diff, 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
    rng = np.random.default_rng(0)
    n = len(a)
    boot_pcts = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        bsum, asum = b[idx].sum(), a[idx].sum()
        boot_pcts.append(100.0 * (1.0 - asum / bsum) if bsum != 0 else 0.0)
    lo, hi = np.percentile(boot_pcts, [2.5, 97.5])
    return {
        "comparison": label,
        "wilcoxon_signed_rank_p": wilcoxon_p,
        "bootstrap_95ci_pct_change": [float(lo), float(hi)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the M5 inventory policy evaluation")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--n-per-stratum", type=int, default=N_PER_STRATUM)
    parser.add_argument("--opening-inventory-mode", choices=("gaussian_target", "zero"), default="gaussian_target")
    parser.add_argument("--planner-seed-base", type=int, default=0)
    parser.add_argument("--integer-order-units", action="store_true")
    parser.add_argument(
        "--ml-training-window-days",
        type=int,
        default=FIT_WINDOW + CAL_WINDOW,
        help="Pooled LightGBM history length in days (default: 270; use 180 for the matched-window sensitivity)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    t0 = time.time()
    meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id = load_shared_data()
    out = evaluate_sample(
        args.sample_seed, meta, zfracs, prices, prices_series, wm_yr_wk, event_days, meta_by_id,
        n_per_stratum=args.n_per_stratum,
        opening_inventory_mode=args.opening_inventory_mode,
        planner_seed_base=args.planner_seed_base,
        integer_order_units=args.integer_order_units,
        ml_training_window_days=args.ml_training_window_days,
    )
    elapsed = time.time() - t0

    out["config"] = {
        "test_days": TEST_DAYS,
        "review_period_days": REVIEW_PERIOD_DAYS,
        "fit_window": FIT_WINDOW,
        "cal_window": CAL_WINDOW,
        "refit_period_days": REFIT_PERIOD_DAYS,
        "aci_gamma": ACI_GAMMA,
        "operational_level": OPERATIONAL_LEVEL,
        "annual_holding_rate": ANNUAL_HOLDING_RATE,
        "sample_seed": args.sample_seed,
        "n_per_stratum": args.n_per_stratum,
        "opening_inventory_mode": args.opening_inventory_mode,
        "planner_seed_base": args.planner_seed_base,
        "integer_order_units": args.integer_order_units,
        "ml_training_window_days": args.ml_training_window_days,
        "stratification_zero_fraction_days_end_exclusive": TEST_START,
    }
    out["runtime_seconds"] = elapsed

    out_path = args.output or (Path(__file__).resolve().parent.parent / "research" / "m5_evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    cov90 = out["conformal_coverage"]["0.9"]["mean_empirical_coverage"]
    print(
        f"n_series={out['n_series']} (missing price: {out['n_missing_price']}) "
        f"coverage@90%={cov90:.3f} "
        f"baseline_total={out['baseline_cost_total']:.1f} sba_total={out['sba_cost_total']:.1f} "
        f"ss_total={out['ss_cost_total']:.1f} ml_total={out['ml_cost_total']:.1f} "
        f"fused_total={out['fused_cost_total']:.1f} "
        f"vs_naive={out['mean_cost_change_pct_aggregate']:.1f}% (n_better={out['n_fused_better']}/{out['n_series']}) "
        f"vs_sba={out['mean_cost_change_pct_aggregate_vs_sba']:.1f}% "
        f"(n_better={out['n_fused_better_vs_sba']}/{out['n_series']}) "
        f"vs_ss={out['mean_cost_change_pct_aggregate_vs_ss']:.1f}% "
        f"(n_better={out['n_fused_better_vs_ss']}/{out['n_series']}) "
        f"vs_ml={out['mean_cost_change_pct_aggregate_vs_ml']:.1f}% "
        f"(n_better={out['n_fused_better_vs_ml']}/{out['n_series']}) "
        f"runtime={elapsed:.1f}s"
    )
    for s in out["significance"]:
        print(f"  {s['comparison']}: wilcoxon p={s['wilcoxon_signed_rank_p']:.4g} "
              f"bootstrap 95% CI on %% change={s['bootstrap_95ci_pct_change']}")
    print(f"Full results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
