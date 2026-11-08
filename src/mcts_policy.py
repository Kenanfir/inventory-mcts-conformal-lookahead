"""Lightweight MCTS ordering-decision planner.

Deliberately cheap for a fail-fast check: full UCB1 tree search only at the
ROOT action choice (the decision actually acted on), with a fixed
order-up-to-forecast rollout policy for the periods beyond that inside each
simulation. That is a documented simplification of "real" MCTS (which would
keep branching at every depth) traded for speed; it still lets multi-period
lookahead inform the one decision that matters, which is what this check is
testing.
"""

import numpy as np

HOLDING_COST = 1.0
STOCKOUT_COST = 5.0


def _step_cost(inventory_before: float, order_qty: float, demand: float, holding_cost: float, stockout_cost: float):
    inv_after_order = inventory_before + order_qty
    sold = min(inv_after_order, demand)
    leftover = inv_after_order - sold
    unmet = demand - sold
    cost = holding_cost * leftover + stockout_cost * unmet
    return cost, leftover  # lost-sales model: inventory clipped at 0


def _period_demand_sample(rng, lo_days: np.ndarray, hi_days: np.ndarray, integer_units: bool = False) -> float:
    draws = rng.uniform(lo_days, hi_days)
    if integer_units:
        draws = np.rint(draws)
    return float(draws.sum())


class _RootStats:
    def __init__(self, n_actions):
        self.visits = np.zeros(n_actions, dtype=int)
        self.cost_sum = np.zeros(n_actions, dtype=float)


def _ucb_select(stats: _RootStats, total_visits: int):
    unexplored = np.where(stats.visits == 0)[0]
    if len(unexplored):
        return int(unexplored[0])
    mean_cost = stats.cost_sum / stats.visits
    bonus = np.sqrt(2 * np.log(total_visits + 1) / stats.visits)
    score = -mean_cost + 2.0 * bonus  # minimizing cost -> maximize -cost + bonus
    return int(np.argmax(score))


def plan_action(
    rng,
    inventory: float,
    forecasts_per_day: np.ndarray,
    cal_q: float,
    risk_mult: float,
    review_period_days: int,
    horizon_periods: int,
    action_base_multipliers,
    n_simulations: int = 15,
    holding_cost: float = HOLDING_COST,
    stockout_cost: float = STOCKOUT_COST,
    extra_candidate_levels=None,
    integer_order_units: bool = False,
) -> float:
    """`holding_cost`/`stockout_cost` default to the module constants (1.0 /
    5.0, a fixed 1:5 ratio), which is what the original synthetic fail-fast
    check used throughout -- passing them keeps that check's numbers
    reproducible. Real per-item evaluations (e.g. M5, where holding and
    stockout cost are both derived from `sell_price`, giving a real ratio of
    roughly 260:1, not 1:5 -- see research/M5_EVALUATION.md's cost-model
    sensitivity section) should pass the actual per-unit costs here, so the
    planner is scored on, and therefore plans for, the same cost structure
    the policy is actually evaluated against.

    `extra_candidate_levels` (default None -- no behavior change, so the
    synthetic fail-fast check in scripts/run_checkfast_fixed.py, which calls
    this with the original argument set, is unaffected) adds absolute
    order-quantity candidates alongside the forecast-multiplier ones. Added
    for M5: under a heavily right-skewed cost ratio, a pure
    multiplier-of-point-forecast grid needs an arbitrarily high multiplier
    ceiling to ever reach the right quantile for high-variance series (see
    research/M5_EVALUATION.md); passing the conformal interval's own upper
    bound (already computed by the caller, not a new estimate) as an anchor
    lets MCTS evaluate distribution-aware order levels directly, still
    selected only if the simulated cost says so.
    """
    lo = np.clip(forecasts_per_day - cal_q, 0.0, None)
    hi = forecasts_per_day + cal_q

    period0_forecast = float(forecasts_per_day[:review_period_days].sum())
    candidates = [round(m * max(period0_forecast, 1.0) * risk_mult, 3) for m in action_base_multipliers]
    if extra_candidate_levels:
        candidates.extend(round(max(0.0, level), 3) for level in extra_candidate_levels)
    if integer_order_units:
        # Retail unit sales are counts. Round candidate orders up so that the
        # planned quantity remains at least as large as the proposed target,
        # then collapse duplicate arms created by integerization.
        candidates = list(dict.fromkeys(float(np.ceil(max(0.0, qty) - 1e-12)) for qty in candidates))

    stats = _RootStats(len(candidates))

    for sim in range(n_simulations):
        a_idx = _ucb_select(stats, sim)
        order_qty = candidates[a_idx]

        sim_inventory = inventory
        total_cost = 0.0

        day0 = slice(0, review_period_days)
        demand0 = _period_demand_sample(rng, lo[day0], hi[day0], integer_order_units)
        cost, sim_inventory = _step_cost(sim_inventory, order_qty, demand0, holding_cost, stockout_cost)
        total_cost += cost

        for p in range(1, horizon_periods):
            day_slice = slice(p * review_period_days, (p + 1) * review_period_days)
            if day_slice.stop > len(forecasts_per_day):
                break
            rollout_order = max(0.0, float(forecasts_per_day[day_slice].sum()) - sim_inventory)
            if integer_order_units:
                rollout_order = float(np.ceil(rollout_order - 1e-12))
            demand = _period_demand_sample(rng, lo[day_slice], hi[day_slice], integer_order_units)
            cost, sim_inventory = _step_cost(sim_inventory, rollout_order, demand, holding_cost, stockout_cost)
            total_cost += cost

        stats.visits[a_idx] += 1
        stats.cost_sum[a_idx] += total_cost

    visited = np.where(stats.visits > 0)[0]
    mean_costs = stats.cost_sum[visited] / stats.visits[visited]
    best = visited[int(np.argmin(mean_costs))]
    return candidates[best]
