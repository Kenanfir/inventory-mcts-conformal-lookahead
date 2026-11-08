"""Simulation-optimized (s, S) policy, driving a third, "tuned OR method"
baseline.

Added after a live impact-benchmark against 10 comparator papers
(research/IMPACT_BENCHMARK.md) found this project's baseline depth (two
classical heuristics, no tuned optimization method, no ML/RL comparator) was
the single dimension on which every fully-confirmed comparator paper in the
subfield does more than this project did -- and the manuscript's own
Conclusion already named "a properly tuned (s, S) policy" as unfinished
future work (research/STRENGTHENING_OPTIONS.md option 1).

Why simulation-optimization, not the classical exact algorithm (Scarf 1960;
Zheng & Federgruen 1991): those results are proven for the BACKORDER case.
This project's cost model is LOST-SALES (unmet demand is simply lost, see
`mcts_policy._step_cost`), and the lost-sales (s, S) problem has no general
closed-form solution -- it remains active research (Žic & Žic 2026, and
multiple "Lost-sales inventory theory" review papers, per
STRENGTHENING_OPTIONS.md). Importing the backorder-case exact algorithm onto
lost-sales data would be an undocumented theory mismatch. Simulation-
optimization sidesteps this entirely: it searches (s, S) pairs directly
against the SAME lost-sales cost function every other policy in this project
is judged on, so there is no theoretical claim being made beyond "this pair
was the best of the ones tried on the fitting window" -- a numerically tuned
baseline, not a claimed-optimal one, and described as such wherever cited.
"""

import numpy as np


def _simulate_cost(demand: np.ndarray, review_period_days: int, s: float, S: float,
                    holding_cost: float, stockout_cost: float, initial_inventory: float = None) -> float:
    """Total lost-sales cost of a periodic-review (s, S) policy on `demand`,
    using the identical cost formula the rest of this project uses
    (holding_cost * leftover + stockout_cost * unmet per period). Reviews
    every `review_period_days`; at each review, orders up to S if inventory
    at or below s, otherwise orders nothing.
    """
    if initial_inventory is None:
        initial_inventory = S
    inv = initial_inventory
    total_cost = 0.0
    t = 0
    n = len(demand)
    while t < n:
        period_len = min(review_period_days, n - t)
        order_qty = (S - inv) if inv <= s else 0.0
        order_qty = max(0.0, order_qty)
        period_demand = float(demand[t:t + period_len].sum())
        inv_after = inv + order_qty
        sold = min(inv_after, period_demand)
        leftover = inv_after - sold
        unmet = period_demand - sold
        total_cost += holding_cost * leftover + stockout_cost * unmet
        inv = leftover
        t += period_len
    return total_cost


def fit_ss(train: np.ndarray, review_period_days: int, holding_cost: float, stockout_cost: float,
           s_multipliers=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
           extra_multipliers=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)):
    """Grid search (s, S) pairs, scaled to `train`'s own period demand mean
    (so the grid adapts to each series' demand level rather than using fixed
    absolute units across series spanning orders of magnitude in scale), and
    return the pair minimizing simulated cost on `train`. `S = s + extra`,
    `extra >= 0`, so S >= s always holds by construction.

    Returns (best_s, best_S, best_cost_on_train).
    """
    if len(train) == 0 or train.sum() == 0:
        return 0.0, 0.0, 0.0
    period_mean = float(train.mean()) * review_period_days
    period_mean = max(period_mean, 1e-6)

    best = None
    for sm in s_multipliers:
        s = period_mean * sm
        for em in extra_multipliers:
            S = s + period_mean * em
            cost = _simulate_cost(train, review_period_days, s, S, holding_cost, stockout_cost)
            if best is None or cost < best[2]:
                best = (s, S, cost)
    return best


def order_quantity(current_inventory: float, s: float, S: float) -> float:
    if current_inventory <= s:
        return max(0.0, S - current_inventory)
    return 0.0
