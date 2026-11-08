"""Classic order-up-to (base-stock) policy: point forecast + z*std safety
stock, no MCTS / fuzzy / conformal. This is the non-fused comparison point."""

import numpy as np


def order_up_to_level(fit_window: np.ndarray, review_period_days: int, z: float = 1.28) -> float:
    mean = float(fit_window.mean())
    std = float(fit_window.std())
    period_mean = mean * review_period_days
    period_std = std * np.sqrt(review_period_days)
    return max(period_mean + z * period_std, 0.0)


def order_quantity(current_inventory: float, s_level: float) -> float:
    return max(0.0, s_level - current_inventory)
