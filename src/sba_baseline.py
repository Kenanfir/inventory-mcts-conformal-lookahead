"""Syntetos-Boylan Approximation (SBA) of Croston's method, driving an
order-up-to safety-stock baseline for intermittent demand.

Added after the naive Gaussian order-up-to baseline (`baseline_policy.py`)
was found to only achieve 84.3% actual service level in aggregate on real
M5 demand (vs the ~90% its z=1.28 coefficient nominally targets) -- a known,
textbook weakness of a symmetric-normal safety-stock formula on skewed,
zero-inflated (intermittent) demand. See research/M5_EVALUATION.md. Croston's
method (1972) and its bias-corrected SBA variant (Syntetos & Boylan 2005)
are the standard forecasting approach for exactly this demand pattern, so
they are the fairer baseline for "is the fused system actually better than
a reasonable/standard method, not just a weak one."

Croston/SBA forecasts two smoothed quantities, updated ONLY on periods with
non-zero demand (this is what makes it suited to intermittent series, unlike
a plain exponential smoother that would drag the estimate toward zero on
every zero-demand day):
  z_t: smoothed non-zero demand size
  p_t: smoothed inter-demand interval (periods between non-zero demands)
Forecast rate (demand per period) = z_t / p_t (Croston) or
(1 - alpha/2) * z_t / p_t (SBA -- corrects Croston's known upward bias).
The forecast is flat between non-zero-demand events, by construction.
"""

import numpy as np


def sba_fit(train: np.ndarray, alpha: float = 0.1):
    """Fit SBA on `train` (daily demand). Returns the forecast RATE (demand
    per day) implied by the state at the end of the training window, to be
    used as a constant forecast going forward until the next refit.
    """
    nonzero_idx = np.nonzero(train)[0]
    if len(nonzero_idx) == 0:
        return 0.0
    z = float(train[nonzero_idx[0]])
    p = float(nonzero_idx[0] + 1) if nonzero_idx[0] > 0 else 1.0
    last_idx = nonzero_idx[0]
    for idx in nonzero_idx[1:]:
        q = idx - last_idx
        z = z + alpha * (train[idx] - z)
        p = p + alpha * (q - p)
        last_idx = idx
    if p <= 0:
        return 0.0
    rate = (1.0 - alpha / 2.0) * z / p
    return max(0.0, float(rate))


def sba_order_up_to_level(train: np.ndarray, review_period_days: int, alpha: float = 0.1, z: float = 1.28) -> float:
    """SBA-forecast-driven order-up-to level: period_mean from the SBA rate
    (intermittency-aware) instead of the raw sample mean, period_std from
    the empirical residuals of (actual - SBA rate) over the training window
    (a standard practical shortcut in place of the full closed-form
    Syntetos-Boylan variance formula, which needs additional distributional
    assumptions -- documented here rather than silently substituted).
    """
    rate = sba_fit(train, alpha=alpha)
    resid_std = float(np.std(train - rate)) if len(train) else 0.0
    period_mean = rate * review_period_days
    period_std = resid_std * np.sqrt(review_period_days)
    return max(period_mean + z * period_std, 0.0)


def order_quantity(current_inventory: float, s_level: float) -> float:
    return max(0.0, s_level - current_inventory)
