"""Synthetic multi-series demand generator, standing in for M5 until the real
dataset is wired up. Each series mixes trend, weekly seasonality, and
intermittency (zero-inflated counts), which is the shape that makes M5 hard
(the M5 uncertainty track's own writeup flags widespread intermittency)."""

import numpy as np


def generate_series(seed: int, n_days: int = 150) -> np.ndarray:
    rng = np.random.default_rng(seed)

    base_level = rng.uniform(2.0, 20.0)
    trend_per_day = rng.uniform(-0.01, 0.02) * base_level
    weekday_mult = rng.uniform(0.6, 1.6, size=7)
    weekday_mult = weekday_mult / weekday_mult.mean()
    zero_inflation_p = rng.uniform(0.0, 0.5)  # fraction of days forced to 0
    noise_scale = rng.uniform(0.15, 0.45)

    days = np.arange(n_days)
    weekday = days % 7
    rate = (base_level + trend_per_day * days) * weekday_mult[weekday]
    rate = np.clip(rate, 0.05, None)
    rate = rate * rng.lognormal(mean=0.0, sigma=noise_scale, size=n_days)

    demand = rng.poisson(rate).astype(float)
    zero_mask = rng.uniform(size=n_days) < zero_inflation_p
    demand[zero_mask] = 0.0

    return demand


def generate_batch(n_series: int = 64, n_days: int = 150, base_seed: int = 0):
    return [generate_series(base_seed + i, n_days) for i in range(n_series)]
