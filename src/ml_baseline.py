"""Pooled LightGBM quantile-regression baseline: a data-driven newsvendor.

Added alongside `ss_baseline.py` after a live impact-benchmark
(research/IMPACT_BENCHMARK.md) found this project's baseline set had no
ML-driven comparator, while the field's dataset-scale ceiling paper in the
same recent IJPE volume (Long et al. 2025, IJPE 279) uses exactly this
method family (quantile LightGBM) on M5 among other datasets. The pattern
itself -- train a quantile forecaster, order at the newsvendor critical
ratio -- is standard, citable practice (Trapero, Cardos & Kourentzes 2019,
IJF; a public real-retail-data implementation reviewed in
research/STRENGTHENING_OPTIONS.md option 2), not invented for this project.

Design, stated plainly for what it is:
  - ONE global (pooled) quantile model trained across all sampled series,
    not one model per series -- the standard practice at this data volume
    (also what Long et al. and the reviewed reference implementation do),
    and the only practical choice for very short/sparse individual M5
    series.
  - Target: total demand over the coming REVIEW_PERIOD_DAYS-day period.
  - Quantile: the newsvendor critical ratio u / (u + h), u = stockout cost
    per unit, h = holding cost per unit -- i.e. the model predicts the
    demand level that exactly balances this project's own real cost
    asymmetry, used directly as the order-up-to level (no separate z*sigma
    step, unlike the other two baselines -- the model's quantile output IS
    the safety-stock-inclusive target level).
  - Features: only information available strictly before the day being
    predicted for (lagged/rolling demand statistics, calendar features
    known in advance, price, category/state) -- no lookahead, same
    discipline the conformal calibrator and both other baselines already
    enforce.
  - Refit cadence: same 28-day cycle as the other two baselines and the
    conformal calibrator, retraining on all pooled data observed up to
    that point.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

import m5_loader

LAG_LENGTHS = (7, 14, 21, 28)
ROLLING_WINDOW = 28


def _series_features_row(series: np.ndarray, event_days: np.ndarray, day: int, review_period_days: int,
                          price: float, cat_id: str, state_id: str):
    """Features for the review period starting at `day` (predicting
    sum(series[day:day+review_period_days])), using only series[:day] and
    calendar knowledge (event_days is deterministic, known in advance, not
    a demand lookahead).
    """
    history = series[:day]
    if len(history) < ROLLING_WINDOW:
        return None
    lags = {}
    for lag in LAG_LENGTHS:
        if day - lag < 0:
            return None
        lags[f"lag_{lag}"] = float(series[day - lag:day].sum())
    recent = history[-ROLLING_WINDOW:]
    target_event_days = event_days[day:day + review_period_days]
    return {
        **lags,
        "roll_mean": float(recent.mean()),
        "roll_std": float(recent.std()),
        "roll_zero_frac": float((recent == 0).mean()),
        "weekday_start": int(day % 7),
        "n_event_days_in_period": int(target_event_days.sum()) if len(target_event_days) else 0,
        "price": float(price),
        "cat_id": cat_id,
        "state_id": state_id,
    }


def build_training_table(series_map: dict, meta_by_id: dict, prices_series: dict, wm_yr_wk: np.ndarray,
                          event_days: np.ndarray, review_period_days: int, train_end_day: int, start_day: int = 0):
    """Pooled training table across every series in `series_map`, using only
    review periods whose target (sum(series[d:d+review_period_days])) is
    fully observed before `train_end_day` -- i.e. d + review_period_days <=
    train_end_day. Returns (DataFrame of features, Series of targets).

    `prices_series`/`wm_yr_wk` (from m5_loader.load_prices_series /
    load_calendar) give each row's `price` feature as of day `d` itself
    (m5_loader.mean_price_as_of), not a single whole-history mean -- fixed
    2026-09-23 after independent code review found the whole-history mean
    (m5_loader.load_prices_index) leaks future price information into this
    model's predictive features (it's fine as a cost-model coefficient,
    used identically by every policy, but not as a feature only this
    baseline consumes).
    """
    rows = []
    targets = []
    for sid, series in series_map.items():
        m = meta_by_id[sid]
        key = (m["store_id"], m["item_id"])
        d = start_day
        while d + review_period_days <= train_end_day:
            price = m5_loader.mean_price_as_of(prices_series, key, int(wm_yr_wk[d]))
            if price is not None:
                feat = _series_features_row(series, event_days, d, review_period_days, price,
                                             m["cat_id"], m["state_id"])
                if feat is not None:
                    rows.append(feat)
                    targets.append(float(series[d:d + review_period_days].sum()))
            d += review_period_days
    if not rows:
        return None, None
    X = pd.DataFrame(rows)
    for col in ("cat_id", "state_id"):
        X[col] = X[col].astype("category")
    y = pd.Series(targets)
    return X, y


def train_quantile_model(X: pd.DataFrame, y: pd.Series, alpha: float):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        verbosity=-1,
    )
    model.fit(X, y, categorical_feature=["cat_id", "state_id"])
    return model


def predict_order_up_to_level(model, series: np.ndarray, event_days: np.ndarray, day: int,
                               review_period_days: int, price: float, cat_id: str, state_id: str) -> float:
    """`price` here should already be the walk-forward price as-of `day`
    (m5_loader.mean_price_as_of(prices_series, key, wm_yr_wk[day])), for the
    same reason build_training_table takes prices_series/wm_yr_wk rather
    than a single whole-history mean -- see that function's docstring."""
    feat = _series_features_row(series, event_days, day, review_period_days, price, cat_id, state_id)
    if feat is None:
        return 0.0
    X = pd.DataFrame([feat])
    for col in ("cat_id", "state_id"):
        X[col] = X[col].astype("category")
    pred = model.predict(X)[0]
    return max(0.0, float(pred))


def order_quantity(current_inventory: float, target_level: float) -> float:
    return max(0.0, target_level - current_inventory)
