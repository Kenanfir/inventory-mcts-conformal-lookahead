import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import conformal
import m5_loader
import mcts_policy
import ml_baseline
import run_m5_evaluation as m5_evaluation


def _training_table(series):
    series_map = {"item_store": np.asarray(series, dtype=float)}
    meta_by_id = {
        "item_store": {
            "store_id": "store_1",
            "item_id": "item_1",
            "cat_id": "FOODS",
            "state_id": "CA",
        }
    }
    prices = {("store_1", "item_1"): (np.array([100, 101]), np.array([2.0, 4.0]))}
    wm_yr_wk = np.array([100] * 35 + [101] * (len(series) - 35))
    event_days = np.zeros(len(series), dtype=bool)
    return ml_baseline.build_training_table(
        series_map,
        meta_by_id,
        prices,
        wm_yr_wk,
        event_days,
        review_period_days=7,
        train_end_day=42,
        start_day=28,
    )


def test_training_table_excludes_targets_and_features_after_cutoff():
    original = np.arange(60, dtype=float)
    changed_future = original.copy()
    changed_future[42:] = -10000

    X_a, y_a = _training_table(original)
    X_b, y_b = _training_table(changed_future)

    pd.testing.assert_frame_equal(X_a, X_b)
    pd.testing.assert_series_equal(y_a, y_b)
    assert y_a.tolist() == [sum(range(28, 35)), sum(range(35, 42))]
    assert X_a["price"].tolist() == [2.0, 3.0]


def test_price_as_of_ignores_future_price_weeks():
    prices = {("store_1", "item_1"): (np.array([100, 101, 102]), np.array([2.0, 4.0, 100.0]))}

    assert m5_loader.mean_price_as_of(prices, ("store_1", "item_1"), 101) == 3.0


def test_rolling_conformal_refit_uses_only_predecision_observations():
    series = np.arange(30, dtype=float)
    changed_future = series.copy()
    changed_future[20:] = 99999
    events = np.zeros(len(series), dtype=bool)

    first = conformal.RollingConformalCalibrator(series, 10, 5, 4, 0.9, 0.03, events)
    second = conformal.RollingConformalCalibrator(changed_future, 10, 5, 4, 0.9, 0.03, events)
    first.refit_if_due(20)
    second.refit_if_due(20)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.forecast_days(20, 4), second.forecast_days(20, 4))
    assert first.current_q() == second.current_q()


def test_integer_mcts_action_is_reproducible_and_nonnegative():
    def choose(seed):
        return mcts_policy.plan_action(
            np.random.default_rng(seed),
            inventory=3,
            forecasts_per_day=np.full(21, 4.0),
            cal_q=1.0,
            risk_mult=1.0,
            review_period_days=7,
            horizon_periods=3,
            action_base_multipliers=[0.0, 0.5, 1.0],
            n_simulations=15,
            extra_candidate_levels=[30.0],
            integer_order_units=True,
        )

    order_a = choose(12)
    order_b = choose(12)
    assert order_a == order_b
    assert order_a >= 0
    assert order_a.is_integer()


def test_ml_training_window_can_match_conformal_fit_window():
    calls = []

    def capture_training_window(*args, **kwargs):
        calls.append(kwargs)
        return None, None

    with patch.object(m5_evaluation.ml_baseline, "build_training_table", capture_training_window):
        m5_evaluation.train_ml_models_by_epoch(
            {}, {}, {}, np.array([]), np.array([]), ml_training_window_days=180
        )

    first_anchor = next(iter(m5_evaluation._ml_epoch_anchor_days(
        m5_evaluation.TEST_START,
        m5_evaluation.TEST_DAYS,
        m5_evaluation.REVIEW_PERIOD_DAYS,
        m5_evaluation.REFIT_PERIOD_DAYS,
    ).values()))
    assert calls[0]["start_day"] == first_anchor - 180
    assert calls[0]["train_end_day"] == first_anchor


def test_ml_training_window_defaults_to_existing_270_day_history():
    calls = []

    def capture_training_window(*args, **kwargs):
        calls.append(kwargs)
        return None, None

    with patch.object(m5_evaluation.ml_baseline, "build_training_table", capture_training_window):
        m5_evaluation.train_ml_models_by_epoch({}, {}, {}, np.array([]), np.array([]))

    assert calls[0]["start_day"] == m5_evaluation.TEST_START - (
        m5_evaluation.FIT_WINDOW + m5_evaluation.CAL_WINDOW
    )


def test_ml_training_window_rejects_nonpositive_lengths():
    with pytest.raises(ValueError, match="positive integer"):
        m5_evaluation.train_ml_models_by_epoch(
            {}, {}, {}, np.array([]), np.array([]), ml_training_window_days=0
        )
