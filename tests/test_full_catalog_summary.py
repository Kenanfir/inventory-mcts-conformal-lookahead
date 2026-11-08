import pytest

from summarize_m5_full_catalog import summarize


def _row(series_id, category, fused, ml):
    return {
        "series_id": series_id,
        "cat_id": category,
        "state_id": "CA",
        "store_id": "CA_1",
        "baseline_cost": 20.0,
        "sba_cost": 18.0,
        "ss_cost": 15.0,
        "ml_cost": ml,
        "fused_cost": fused,
        "coverage_by_level": {"0.9": {"empirical_coverage": 0.9}},
    }


def _payload():
    return {
        "n_evaluated": 2,
        "n_missing_price": 0,
        "strata_report": [{"n_available": 2, "n_picked": 2}],
        "config": {"integer_order_units": True},
        "per_series": [
            _row("a", "FOODS", 5.0, 10.0),
            _row("b", "HOUSEHOLD", 15.0, 10.0),
        ],
    }


def test_summarize_reports_aggregate_effect_and_win_share():
    result = summarize(_payload())

    comparison = result["mcts_comparisons"]["pooled_lightgbm"]
    assert result["all_available_series_selected"] is True
    assert result["aggregate_costs"]["mcts_lookahead"] == 20.0
    assert comparison["aggregate_cost_reduction_pct"] == 0.0
    assert comparison["series_won"] == 1
    assert comparison["series_win_share"] == 0.5
    assert result["by_category"]["FOODS"]["mcts_cost_reduction_vs_lightgbm_pct"] == 50.0


def test_summarize_rejects_a_sampled_stratum():
    payload = _payload()
    payload["strata_report"][0]["n_picked"] = 1

    with pytest.raises(ValueError, match="not a full-catalog census"):
        summarize(payload)
