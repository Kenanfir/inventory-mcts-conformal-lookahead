# Research and analysis scripts

This directory is part of the canonical project layout.

- `run_m5_evaluation.py` runs one 324-series sample, with `--integer-order-units`
  for the matched whole-unit specification.
- For a full-catalog census, set `--n-per-stratum 100000`; when every stratum
  is fully selected, this evaluates all 30,490 M5 series. The run uses the same
  test window and policy settings as the sample analysis.
- `summarize_m5_full_catalog.py` validates census completeness and produces a
  finite-benchmark descriptive summary with aggregate costs and subgroup
  breakdowns. It does not interpret the series as independent observations.
- `run_m5_component_ablation.py` runs matched policy-component ablations on
  one sample; `--integer-order-units` selects the whole-unit specification.
- `run_m5_distribution_shift.py` runs one constructed-shock evaluation;
  `--integer-order-units` selects whole-unit orders and `--aci-gamma 0`
  disables online adaptation while preserving periodic refits.
- `run_m5_integer_distribution_shift_suite.py` runs the matched whole-unit
  shock and ACI-ablation cases for sample seeds 0, 1, 4, and 7.
- `summarize_m5_integer_distribution_shift.py` checks and summarizes those
  paired results, including per-series ACI comparisons.
- `summarize_m5_integer_order_robustness.py` summarizes whole-unit result
  files for seeds 0–7 against the stored continuous-order sample results.
- `run_m5_integer_order_sensitivity.py` repeats cost, opening-state, and
  planner-seed checks under the whole-unit specification.
- `run_m5_integer_multisample_cost_sensitivity.py` repeats the key whole-unit
  cost scenarios across sample seeds 0–7, reusing the completed seed-0 sweep.
- `run_m5_robustness.py` runs the original continuous-order sample,
  cost-scenario, opening-inventory, and planner-seed checks.
- `make_figures.py` regenerates manuscript figures from saved project outputs.
- `run_trajectory_example.py --integer-order-units` regenerates representative
  whole-unit inventory trajectories using the headline planner seeds.

Use the package versions and command forms recorded in the project analysis
notes. Results and figure sources are stored under `research/` and
`paper/figures/` respectively.
