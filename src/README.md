# Src

Ported from the fail-fast check at
`workspace/tmp/inventory-mcts-fuzzy-conformal-checkfast/` (see
`research/FAIL_FAST_FINDINGS.md` for what was validated and what wasn't).

**Conformal calibration fixed (2026-09-22).** `conformal.py`'s original
static split-conformal calibration undercovered under demand-trend drift
(84.8% empirical vs 90% nominal — see `research/FAIL_FAST_FINDINGS.md`).
Fixed with `RollingConformalCalibrator` (periodic refit + Adaptive Conformal
Inference): 89.3% empirical coverage on the same synthetic check. See
`research/ADAPTIVE_CONFORMAL.md` for the method and citation.

**Important correction**: fixing the calibration bug removed the
previously-reported 28.3% synthetic cost advantage over baseline (now a
statistical tie, 925.7 vs 925.9 mean cost, 23/64 series won instead of
33/64) — the old number was partly an artifact of intervals being too
narrow, not a clean signal of the fused system's value. Read
`research/ADAPTIVE_CONFORMAL.md`'s "Correction to the record" section before
citing any pre-fix number. This does not predict the real-M5 outcome either
way; it resets the baseline expectation before Part 2 (real data) runs.

- `synthetic_demand.py` — synthetic series generator, kept for regression
  testing the pipeline against a known-shape input, not for producing paper
  results.
- `conformal.py` — includes both the original static split-conformal
  functions (kept for the before/after comparison in
  `ADAPTIVE_CONFORMAL.md`) and `RollingConformalCalibrator`, the calibrator
  actually used downstream.
- `fuzzy_risk.py`, `mcts_policy.py`, `baseline_policy.py` — unchanged from
  the fail-fast check.

`scripts/run_checkfast_fixed.py` is the fixed-calibration re-run of the
synthetic fail-fast check. Still needed: an M5 data loader and a real-data
run script (Part 2 of
`docs/handoff/contracts/23-inventory-mcts-fuzzy-conformal-build.md`).
