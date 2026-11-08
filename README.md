# Monte Carlo lookahead with conformal demand intervals: code and saved results

Code, tests and saved result files for the article "Monte Carlo Lookahead with Conformal Demand Intervals for Retail Inventory Replenishment: A Full-Catalog Simulation Evaluation" (under review).

## Contents
- `src/`: the conformal calibrator, fuzzy risk module, root-action Monte Carlo lookahead, and the four baselines (Gaussian order-up-to, SBA/Croston, tuned (s, S), pooled LightGBM quantile), plus the M5 loader.
- `scripts/`: the evaluation, ablation, sensitivity, shock and census runs, and `make_figures.py` (see `scripts/README.md` for which script produces which result).
- `tests/`: nine regression tests (no-lookahead data cutoffs, price as-of handling, rolling conformal refits, whole-unit action reproducibility).
- `research/*.json`: the saved result files the article's tables and figures are computed from. `research/DATA.md` describes the data.
- `requirements.txt`: pinned versions used for the analyses.

## Not included
The M5 data itself. Download the M5 Forecasting Accuracy files from Zenodo (DOI 10.5281/zenodo.12636070) and place them in `research/data/` (`calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`).

## Run
```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q tests
```
The tests ran green (9 passed) on the archived tree on 2026-11-08. The full-catalog census and the eight-sample runs are long; the saved JSON files hold their outputs.

## Limits
M5 has no margins, cost of goods, inventory positions or observed stockouts, so costs in the scripts are price-based scenarios. See the article for what the results do and do not support.

## License
MIT (see `LICENSE`).
