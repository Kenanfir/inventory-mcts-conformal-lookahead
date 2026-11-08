# Data — M5 Forecasting Accuracy dataset

- **Source:** Zenodo, DOI [10.5281/zenodo.12636070](https://zenodo.org/records/12636070),
  uploaded by "Kaggle" 2024-07-03 — a public mirror of the original M5
  competition data, no account/login required (unlike the original Kaggle
  competition page, which does gate behind a Kaggle account for the same
  files).
- **License status:** the Zenodo landing page does not display a license,
  while its DataCite metadata record lists CC BY 4.0 and Kaggle's official
  competition data page says the files are subject to the competition
  rules. Treat the reuse terms as ambiguous; cite the DOI and do not
  redistribute the downloaded files with a submission package.
- **Size:** ~48 MB zipped (`m5-forecasting-accuracy.zip`,
  md5 `86f57416a314197f40a17cc6fc60cbb4` per the Zenodo record — verify on
  download).
- **Contents:**
  - `sales_train_validation.csv` — daily unit sales per item x store, wide
    format (one column per day, d_1-d_1913), columns `id`, `item_id`,
    `dept_id`, `cat_id`, `store_id`, `state_id`. 30,490 bottom-level
    item-store series. The often-cited 42,840 count includes hierarchical
    aggregate series as well as bottom-level series; it is not the row
    count of this CSV.
  - `calendar.csv` — date, wm_yr_wk, weekday/wday/month/year, two event
    columns, 3 binary SNAP flags per state.
  - `sell_prices.csv` — store_id, item_id, wm_yr_wk, weekly average
    sell_price.
- **Backup path if Zenodo ever goes down:** Nixtla's `datasetsforecast`
  package pulls the same data from its own public GitHub mirror
  (github.com/Nixtla/m5-forecasts), also no auth.
- **Backup domain dataset (if the inventory framing hits a real blocker):**
  UCI "Individual Household Electric Power Consumption"
  (archive.ics.uci.edu/dataset/235), no login, CC BY 4.0, ~2.07M minute-level
  readings, single household — thinner than M5 for a paper on its own, and
  it's a single house not a community/grid dataset. ENTSO-E has real
  grid-scale data but needs a registered account plus a manual ~3-business-day
  email approval for API access — a real (if small) friction point, so it's
  the second-choice backup, not the first.

## Verified this session

Downloaded to `research/data/m5-forecasting-accuracy.zip` — see the download
log / checksum verification in the project's build notes for confirmation the
md5 matches the Zenodo record.

Source: feasibility research agent, this session, 2026-09-22.

## Source checks — 2026-10-26

- The local `sales_train_evaluation.csv` and `sales_train_validation.csv`
  each contain 30,490 data rows. The evaluation file contains 1,941 daily
  columns; the validation file contains 1,913.
- The [Zenodo record](https://zenodo.org/records/12636070)'s displayed
  license field is blank. Its [DataCite API metadata](https://api.datacite.org/dois/10.5281/zenodo.12636070)
  lists CC BY 4.0, while [Kaggle's official M5 data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
  says the files are subject to the competition rules. This mismatch is
  unresolved; do not bundle the CSVs or ZIP in a submission or public code
  archive.
- The ZIP checksum recorded above matches the Zenodo record's published
  MD5. This validates file identity, not the right to redistribute it.
