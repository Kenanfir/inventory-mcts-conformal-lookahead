"""M5 Forecasting Accuracy data loader.

Parses `research/data/sales_train_evaluation.csv` (the fuller of the two
sales files: 1941 days, 2011-01-29 through 2016-06-19, vs 1913 in
`sales_train_validation.csv` -- we use the extra 28 days since this project
is not submitting to the Kaggle competition and just wants the longest real
history) into per-series daily demand arrays shaped like
`synthetic_demand.py`'s output, plus `calendar.csv` for real weekday/event/
SNAP alignment and `sell_prices.csv` for item value.

Day indexing: array index 0 == `d_1` == 2011-01-29 (a Saturday) throughout
this module and downstream scripts, matching `conformal.py`'s `day_index`
convention. Because the sales columns are a complete, gap-free daily
sequence, `day_index % 7` groups the same real day-of-week every time (it is
not an arbitrary synthetic index) -- verified directly against
`calendar.csv`'s own `wday` column in `_verify_weekday_alignment` below,
rather than assumed.
"""

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "research" / "data"
SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_calendar(data_dir: Path = DATA_DIR):
    """Return a dict with parallel arrays indexed by day_index (0-based,
    day_index 0 == d_1), plus a d_col -> day_index map."""
    path = data_dir / CALENDAR_FILE
    dates, wdays, events, snap = {}, [], [], {"CA": [], "TX": [], "WI": []}
    d_to_idx = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        wm_yr_wks = []
        for i, row in enumerate(reader):
            d_to_idx[row["d"]] = i
            wdays.append(int(row["wday"]))
            has_event = bool(row["event_name_1"]) or bool(row["event_name_2"])
            events.append(has_event)
            snap["CA"].append(row["snap_CA"] == "1")
            snap["TX"].append(row["snap_TX"] == "1")
            snap["WI"].append(row["snap_WI"] == "1")
            wm_yr_wks.append(int(row["wm_yr_wk"]))
    return {
        "wday": np.array(wdays, dtype=int),
        "is_event": np.array(events, dtype=bool),
        "snap": {k: np.array(v, dtype=bool) for k, v in snap.items()},
        "d_to_idx": d_to_idx,
        "n_days": len(wdays),
        "wm_yr_wk": np.array(wm_yr_wks, dtype=int),
    }


def _verify_weekday_alignment(calendar) -> bool:
    """Confirm day_index % 7 groups a single true weekday each time, i.e.
    calendar['wday'][i] == calendar['wday'][i % 7] for every i. This is what
    lets `conformal.fit_seasonal_forecaster`'s `weekday_avg[day_index % 7]`
    bucketing be a real-weekday-seasonal average, not an assumption."""
    wday = calendar["wday"]
    return bool(np.all(wday == wday[np.arange(len(wday)) % 7]))


def load_series_meta(data_dir: Path = DATA_DIR):
    """Return a list of dicts (one per series) with id/item_id/dept_id/
    cat_id/store_id/state_id, in file order -- used for stratified sampling
    without loading all demand arrays into memory."""
    path = data_dir / SALES_FILE
    meta = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        meta_idx = [header.index(c) for c in META_COLS]
        for row in reader:
            meta.append({c: row[i] for c, i in zip(META_COLS, meta_idx)})
    return meta


def load_series_by_ids(ids, data_dir: Path = DATA_DIR):
    """Load demand arrays only for the given set of series `id`s. Returns
    dict id -> np.ndarray (float, one value per day, day_index 0 == d_1).
    Single pass over the file; ids not found are simply absent from the
    result (caller should check).
    """
    ids = set(ids)
    path = data_dir / SALES_FILE
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_col = header.index("id")
        d_start = header.index("d_1")
        for row in reader:
            if row[id_col] in ids:
                out[row[id_col]] = np.array(row[d_start:], dtype=float)
    return out


def compute_zero_fracs(data_dir: Path = DATA_DIR, end_day_exclusive: int | None = None):
    """Single pass over the sales file computing each series' zero-demand
    fraction, without retaining any demand array (~30k series x ~1941 days
    would be ~460MB kept in memory; this keeps only one float per series).

    `end_day_exclusive` is the number of initial daily observations to use.
    It supports pre-test-only sample stratification; None preserves the
    historical full-record behavior.
    """
    path = data_dir / SALES_FILE
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_col = header.index("id")
        d_start = header.index("d_1")
        for row in reader:
            n_available = len(row) - d_start
            if end_day_exclusive is None:
                n = n_available
            else:
                if end_day_exclusive <= 0:
                    raise ValueError("end_day_exclusive must be positive")
                n = min(end_day_exclusive, n_available)
            vals = row[d_start:d_start + n]
            zeros = sum(1 for v in vals if v == "0")
            out[row[id_col]] = zeros / n
    return out


def load_prices_index(data_dir: Path = DATA_DIR):
    """Return dict (store_id, item_id) -> mean sell_price across all weeks
    present (a single representative value per series, adequate for a
    per-item holding-cost coefficient -- not a week-by-week price series)."""
    path = data_dir / PRICES_FILE
    sums, counts = {}, {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["store_id"], row["item_id"])
            p = float(row["sell_price"])
            sums[key] = sums.get(key, 0.0) + p
            counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def load_prices_series(data_dir: Path = DATA_DIR):
    """Return dict (store_id, item_id) -> (wm_yr_wk_sorted: np.ndarray,
    sell_price_sorted: np.ndarray), i.e. each series' full weekly price
    history, for callers that need a price known only as-of a given day
    (see `mean_price_as_of`) rather than `load_prices_index`'s single
    whole-history mean.

    Added 2026-09-23: `load_prices_index`'s whole-history mean was found
    (independent code review) to leak future price information into
    ml_baseline.py's per-epoch quantile model, which uses price as a
    predictive feature, not just a cost coefficient (a per-series
    constant used identically by every policy's cost model isn't a
    leak; a future-dependent value handed to only one policy's
    predictive model is).
    """
    path = data_dir / PRICES_FILE
    weeks, prices = {}, {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["store_id"], row["item_id"])
            weeks.setdefault(key, []).append(int(row["wm_yr_wk"]))
            prices.setdefault(key, []).append(float(row["sell_price"]))
    out = {}
    for key in weeks:
        order = np.argsort(weeks[key])
        out[key] = (np.array(weeks[key])[order], np.array(prices[key])[order])
    return out


def mean_price_as_of(prices_series: dict, key, wm_yr_wk_cutoff: int):
    """Mean sell_price using only weeks with wm_yr_wk <= wm_yr_wk_cutoff,
    i.e. price information actually available as of that calendar week.
    Falls back to the full series mean if no week is that old yet (should
    only happen for decision points earlier than this item's first
    recorded price week)."""
    entry = prices_series.get(key)
    if entry is None:
        return None
    wks, prices = entry
    n = int(np.searchsorted(wks, wm_yr_wk_cutoff, side="right"))
    if n == 0:
        return float(prices.mean())
    return float(prices[:n].mean())


def stratified_sample(
    meta,
    n_per_stratum: int,
    zero_frac_by_id: dict,
    seed: int = 0,
):
    """Stratify by (cat_id, state_id, intermittency tertile) and sample up
    to `n_per_stratum` series ids per cell, reproducibly.

    `zero_frac_by_id`: dict id -> fraction of zero-demand days, used to cut
    each (cat_id, state_id) group into low/medium/high-intermittency
    tertiles (quantile cuts within that group, not global -- categories
    have very different baseline intermittency, e.g. FOODS vs HOBBIES, so a
    global cut would just reproduce the category split).
    """
    rng = np.random.default_rng(seed)
    by_group = {}
    for m in meta:
        key = (m["cat_id"], m["state_id"])
        by_group.setdefault(key, []).append(m["id"])

    selected = []
    strata_report = []
    for (cat, state), ids in sorted(by_group.items()):
        zfracs = np.array([zero_frac_by_id[i] for i in ids])
        order = np.argsort(zfracs)
        ids_sorted = [ids[i] for i in order]
        n = len(ids_sorted)
        cuts = [0, n // 3, 2 * n // 3, n]
        labels = ["low", "medium", "high"]
        for lab, lo, hi in zip(labels, cuts[:-1], cuts[1:]):
            cell_ids = ids_sorted[lo:hi]
            k = min(n_per_stratum, len(cell_ids))
            picked = list(rng.choice(cell_ids, size=k, replace=False))
            selected.extend(picked)
            strata_report.append(
                {"cat_id": cat, "state_id": state, "intermittency": lab, "n_available": len(cell_ids), "n_picked": k}
            )
    return selected, strata_report
