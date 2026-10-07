"""One row per second: the planet's finest weather, rebuilt from every trade.

Rows use the same 24 columns as the minute store (``data.COLUMNS``), so
everything downstream reads either resolution.

* Fast columns, per second, from Binance USD-M perpetual aggTrades (every
  trade): open, high, low, close, volume, quote_volume, trades,
  taker_buy_volume. A second without trades repeats the last price with zero
  volume. Nothing is lost.
* Slow columns, from the minute store, each taken from the latest minute that
  had already *closed* (so a second never sees its own minute's close): mark
  price, premium, spot, open interest and positioning. Funding is placed in
  the second that ends on the settlement, exactly as in the minute store.
* mark_high / mark_low at 1 second are the trade high and low. No 1-second
  mark price exists, and the trade range is the stricter test for liquidation.

Stored as one compressed numpy file per day: <store>/<SYMBOL>/<YYYY-MM-DD>.npz.
"""

import datetime as dt
import glob
import io
import os
import zipfile

import numpy as np

from . import data
from .data import ARCHIVE_URL, C, COLUMNS, MINUTE

SECOND = 1000
DAY_MS = 86_400_000
_SLOW = ("premium", "spot_close", "spot_volume", "spot_taker_buy_volume", "open_interest",
         "open_interest_value", "top_account_ls", "top_position_ls", "account_ls", "taker_ls",
         "mark_close", "perp")


def parse_aggtrades(blob):
    """(time_ms, price, qty, buyer_is_maker) arrays from an aggTrades zip."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        raw = zf.read(name)
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
        if not df.iloc[0, 0].strip().isdigit():
            df = df.iloc[1:]
        t = df.iloc[:, 5].astype(np.int64).to_numpy()
        price = df.iloc[:, 1].astype(float).to_numpy()
        qty = df.iloc[:, 2].astype(float).to_numpy()
        maker = df.iloc[:, 6].str.strip().str.lower().isin(("true", "1")).to_numpy()
    except ImportError:                                    # pragma: no cover
        rows = [r.split(",") for r in raw.decode().splitlines() if r[:1].isdigit()]
        t = np.array([int(r[5]) for r in rows], np.int64)
        price = np.array([float(r[1]) for r in rows])
        qty = np.array([float(r[2]) for r in rows])
        maker = np.array([r[6].strip().lower() in ("true", "1") for r in rows])
    t = np.where(t > 10 ** 14, t // 1000, t)               # microsecond stamps
    order = np.argsort(t, kind="stable")
    return t[order], price[order], qty[order], maker[order]


def seconds_from_trades(t_ms, price, qty, buyer_maker, day_start_ms, prev_close=np.nan):
    """86,400 one-second bars for the day starting at day_start_ms."""
    rows = np.full((86_400, len(COLUMNS)), np.nan)
    rows[:, 0] = day_start_ms + np.arange(86_400) * SECOND
    keep = (t_ms >= day_start_ms) & (t_ms < day_start_ms + DAY_MS)
    t_ms, price, qty, buyer_maker = t_ms[keep], price[keep], qty[keep], buyer_maker[keep]
    sec = (t_ms - day_start_ms) // SECOND
    vol = np.zeros(86_400)
    quote = np.zeros(86_400)
    count = np.zeros(86_400)
    taker = np.zeros(86_400)
    np.add.at(vol, sec, qty)
    np.add.at(quote, sec, qty * price)
    np.add.at(count, sec, 1.0)
    np.add.at(taker, sec, np.where(buyer_maker, 0.0, qty))
    opn = np.full(86_400, np.nan)
    close = np.full(86_400, np.nan)
    high = np.full(86_400, -np.inf)
    low = np.full(86_400, np.inf)
    if len(sec):
        first = np.r_[True, sec[1:] != sec[:-1]]
        last = np.r_[sec[1:] != sec[:-1], True]
        opn[sec[first]] = price[first]
        close[sec[last]] = price[last]
        np.maximum.at(high, sec, price)
        np.minimum.at(low, sec, price)
    # seconds without trades: the price stands still
    filled = close.copy()
    if np.isnan(filled[0]):
        filled[0] = prev_close
    idx = np.where(np.isnan(filled), 0, np.arange(86_400))
    np.maximum.accumulate(idx, out=idx)
    filled = filled[idx]
    empty = np.isnan(close)
    opn = np.where(empty, filled, opn)
    high = np.where(empty, filled, high)
    low = np.where(empty, filled, low)
    close = filled
    rows[:, 1:9] = np.stack([opn, high, low, close, vol, quote, count, taker], 1)
    rows[:, C["mark_high"]], rows[:, C["mark_low"]] = high, low
    return rows


def join_slow(rows, minute_rows):
    """Fill slow columns of second rows from minute rows, causally."""
    if len(minute_rows) == 0:
        return rows
    mt = minute_rows[:, 0].astype(np.int64)
    t = rows[:, 0].astype(np.int64)
    # the latest minute that has closed by the start of this second
    k = np.searchsorted(mt, (t // MINUTE) * MINUTE - MINUTE, side="right") - 1
    ok = k >= 0
    for name in _SLOW:
        col = np.full(len(rows), np.nan)
        col[ok] = minute_rows[k[ok], C[name]]
        rows[:, C[name]] = col
    rows[:, C["funding_rate"]] = 0.0
    settle = minute_rows[minute_rows[:, C["funding_rate"]] != 0]
    for m in settle:
        last_second = int(m[0]) + MINUTE - SECOND           # ends on the settlement
        j = (last_second - int(t[0])) // SECOND
        if 0 <= j < len(rows):
            rows[j, C["funding_rate"]] = m[C["funding_rate"]]
    return rows


def day_path(store, symbol, day):
    return os.path.join(store, symbol, f"{day:%Y-%m-%d}.npz")


def build_seconds(store="data/seconds", minute_store="data/market", symbol="BTCUSDT",
                  start="2019-09-08", end=None, cache_dir="data/raw", log=print):
    """Build (or extend) the per-day 1-second store from perpetual aggTrades."""
    day = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end) if end else dt.datetime.now(dt.timezone.utc).date()
    prev_close, total = np.nan, 0
    while day < last:
        path = day_path(store, symbol, day)
        if os.path.exists(path):
            with np.load(path) as z:
                prev_close = z["bars"][-1, C["close"]]
            day += dt.timedelta(1)
            continue
        url = (f"{ARCHIVE_URL}/futures/um/daily/aggTrades/{symbol}/"
               f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip")
        try:
            blob = data._get(url, cache_dir)
        except OSError as err:
            log(f"\n  stopped at {day}: {err}")
            break
        if blob is not None:
            start_ms = int(dt.datetime(day.year, day.month, day.day,
                                       tzinfo=dt.timezone.utc).timestamp() * 1000)
            minutes = np.array(list(data.iter_rows(minute_store, symbol, start_ms - 2 * MINUTE,
                                                   start_ms + DAY_MS))).reshape(-1, len(COLUMNS))
            if prev_close != prev_close:                   # first day: the last price before it
                before = minutes[minutes[:, 0] < start_ms]
                if len(before):
                    prev_close = before[-1, C["close"]]
            rows = seconds_from_trades(*parse_aggtrades(blob), start_ms, prev_close)
            rows = join_slow(rows, minutes)
            save_day(path, rows)
            prev_close = rows[-1, C["close"]]
            total += len(rows)
            log(f"\r  {day}: 86400 seconds", end="", flush=True)
        day += dt.timedelta(1)
    log("")
    return total


def save_day(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, bars=rows, columns=np.array(COLUMNS))
    os.replace(tmp, path)


def iter_seconds(store="data/seconds", symbol="BTCUSDT", start_ms=None, end_ms=None):
    for path in sorted(glob.glob(os.path.join(store, symbol, "????-??-??.npz"))):
        with np.load(path) as z:
            rows = z["bars"]
        t = rows[:, 0]
        keep = np.ones(len(rows), bool)
        if start_ms is not None:
            keep &= t >= start_ms
        if end_ms is not None:
            keep &= t < end_ms
        yield from rows[keep]


def save_seconds(store, symbol, rows):
    """Write rows of any span into the per-day layout."""
    days = rows[:, 0].astype(np.int64) // DAY_MS
    for d in np.unique(days):
        day = dt.date(1970, 1, 1) + dt.timedelta(int(d))
        save_day(day_path(store, symbol, day), rows[days == d])
