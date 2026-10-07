"""Market data for a BTCUSDT perpetual: free Binance public data, CSV, synthetic.

A bar is ``(open_time_ms, open, high, low, close, volume, mark_high, mark_low,
mark_close, funding_rate)``: one minute of BTCUSDT perpetual trades, the mark
price range that decides liquidations, and the funding rate settled at the
end of that minute (0 when there is no settlement). No API key is needed:

* ``data.binance.vision``: zip archives of every futures kline, mark-price
  kline and funding settlement (the BTCUSDT perpetual exists since September
  2019), and spot klines back to August 2017;
* ``fapi.binance.com``: the same over REST, for the current month and live.

Before the perpetual existed the builder uses spot prices, treats them as the
mark price, and charges funding at 0.01% per 8 hours (what the perpetual
settles at when it trades in line with spot). Those rows say ``spot`` in the
CSV's last column; real perpetual rows say ``perp``.
"""

import calendar
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np

ARCHIVE_URL = "https://data.binance.vision/data"
SPOT_URLS = ("https://data-api.binance.vision", "https://api.binance.com")
FUTURES_URLS = ("https://fapi.binance.com",)
MINUTE = 60_000
EIGHT_HOURS = 8 * 3600 * 1000
COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
           "mark_high", "mark_low", "mark_close", "funding_rate", "source"]


class MissingData(Exception):
    """Data the builder needs is not published (yet) or not reachable."""


# ------------------------------------------------------------------ fetching
def _get(url, cache_dir=None, timeout=60):
    """Download `url` (cached on disk if `cache_dir`). None if it does not exist."""
    path = None
    if cache_dir and "/data/" in url:
        path = os.path.join(cache_dir, url.split("/data/", 1)[1])
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(blob)
    return blob


def _rows(blob):
    """CSV rows of a data.binance.vision zip, header lines skipped."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        text = zf.read(name).decode()
    return [r for r in csv.reader(io.StringIO(text)) if r and r[0].strip().isdigit()]


def _ms(stamp):
    t = int(stamp)
    return t // 1000 if t > 10 ** 14 else t       # 2025+ spot files use microseconds


def parse_klines(blob):
    return [(_ms(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in _rows(blob)]


def parse_funding(blob):
    """{settlement time (ms, to the minute): rate}."""
    out = {}
    for r in _rows(blob):
        t = round(_ms(r[0]) / MINUTE) * MINUTE
        out[t] = float(r[-1])
    return out


class BinanceClient:
    """Public REST endpoints (spot and USDⓈ-M futures), mirrors tried in order."""

    def __init__(self, spot_urls=SPOT_URLS, futures_urls=FUTURES_URLS, timeout=15):
        self.urls = {"spot": list(spot_urls), "futures": list(futures_urls)}
        self.timeout = timeout

    def _get(self, kind, path, **params):
        query = urllib.parse.urlencode(params)
        last_err = None
        bases = self.urls[kind]
        for i, base in enumerate(bases):
            try:
                with urllib.request.urlopen(f"{base}{path}?{query}", timeout=self.timeout) as r:
                    data = json.load(r)
                if i:
                    bases.insert(0, bases.pop(i))
                return data
            except Exception as err:      # try the next mirror
                last_err = err
        raise ConnectionError(f"Binance {kind} endpoints failed: {last_err}")

    def klines(self, start_ms, symbol="BTCUSDT", mark=False, limit=1500):
        path = "/fapi/v1/markPriceKlines" if mark else "/fapi/v1/klines"
        rows = self._get("futures", path, symbol=symbol, interval="1m",
                         startTime=int(start_ms), limit=limit)
        return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                 float(r[5]), int(r[6])) for r in rows]

    def funding(self, start_ms, end_ms, symbol="BTCUSDT"):
        rows = self._get("futures", "/fapi/v1/fundingRate", symbol=symbol,
                         startTime=int(start_ms), endTime=int(end_ms), limit=1000)
        return {round(int(r["fundingTime"]) / MINUTE) * MINUTE: float(r["fundingRate"])
                for r in rows}


# ------------------------------------------------------------- the data set
def _funding_rows(trade, mark, funding, source):
    """Join trade klines, mark klines {t: (h, l, c)} and funding {t: rate}."""
    out = []
    for t, o, h, l, c, v in trade:
        mh, ml, mc = mark.get(t, (h, l, c))
        out.append((t, o, h, l, c, v, mh, ml, mc, funding.get(t + MINUTE, 0.0), source))
    return out


def _spot_rows(bars, assumed_funding):
    funding = {t + MINUTE: assumed_funding for t, *_ in bars
               if (t + MINUTE) % EIGHT_HOURS == 0}
    return _funding_rows(bars, {}, funding, "spot")


def _periods(first_day, end_day):
    """(kind, tag, days) chunks: whole past months as monthly files, else daily."""
    day = first_day
    while day < end_day:
        last = day.replace(day=calendar.monthrange(day.year, day.month)[1])
        if day.day == 1 and last < end_day:
            yield "monthly", f"{day:%Y-%m}", [day + dt.timedelta(d) for d in range(last.day)]
            day = last + dt.timedelta(1)
        else:
            yield "daily", f"{day:%Y-%m-%d}", [day]
            day += dt.timedelta(1)


def _chunk(symbol, kind, tag, days, assumed_funding, cache_dir, client):
    fut = f"{ARCHIVE_URL}/futures/um/{kind}"
    spot = f"{ARCHIVE_URL}/spot/{kind}"
    trade_blob = _get(f"{fut}/klines/{symbol}/1m/{symbol}-1m-{tag}.zip", cache_dir)
    just_ended = (dt.datetime.now(dt.timezone.utc).date() - days[-1]).days < 10
    if trade_blob is None and kind == "monthly" and just_ended:
        return None                  # monthly file not published yet: use daily files
    chunk_start = int(dt.datetime(days[0].year, days[0].month, days[0].day,
                                  tzinfo=dt.timezone.utc).timestamp() * 1000)
    trade = parse_klines(trade_blob) if trade_blob else []
    rows = []
    first_perp = trade[0][0] if trade else None
    if first_perp is None or first_perp > chunk_start:  # before the perpetual existed
        spot_blob = _get(f"{spot}/klines/{symbol}/1m/{symbol}-1m-{tag}.zip", cache_dir)
        if spot_blob is not None:
            bars = [b for b in parse_klines(spot_blob) if first_perp is None or b[0] < first_perp]
            rows += _spot_rows(bars, assumed_funding)
    if trade:
        mark_blob = _get(f"{fut}/markPriceKlines/{symbol}/1m/{symbol}-1m-{tag}.zip", cache_dir)
        mark = {b[0]: b[2:5] for b in parse_klines(mark_blob)} if mark_blob else {}
        month = tag[:7]
        fund_blob = _get(f"{ARCHIVE_URL}/futures/um/monthly/fundingRate/{symbol}/"
                         f"{symbol}-fundingRate-{month}.zip", cache_dir)
        if fund_blob is not None:
            funding = parse_funding(fund_blob)
        else:
            try:
                funding = client.funding(trade[0][0], trade[-1][0] + 2 * MINUTE, symbol)
            except ConnectionError as err:
                raise MissingData(f"funding rates for {tag} unavailable ({err})") from err
        rows += _funding_rows(trade, mark, funding, "perp")
    return rows


def build_market(path, symbol="BTCUSDT", start="2017-08-01", assumed_funding=0.0001,
                 cache_dir="data/raw", client=None, log=print):
    """Create or extend the 1-minute BTCUSDT perpetual data set up to yesterday.

    Resumes after the last row already in `path`. Raw archives are cached in
    `cache_dir`, so rebuilding never downloads twice.
    """
    client = client or BinanceClient()
    start_ms = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        last = None
        for last in iter_csv(path):
            pass
        if last is not None:
            start_ms = last[0] + MINUTE
    first_day = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).date()
    end_day = dt.datetime.now(dt.timezone.utc).date()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    n, newest = 0, start_ms - MINUTE
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        try:
            for kind, tag, days in _periods(first_day, end_day):
                rows = _chunk(symbol, kind, tag, days, assumed_funding, cache_dir, client)
                if rows is None:                        # month not archived yet: go daily
                    rows = []
                    for day in days:
                        rows += _chunk(symbol, "daily", f"{day:%Y-%m-%d}", [day],
                                       assumed_funding, cache_dir, client) or []
                for r in rows:
                    if r[0] > newest:
                        w.writerow(r)
                        newest = r[0]
                        n += 1
                log(f"\r  {tag}: {n} bars written", end="", flush=True)
        except (MissingData, OSError) as err:
            log(f"\n  stopped: {err}")
    log("")
    return n


# ---------------------------------------------------------------------- live
def live_bars(start_ms, symbol="BTCUSDT", client=None, poll_s=5.0,
              log=lambda m: print(m, file=sys.stderr)):
    """Yield closed 1-minute perpetual bars forever, from `start_ms`, without gaps.

    A bar that ends on a funding settlement is held back until Binance has
    published that settlement's rate.
    """
    client = client or BinanceClient()
    nxt = start_ms // MINUTE * MINUTE
    backoff = 1.0
    while True:
        try:
            now = time.time() * 1000
            trade = [r for r in client.klines(nxt, symbol) if r[6] < now]
            mark = {r[0]: r[2:5] for r in client.klines(nxt, symbol, mark=True)} if trade else {}
            funding = client.funding(nxt, now, symbol) if trade else {}
            backoff = 1.0
        except ConnectionError as err:
            log(f"feed error: {err}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        for t, o, h, l, c, v, _ in trade:
            end = t + MINUTE
            if end % EIGHT_HOURS == 0 and end not in funding:
                log("waiting for Binance to publish the funding rate...")
                break
            mh, ml, mc = mark.get(t, (h, l, c))
            yield (t, o, h, l, c, v, mh, ml, mc, funding.get(end, 0.0))
            nxt = end
        else:
            if len(trade) < 1000:
                time.sleep(poll_s)
            continue
        time.sleep(poll_s)


def recent_bars(minutes, symbol="BTCUSDT", client=None):
    """The last `minutes` closed 1-minute trade bars (for warming up candle books)."""
    client = client or BinanceClient()
    now = int(time.time() * 1000)
    nxt, out = now // MINUTE * MINUTE - minutes * MINUTE, []
    while True:
        rows = [r for r in client.klines(nxt, symbol) if r[6] < now]
        if not rows:
            return out
        out += [r[:6] for r in rows]
        nxt = rows[-1][0] + MINUTE


# ----------------------------------------------------------------------- CSV
def iter_csv(path, start_ms=None, end_ms=None):
    """Bars from a CSV; files without mark/funding columns get trade price / 0."""
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            t = int(float(row[0]))
            if start_ms is not None and t < start_ms:
                continue
            if end_ms is not None and t >= end_ms:
                break
            o, h, l, c, v = (float(x) for x in row[1:6])
            if len(row) >= 10:
                mh, ml, mc, f = (float(x) for x in row[6:10])
            else:
                mh, ml, mc, f = h, l, c, 0.0
            yield (t, o, h, l, c, v, mh, ml, mc, f)


def write_csv(path, bars):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for b in bars:
            w.writerow(b)


def detect_bar_seconds(path):
    it = iter_csv(path)
    a, b = next(it), next(it)
    return (b[0] - a[0]) // 1000


# ----------------------------------------------------------------- synthetic
def synthetic_bars(n, seed=0, start_price=60000.0, start_ms=1_700_000_000_000,
                   funding=0.0001):
    """BTC-like 1-minute bars: regime-switching drift, clustered volatility, jumps.

    For testing the machinery offline only; whatever agents learn here says
    nothing about the real market.
    """
    rng = np.random.default_rng(seed)
    dt_y = 60 / (365 * 86400)
    regimes = [(-0.8, 0.9), (0.0, 0.5), (0.9, 0.6)]   # (annual drift, annual vol)
    stay = 1 - 60 / (3 * 86400)                        # regimes last ~3 days
    state, logvol, price = 1, 0.0, start_price
    t0 = start_ms // MINUTE * MINUTE
    for k in range(n):
        if rng.random() > stay:
            state = rng.integers(len(regimes))
        mu, sig = regimes[state]
        logvol = 0.995 * logvol + 0.07 * rng.normal()
        s = sig * np.exp(logvol) * np.sqrt(dt_y)
        r = (mu - 0.5 * sig ** 2) * dt_y + s * rng.standard_t(4) / np.sqrt(2)
        if rng.random() < 2e-4:
            r += rng.normal(0, 0.02)
        o, c = price, price * np.exp(r)
        h = max(o, c) * np.exp(abs(rng.normal(0, 0.4 * s)))
        lo = min(o, c) * np.exp(-abs(rng.normal(0, 0.4 * s)))
        v = 5.0 * np.exp(rng.normal(0, 0.5)) * (1 + 200 * abs(r))
        price = c
        t = t0 + k * MINUTE
        f = funding if (t + MINUTE) % EIGHT_HOURS == 0 else 0.0
        o, h, lo, c = round(o, 1), round(h, 1), round(lo, 1), round(c, 1)
        yield (t, o, h, lo, c, round(v, 3), h, lo, c, f)
