"""Market data: free Binance BTC klines (bulk archive, REST, live), CSV, synthetic.

A bar is a tuple ``(open_time_ms, open, high, low, close, volume)``. Only public
endpoints are used, so no API key or account is needed:

* ``data.binance.vision``: daily/monthly zip archives of every spot kline,
  1s and 1m included, back to 2017. Best for deep history.
* ``data-api.binance.vision`` (REST): recent bars and the live feed.
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

DEFAULT_BASE_URLS = (
    "https://data-api.binance.vision",   # market-data mirror, fewest geo blocks
    "https://api.binance.com",
    "https://api.binance.us",
)
ARCHIVE_URL = "https://data.binance.vision/data/spot"
INTERVALS = {1: "1s", 60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h"}


def interval_name(seconds):
    if seconds not in INTERVALS:
        raise ValueError(f"unsupported bar size {seconds}s; choose from {sorted(INTERVALS)}")
    return INTERVALS[seconds]


# --------------------------------------------------------------------- Binance
class BinanceClient:
    def __init__(self, base_urls=DEFAULT_BASE_URLS, timeout=15):
        self.base_urls = list(base_urls)
        self.timeout = timeout

    def klines(self, symbol, seconds, start_ms, limit=1000):
        query = urllib.parse.urlencode({
            "symbol": symbol, "interval": interval_name(seconds),
            "startTime": int(start_ms), "limit": limit,
        })
        last_err = None
        for i, base in enumerate(self.base_urls):
            try:
                with urllib.request.urlopen(f"{base}/api/v3/klines?{query}",
                                            timeout=self.timeout) as resp:
                    rows = json.load(resp)
                if i:   # remember the mirror that worked
                    self.base_urls.insert(0, self.base_urls.pop(i))
                return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5]), int(r[6])) for r in rows]
            except Exception as err:   # try the next mirror
                last_err = err
        raise ConnectionError(f"all Binance endpoints failed: {last_err}")


def parse_archive(blob):
    """Bars from one data.binance.vision kline zip (no header, ms or us stamps)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        text = zf.read(name).decode()
    bars = []
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].isdigit():
            continue                                   # tolerate a header line
        t = int(row[0])
        if t > 10 ** 14:                               # 2025+ files use microseconds
            t //= 1000
        bars.append((t, float(row[1]), float(row[2]), float(row[3]),
                     float(row[4]), float(row[5])))
    return bars


def _get(url, timeout=60):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None                                # not published (yet)
        raise


def archive_bars(symbol, seconds, start_ms, end_ms, log=print):
    """Yield archived bars in [start_ms, end_ms): monthly zips, daily for the rest."""
    iv = interval_name(seconds)
    day = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).date()
    last_day = dt.datetime.fromtimestamp(end_ms / 1000, dt.timezone.utc).date()
    while day < last_day:
        month_end = day.replace(day=calendar.monthrange(day.year, day.month)[1])
        blob = None
        if day.day == 1 and month_end < last_day:
            tag = f"{day:%Y-%m}"
            blob = _get(f"{ARCHIVE_URL}/monthly/klines/{symbol}/{iv}/{symbol}-{iv}-{tag}.zip")
            span_end = month_end
        if blob is None:
            tag = f"{day:%Y-%m-%d}"
            blob = _get(f"{ARCHIVE_URL}/daily/klines/{symbol}/{iv}/{symbol}-{iv}-{tag}.zip")
            span_end = day
        if blob is not None:
            log(f"\r  archive {tag}", end="", flush=True)
            for b in parse_archive(blob):
                if start_ms <= b[0] < end_ms:
                    yield b
        day = span_end + dt.timedelta(days=1)


def api_bars(client, symbol, seconds, start_ms, end_ms):
    """Yield closed bars in [start_ms, end_ms) from the REST API."""
    step = seconds * 1000
    nxt = start_ms
    while nxt < end_ms:
        rows = [r for r in client.klines(symbol, seconds, nxt)
                if r[0] < end_ms and r[6] < time.time() * 1000]
        if not rows:
            return
        yield from (r[:6] for r in rows)
        nxt = rows[-1][0] + step


def download(path, symbol="BTCUSDT", seconds=60, days=30, source="auto",
             client=None, log=print):
    """Create or extend a CSV of closed klines, resuming after its last row.

    ``source``: "archive" (bulk zips up to yesterday), "api" (REST), or "auto"
    (archive for whole past days, REST for today).
    """
    client = client or BinanceClient()
    step = seconds * 1000
    now = int(time.time() * 1000)
    start = now - int(days * 86400 * 1000)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        last = None
        for last in iter_csv(path):
            pass
        if last is not None:
            start = last[0] + step
    today = now // 86_400_000 * 86_400_000
    streams = []
    if source in ("auto", "archive"):
        streams.append(archive_bars(symbol, seconds, start, today, log))
    if source == "api":
        streams.append(api_bars(client, symbol, seconds, start, now))
    elif source == "auto":
        streams.append(api_bars(client, symbol, seconds, max(start, today), now))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    n, newest = 0, start - step
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for stream in streams:
            try:
                for b in stream:
                    if b[0] <= newest:
                        continue                         # keep strictly increasing
                    w.writerow(b)
                    newest = b[0]
                    n += 1
            except (OSError, ConnectionError) as err:
                log(f"\n  stopped early: {err}")
    log("")
    return n


def live_bars(symbol="BTCUSDT", seconds=1, poll_s=2.0, start_ms=None, client=None,
              log=lambda m: print(m, file=sys.stderr)):
    """Yield closed klines forever, catching up after outages without gaps."""
    client = client or BinanceClient()
    step = seconds * 1000
    nxt = start_ms if start_ms is not None else (int(time.time() * 1000) // step) * step
    backoff = 1.0
    while True:
        try:
            rows = client.klines(symbol, seconds, nxt)
            backoff = 1.0
        except Exception as err:
            log(f"feed error: {err}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        now = int(time.time() * 1000)
        closed = [r for r in rows if r[6] < now]
        for r in closed:
            yield r[:6]
        if closed:
            nxt = closed[-1][0] + step
        if len(closed) < 1000:
            time.sleep(poll_s)


# ------------------------------------------------------------------------- CSV
def iter_csv(path):
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            yield (int(float(row[0])), float(row[1]), float(row[2]),
                   float(row[3]), float(row[4]), float(row[5]))


def write_csv(path, bars):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow(b)


def detect_bar_seconds(path):
    it = iter_csv(path)
    a, b = next(it), next(it)
    return (b[0] - a[0]) // 1000


# ------------------------------------------------------------------- synthetic
def synthetic_bars(n, seconds=60, seed=0, start_price=60000.0,
                   start_ms=1_700_000_000_000):
    """BTC-like bars: regime-switching drift, clustered volatility, rare jumps.

    Useful for testing the machinery offline. Whatever agents learn here says
    nothing about the real market.
    """
    rng = np.random.default_rng(seed)
    dt = seconds / (365 * 86400)
    regimes = [(-0.8, 0.9), (0.0, 0.5), (0.9, 0.6)]   # (annual drift, annual vol)
    stay = 1 - seconds / (3 * 86400)                   # regimes last ~3 days
    state, logvol = 1, 0.0
    price = start_price
    step = seconds * 1000
    t0 = (start_ms // step) * step
    for k in range(n):
        if rng.random() > stay:
            state = rng.integers(len(regimes))
        mu, sig = regimes[state]
        logvol = 0.995 * logvol + 0.07 * rng.normal()
        s = sig * np.exp(logvol) * np.sqrt(dt)
        r = (mu - 0.5 * sig ** 2) * dt + s * rng.standard_t(4) / np.sqrt(2)
        if rng.random() < 2e-4:
            r += rng.normal(0, 0.02)
        o, c = price, price * np.exp(r)
        h = max(o, c) * np.exp(abs(rng.normal(0, 0.4 * s)))
        lo = min(o, c) * np.exp(-abs(rng.normal(0, 0.4 * s)))
        v = 5.0 * np.exp(rng.normal(0, 0.5)) * (1 + 200 * abs(r))
        price = c
        yield (t0 + k * step, round(o, 2), round(h, 2), round(lo, 2), round(c, 2), round(v, 4))
