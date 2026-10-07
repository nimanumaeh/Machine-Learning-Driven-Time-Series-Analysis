"""The arena's raw material: one row per minute of the BTCUSDT perpetual market.

Columns come in two kinds.

Physics, which decides what happens to an agent's money and is never invented:
  open, high, low, close        perpetual trades (fills at the next open)
  mark_high, mark_low, mark_close   the mark price, which triggers liquidations
  funding_rate                  rate settled at the end of this minute, else 0

Stimuli, which agents can perceive (NaN when nothing was recorded yet):
  volume, quote_volume, trades, taker_buy_volume   perpetual order flow
  premium                       perpetual premium index (what funding follows)
  spot_close, spot_volume, spot_taker_buy_volume   the spot market next door
  open_interest, open_interest_value               leverage in the system
  top_account_ls, top_position_ls, account_ls, taker_ls
                                long/short positioning of the crowd (5-minute,
                                visible METRICS_LAG after their timestamp)
  perp                          1 = real perpetual, 0 = spot stand-in

Sources (free, no API key): zip archives on data.binance.vision, and the
public REST API for the current month's funding and for live rows. Before the
perpetual listed (September 2019) the trade and mark columns come from spot,
funding is the 0.01%/8h baseline and ``perp`` is 0. Rows are stored as one
compressed numpy file per month: ``<store>/<SYMBOL>/<YYYY-MM>.npz``.
"""

import calendar
import csv
import datetime as dt
import glob
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

COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_volume", "mark_high", "mark_low", "mark_close", "funding_rate", "premium",
    "spot_close", "spot_volume", "spot_taker_buy_volume", "open_interest",
    "open_interest_value", "top_account_ls", "top_position_ls", "account_ls", "taker_ls",
    "perp",
)
C = {name: i for i, name in enumerate(COLUMNS)}
METRICS = ("open_interest", "open_interest_value", "top_account_ls", "top_position_ls",
           "account_ls", "taker_ls")

ARCHIVE_URL = "https://data.binance.vision/data"
SPOT_URLS = ("https://data-api.binance.vision", "https://api.binance.com")
FUTURES_URLS = ("https://fapi.binance.com",)
MINUTE = 60_000
EIGHT_HOURS = 8 * 3600 * 1000
METRICS_LAG = 6 * MINUTE          # positioning stats count as known 6 minutes later
# Earliest dates worth asking the archive for (nothing older exists there).
SPOT_FROM = dt.date(2017, 8, 1)
PERP_FROM = dt.date(2019, 9, 1)
PREMIUM_FROM = dt.date(2020, 1, 1)
METRICS_FROM = dt.date(2020, 9, 1)


class MissingData(Exception):
    """Physics the arena needs is not published (yet) or not reachable."""


# ------------------------------------------------------------------- parsing
def _get(url, cache_dir=None, timeout=60):
    """Download `url`, cached on disk under `cache_dir`. None if it does not exist."""
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


def _csv(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        return list(csv.reader(io.StringIO(zf.read(name).decode())))


def _ms(stamp):
    """Milliseconds from a ms/us epoch number or a 'YYYY-MM-DD HH:MM:SS' string."""
    s = str(stamp).strip()
    if s.isdigit():
        t = int(s)
        return t // 1000 if t > 10 ** 14 else t
    when = dt.datetime.fromisoformat(s.replace("T", " ").replace("Z", ""))
    return int(when.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def parse_klines(blob):
    """{open_time: (o, h, l, c, volume, quote_volume, trades, taker_buy_volume)}."""
    out = {}
    for r in _csv(blob):
        if r and r[0].strip().isdigit():
            out[_ms(r[0])] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                              float(r[5]), float(r[7]), float(r[8]), float(r[9]))
    return out


def parse_funding(blob):
    """{settlement time to the minute: rate}."""
    out = {}
    for r in _csv(blob):
        if r and r[0].strip().isdigit():
            out[round(_ms(r[0]) / MINUTE) * MINUTE] = float(r[-1])
    return out


def parse_metrics(blob):
    """Sorted [(time, open_interest, value, top_account, top_position, account, taker)]."""
    rows = _csv(blob)
    header = [h.strip() for h in rows[0]] if rows and not rows[0][0][:1].isdigit() else None
    names = ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
             "sum_toptrader_long_short_ratio", "count_long_short_ratio",
             "sum_taker_long_short_vol_ratio")
    cols = [header.index(n) for n in names] if header else [2, 3, 4, 5, 6, 7]
    out = []
    for r in rows[1 if header else 0:]:
        if not r:
            continue
        vals = []
        for k in cols:
            try:
                vals.append(float(r[k]))
            except (ValueError, IndexError):
                vals.append(np.nan)
        out.append((_ms(r[0]), *vals))
    return sorted(out)


# ---------------------------------------------------------------- assembling
def assemble(minutes, perp, mark, premium, spot, funding, metrics, carry=None,
             assumed_funding=0.0001):
    """Rows for `minutes` (sorted open times). Archive and live both use this.

    perp, spot: {t: kline}; mark: {t: kline}; premium: {t: kline};
    funding: {settlement time: rate}; metrics: sorted tuples from parse_metrics;
    carry: the metrics tuple in force before the first minute. A minute is a
    perpetual row if the perpetual traded then, otherwise a spot stand-in row.
    Returns (rows, carry).
    """
    rows = np.full((len(minutes), len(COLUMNS)), np.nan)
    j = 0
    while j < len(metrics) and metrics[j][0] + METRICS_LAG <= (minutes[0] if minutes else 0):
        carry = metrics[j]
        j += 1
    for n, t in enumerate(minutes):
        row = rows[n]
        row[C["open_time"]] = t
        while j < len(metrics) and metrics[j][0] + METRICS_LAG <= t:
            carry = metrics[j]
            j += 1
        if t in perp:
            o, h, l, c, v, qv, nt, tb = perp[t]
            mh, ml, mc = mark[t][1:4] if t in mark else (h, l, c)
            row[C["funding_rate"]] = funding.get(t + MINUTE, 0.0)
            row[C["premium"]] = premium[t][3] if t in premium else np.nan
            if t in spot:
                s = spot[t]
                row[C["spot_close"]], row[C["spot_volume"]], row[C["spot_taker_buy_volume"]] = \
                    s[3], s[4], s[7]
            if carry is not None:
                for name, value in zip(METRICS, carry[1:]):
                    row[C[name]] = value
            row[C["perp"]] = 1.0
        else:
            o, h, l, c, v, qv, nt, tb = spot[t]
            mh, ml, mc = h, l, c
            row[C["funding_rate"]] = assumed_funding if (t + MINUTE) % EIGHT_HOURS == 0 else 0.0
            row[C["perp"]] = 0.0
        row[1:9] = (o, h, l, c, v, qv, nt, tb)
        row[C["mark_high"]], row[C["mark_low"]], row[C["mark_close"]] = mh, ml, mc
    return rows, carry


# ------------------------------------------------------------------- archive
def _zip_url(market, period, dtype, symbol, tag, interval="1m"):
    if interval:
        return f"{ARCHIVE_URL}/{market}/{period}/{dtype}/{symbol}/{interval}/{symbol}-{interval}-{tag}.zip"
    return f"{ARCHIVE_URL}/{market}/{period}/{dtype}/{symbol}/{symbol}-{dtype}-{tag}.zip"


def _archive(market, dtype, symbol, days, month_done, cache_dir, parse, monthly=True,
             daily=True, interval="1m"):
    """Parsed data for `days` (one month): the monthly zip, else the daily ones."""
    tag = f"{days[0]:%Y-%m}"
    if monthly and month_done:
        blob = _get(_zip_url(market, "monthly", dtype, symbol, tag, interval), cache_dir)
        if blob is not None:
            return parse(blob)
        just_ended = (dt.datetime.now(dt.timezone.utc).date() - days[-1]).days < 10
        if not just_ended:
            return parse_empty(parse)              # long past and absent: never existed
    if not daily:
        return parse_empty(parse)
    out = parse_empty(parse)
    for day in days:
        blob = _get(_zip_url(market, "daily", dtype, symbol, f"{day:%Y-%m-%d}", interval), cache_dir)
        if blob is not None:
            part = parse(blob)
            out = out + part if isinstance(out, list) else {**out, **part}
    return sorted(out) if isinstance(out, list) else out


def parse_empty(parse):
    return [] if parse is parse_metrics else {}


def build_month(symbol, year, month, client, cache_dir, carry=None, perp_started=False,
                assumed_funding=0.0001, today=None):
    """Rows for one calendar month up to yesterday. Returns (rows, carry, perp_started, done)."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    last = dt.date(year, month, calendar.monthrange(year, month)[1])
    days = [d for d in (dt.date(year, month, k) for k in range(1, last.day + 1)) if d < today]
    if not days:
        return np.empty((0, len(COLUMNS))), carry, perp_started, False
    done = last < today
    month_start = days[0]

    def get(market, dtype, since, parse, **kw):
        if days[-1] < since:
            return parse_empty(parse)
        return _archive(market, dtype, symbol, days, done, cache_dir, parse, **kw)

    perp = get("futures/um", "klines", PERP_FROM, parse_klines)
    spot = get("spot", "klines", SPOT_FROM, parse_klines)
    mark = premium = funding = {}
    metrics = []
    if perp:
        mark = get("futures/um", "markPriceKlines", PERP_FROM, parse_klines)
        premium = get("futures/um", "premiumIndexKlines", PREMIUM_FROM, parse_klines)
        metrics = get("futures/um", "metrics", METRICS_FROM, parse_metrics,
                      monthly=False, interval=None)
        funding = get("futures/um", "fundingRate", PERP_FROM, parse_funding,
                      daily=False, interval=None)
        if not funding:
            try:
                funding = client.funding(min(perp), max(perp) + 2 * MINUTE, symbol)
            except ConnectionError as err:
                raise MissingData(f"funding rates for {month_start:%Y-%m} unavailable ({err})")

    if perp:        # spot stand-ins only before the perpetual's very first minute
        first = min(perp)
        standins = set() if perp_started else {t for t in spot if t < first}
        minutes = sorted(set(perp) | standins)
    elif not perp_started:
        minutes = sorted(spot)
    else:
        minutes = []    # perpetual era without perpetual data: a gap, never a stand-in
    perp_started = perp_started or bool(perp)
    minutes = [t for t in minutes if t < _ms_day(today)]
    rows, carry = assemble(minutes, perp, mark, premium, spot, funding, metrics, carry,
                           assumed_funding)
    return rows, carry, perp_started, done


def _ms_day(day):
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


# --------------------------------------------------------------------- store
def month_path(store, symbol, year, month):
    return os.path.join(store, symbol, f"{year:04d}-{month:02d}.npz")


def save_month(path, rows, done):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, bars=rows, columns=np.array(COLUMNS), complete=done)
    os.replace(tmp, path)


def month_files(store, symbol="BTCUSDT"):
    return sorted(glob.glob(os.path.join(store, symbol, "????-??.npz")))


def _carry_from(rows):
    if len(rows) == 0 or not rows[-1][C["perp"]] == 1:
        return None
    last = rows[-1]
    vals = tuple(last[C[m]] for m in METRICS)
    return None if np.isnan(vals[0]) else (int(last[C["open_time"]]) - METRICS_LAG, *vals)


def build_market(store="data/market", symbol="BTCUSDT", start="2017-08", client=None,
                 cache_dir="data/raw", assumed_funding=0.0001, log=print):
    """Build (or bring up to date) the monthly store from August 2017 to yesterday.

    Months already complete on disk are skipped; the current month is rebuilt.
    Raw archives are cached under `cache_dir`, so nothing downloads twice.
    """
    client = client or BinanceClient()
    year, month = (int(x) for x in start.split("-")[:2])
    today = dt.datetime.now(dt.timezone.utc).date()
    carry, perp_started, total = None, False, 0
    while (year, month) <= (today.year, today.month):
        path = month_path(store, symbol, year, month)
        existing = None
        if os.path.exists(path):
            with np.load(path) as z:
                if bool(z["complete"]):
                    existing = z["bars"]
        if existing is not None:
            rows = existing
        else:
            try:
                rows, carry, perp_started, done = build_month(
                    symbol, year, month, client, cache_dir, carry, perp_started,
                    assumed_funding, today)
            except (MissingData, OSError) as err:
                log(f"\n  stopped at {year:04d}-{month:02d}: {err}")
                break
            if len(rows):
                save_month(path, rows, done)
                total += len(rows)
            log(f"\r  {year:04d}-{month:02d}: {len(rows)} minutes", end="", flush=True)
        if len(rows):
            perp_started = perp_started or bool((rows[:, C["perp"]] == 1).any())
            carry = _carry_from(rows) or carry
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    log("")
    return total


def iter_rows(store="data/market", symbol="BTCUSDT", start_ms=None, end_ms=None):
    """Every stored minute in order, as numpy rows."""
    for path in month_files(store, symbol):
        with np.load(path) as z:
            rows = z["bars"]
        t = rows[:, 0]
        keep = np.ones(len(rows), bool)
        if start_ms is not None:
            keep &= t >= start_ms
        if end_ms is not None:
            keep &= t < end_ms
        yield from rows[keep]


def save_store(store, symbol, rows):
    """Write rows (any span) into the monthly layout."""
    months = rows[:, 0].astype(np.int64).astype("datetime64[ms]").astype("datetime64[M]")
    for month in np.unique(months):
        y, m = (int(x) for x in str(month).split("-"))
        save_month(month_path(store, symbol, y, m), rows[months == month], True)


def export_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(["" if np.isnan(x) else (int(x) if k == 0 else repr(float(x)))
                        for k, x in enumerate(r)])


def coverage(store="data/market", symbol="BTCUSDT"):
    """Per month: minutes, share of real perpetual rows, share of each stimulus present."""
    out = []
    for path in month_files(store, symbol):
        with np.load(path) as z:
            rows = z["bars"]
        present = {name: float(np.isfinite(rows[:, C[name]]).mean()) for name in
                   ("premium", "spot_close", "open_interest", "top_position_ls")}
        out.append((os.path.basename(path)[:7], len(rows), float(rows[:, C["perp"]].mean()), present))
    return out


# ---------------------------------------------------------------------- REST
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
            except Exception as err:          # try the next mirror
                last_err = err
        raise ConnectionError(f"Binance {kind} endpoints failed: {last_err}")

    @staticmethod
    def _klines(rows):
        return {int(r[0]): (float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
                            float(r[7]), float(r[8]), float(r[9])) for r in rows}, \
               {int(r[0]): int(r[6]) for r in rows}

    def klines(self, start_ms, symbol="BTCUSDT", source="perp"):
        """({open_time: kline}, {open_time: close_time}) from `start_ms`."""
        path = {"perp": "/fapi/v1/klines", "mark": "/fapi/v1/markPriceKlines",
                "premium": "/fapi/v1/premiumIndexKlines", "spot": "/api/v3/klines"}[source]
        kind = "spot" if source == "spot" else "futures"
        rows = self._get(kind, path, symbol=symbol, interval="1m", startTime=int(start_ms),
                         limit=1000)
        return self._klines(rows)

    def funding(self, start_ms, end_ms, symbol="BTCUSDT"):
        rows = self._get("futures", "/fapi/v1/fundingRate", symbol=symbol,
                         startTime=int(start_ms), endTime=int(end_ms), limit=1000)
        return {round(int(r["fundingTime"]) / MINUTE) * MINUTE: float(r["fundingRate"])
                for r in rows}

    def metrics(self, start_ms, symbol="BTCUSDT"):
        """5-minute positioning stats (Binance keeps the last 30 days)."""
        series = {}
        for path, fields, slot in (
                ("/futures/data/openInterestHist", ("sumOpenInterest", "sumOpenInterestValue"), 0),
                ("/futures/data/topLongShortAccountRatio", ("longShortRatio",), 2),
                ("/futures/data/topLongShortPositionRatio", ("longShortRatio",), 3),
                ("/futures/data/globalLongShortAccountRatio", ("longShortRatio",), 4),
                ("/futures/data/takerlongshortRatio", ("buySellRatio",), 5)):
            for r in self._get("futures", path, symbol=symbol, period="5m",
                               startTime=int(start_ms), limit=500):
                vals = series.setdefault(int(r["timestamp"]), [np.nan] * 6)
                for k, f in enumerate(fields):
                    vals[slot + k] = float(r[f])
        return sorted((t, *v) for t, v in series.items())


# ---------------------------------------------------------------------- live
def live_rows(start_ms, symbol="BTCUSDT", client=None, carry=None, poll_s=5.0,
              log=lambda m: print(m, file=sys.stderr)):
    """Closed minutes from `start_ms` on, forever, assembled exactly like the archive.

    A minute that ends on a funding settlement waits until Binance publishes
    that settlement's rate.
    """
    client = client or BinanceClient()
    nxt = start_ms // MINUTE * MINUTE
    backoff = 1.0
    metrics = []
    while True:
        try:
            now = time.time() * 1000
            perp, closes = client.klines(nxt, symbol, "perp")
            ready = sorted(t for t in perp if closes[t] < now)
            if ready:
                mark, _ = client.klines(nxt, symbol, "mark")
                premium, _ = client.klines(nxt, symbol, "premium")
                spot, _ = client.klines(nxt, symbol, "spot")
                funding = client.funding(nxt, now, symbol)
                fresh = client.metrics(nxt - 3600_000, symbol)
                known = {m[0] for m in metrics}
                metrics = sorted(metrics + [m for m in fresh if m[0] not in known])[-500:]
            backoff = 1.0
        except ConnectionError as err:
            log(f"feed error: {err}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        held = [t for t in ready if (t + MINUTE) % EIGHT_HOURS == 0 and t + MINUTE not in funding]
        if held:
            log("waiting for Binance to publish the funding rate...")
            ready = [t for t in ready if t < held[0]]
        if ready:
            rows, carry = assemble(ready, perp, mark, premium, spot, funding,
                                   [m for m in metrics if m[0] + METRICS_LAG <= ready[-1]], carry)
            yield from rows
            nxt = ready[-1] + MINUTE
        if held or len(ready) < 900:
            time.sleep(poll_s)
