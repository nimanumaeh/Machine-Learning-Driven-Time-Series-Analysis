"""The planet's weather as matter feels it, computed for whole chunks of rows at once.

Each latitude (band) aggregates the rows into bars of its own length; when a bar
closes, every stream's level and change is read as the percentile of its own
recent history in that band (histogram equalization, Laughlin 1981) and written
as one signed byte: 0 is the median, -128..127 below to above, unknown reads 0.
This is the same law as Planet._close_bar and Planet._transduce, compiled, so
that the soup (on the CPU or a GPU) is not held up by Python for every row.
"""

import math

import numpy as np
from numba import njit

from .data import C, EIGHT_HOURS
from .physics import E_CLOSE, E_FUNDING, E_HIGH, E_LOW, E_MARK, E_OPEN, E_T, N_ENV
from .selfmade import N_REC, RECEPTORS

_OPEN_TIME, _OPEN, _HIGH, _LOW, _CLOSE = C["open_time"], C["open"], C["high"], C["low"], C["close"]
_VOLUME, _QUOTE, _TRADES, _TAKER = C["volume"], C["quote_volume"], C["trades"], C["taker_buy_volume"]
_MARK_HIGH, _MARK_LOW, _MARK_CLOSE = C["mark_high"], C["mark_low"], C["mark_close"]
_FUNDING, _PREMIUM = C["funding_rate"], C["premium"]
# streams read as the log of a positive value, and where they come from in a row
_LOGS = np.array([[RECEPTORS.index(n), C[c]] for n, c in (
    ("mark", "mark_close"), ("spot", "spot_close"), ("open_interest", "open_interest"),
    ("open_interest_value", "open_interest_value"), ("top_account_ls", "top_account_ls"),
    ("top_position_ls", "top_position_ls"), ("account_ls", "account_ls"), ("taker_ls", "taker_ls"))],
    np.int64)
_LOG1PS = np.array([[RECEPTORS.index(n), C[c]] for n, c in (
    ("spot_volume", "spot_volume"), ("spot_taker_buy", "spot_taker_buy_volume"))], np.int64)
# the bar's own streams: close, high, low (log) and volume, quote, trades, taker (log1p)
_FAST = np.array([RECEPTORS.index(n) for n in ("close", "high", "low", "volume", "quote_volume",
                                                "trades", "taker_buy")], np.int64)
_BAR_OF_FAST = np.array([3, 1, 2, 4, 5, 6, 7], np.int64)
_PREMIUM_R, _FUNDING_R = RECEPTORS.index("premium"), RECEPTORS.index("funding")
_CLOCKS = np.array([RECEPTORS.index(n) for n in ("day_sin", "day_cos", "week_sin", "week_cos",
                                                  "cycle_sin", "cycle_cos")], np.int64)
NS = 2 * N_REC                                    # sense bytes per band: levels, then changes


@njit(cache=True)
def _fmax(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a >= b else b


@njit(cache=True)
def _fmin(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a <= b else b


@njit(cache=True)
def _num(v):                                       # np.nan_to_num for one value
    if v != v:
        return 0.0
    if v == np.inf:
        return 1.7976931348623157e308
    if v == -np.inf:
        return -1.7976931348623157e308
    return v


@njit(cache=True)
def _rint(x):
    f = np.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1.0
    if d < 0.5:
        return f
    return f if f - 2.0 * np.floor(f / 2.0) == 0.0 else f + 1.0


@njit(cache=True)
def _feed(rows, taus, step_s, t0, bar, fsum, prev, hist, count, cur, env, done, band):
    """Advance the weather through `rows`; fill env (T, N_ENV), done (T, Y) and band (T, Y, NS)."""
    Y = taus.shape[0]
    W = hist.shape[3]
    t = t0
    level = np.empty(N_REC)
    x = np.empty((2, N_REC))
    for i in range(rows.shape[0]):
        row = rows[i]
        t = (row[_OPEN_TIME] + 1000.0 * step_s) / 1000.0
        o, h, l, c = row[_OPEN], row[_HIGH], row[_LOW], row[_CLOSE]
        for y in range(Y):
            if bar[y, 0] != bar[y, 0]:
                bar[y, 0] = o
                bar[y, 1] = h
                bar[y, 2] = l
                for k in range(4, 8):
                    bar[y, k] = 0.0
            bar[y, 1] = _fmax(bar[y, 1], h)
            bar[y, 2] = _fmin(bar[y, 2], l)
            bar[y, 3] = c
            bar[y, 4] += _num(row[_VOLUME])
            bar[y, 5] += _num(row[_QUOTE])
            bar[y, 6] += _num(row[_TRADES])
            bar[y, 7] += _num(row[_TAKER])
        f = row[_FUNDING]
        if f != 0.0 and f == f:
            for y in range(Y):
                fsum[y] += f
        # the market as the soup meets it this tick
        env[i, E_T] = t
        env[i, E_OPEN] = o
        env[i, E_CLOSE] = c
        hi, lo = row[_MARK_HIGH], row[_MARK_LOW]
        if not (hi > 0 and lo > 0):
            hi, lo = h, l
        env[i, E_HIGH] = hi
        env[i, E_LOW] = lo
        env[i, E_MARK] = row[_MARK_CLOSE] if row[_MARK_CLOSE] > 0 else c
        env[i, E_FUNDING] = f
        tr = _rint(t)
        for y in range(Y):
            done[i, y] = 0
            if tr % taus[y] != 0:
                continue
            done[i, y] = 1
            # the bar closes: each stream's level (Planet._close_bar)
            for k in range(N_REC):
                level[k] = np.nan
            for k in range(7):
                v = bar[y, _BAR_OF_FAST[k]]
                if k < 3:
                    level[_FAST[k]] = np.log(v) if v > 0 else np.nan
                elif v != v:
                    level[_FAST[k]] = np.nan
                else:
                    level[_FAST[k]] = np.log1p(v if v > 0 else 0.0)
            for k in range(_LOGS.shape[0]):
                v = row[_LOGS[k, 1]]
                level[_LOGS[k, 0]] = np.log(v) if v > 0 else np.nan
            for k in range(_LOG1PS.shape[0]):
                v = row[_LOG1PS[k, 1]]
                level[_LOG1PS[k, 0]] = np.log1p(v) if v >= 0 else np.nan
            level[_PREMIUM_R] = row[_PREMIUM]
            level[_FUNDING_R] = fsum[y]
            tm = t * 1000
            day = (tm / 86_400_000.0) % 1.0
            week = ((tm / 86_400_000.0 + 3) % 7) / 7.0
            cyc = (tm % EIGHT_HOURS) / EIGHT_HOURS
            level[_CLOCKS[0]] = np.sin(2 * np.pi * day)
            level[_CLOCKS[1]] = np.cos(2 * np.pi * day)
            level[_CLOCKS[2]] = np.sin(2 * np.pi * week)
            level[_CLOCKS[3]] = np.cos(2 * np.pi * week)
            level[_CLOCKS[4]] = np.sin(2 * np.pi * cyc)
            level[_CLOCKS[5]] = np.cos(2 * np.pi * cyc)
            for k in range(8):
                bar[y, k] = np.nan
            fsum[y] = 0.0
            # and how it feels: percentile of its own recent history (Planet._transduce)
            slot = count[y] % W
            for k in range(N_REC):
                x[0, k] = level[k]
                x[1, k] = level[k] - prev[y, k]
                prev[y, k] = level[k]
                hist[y, 0, k, slot] = x[0, k]
                hist[y, 1, k, slot] = x[1, k]
            count[y] += 1
            for m in range(2):
                for k in range(N_REC):
                    xv = x[m, k]
                    below = 0
                    same = 0
                    n = 0
                    for w in range(W):
                        hv = hist[y, m, k, w]
                        if math.isfinite(hv):
                            n += 1
                            if hv < xv:
                                below += 1
                            elif hv == xv:
                                same += 1
                    pct = (below + 0.5 * same) / max(n, 1) if n > 0 else 0.5
                    if not (math.isfinite(xv) and n >= 16):
                        pct = 0.5
                    sgn = min(max(_rint(255 * pct - 127.5), -128.0), 127.0)
                    cur[y, m * N_REC + k] = np.int64(sgn) & 255
        for y in range(Y):
            for k in range(cur.shape[1]):
                band[i, y, k] = cur[y, k]
    return t


class Weather:
    """The bands' bars and senses, carried from one chunk of rows to the next."""

    def __init__(self, taus, step_s=1, window=256):
        self.taus = np.asarray(taus, np.int64)
        self.step_s = int(step_s)
        Y = len(self.taus)
        self.t = 0.0
        self.bar = np.full((Y, 8), np.nan)
        self.fsum = np.zeros(Y)
        self.prev = np.full((Y, N_REC), np.nan)
        self.hist = np.full((Y, 2, N_REC, window), np.nan)
        self.count = np.zeros(Y, np.int64)
        self.cur = np.zeros((Y, NS), np.uint8)

    def feed(self, rows):
        """(env (T, N_ENV) float64, done (T, Y) uint8, band (T, Y, NS) uint8) for these rows."""
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        T, Y = len(rows), len(self.taus)
        env = np.empty((T, N_ENV))
        done = np.empty((T, Y), np.uint8)
        band = np.empty((T, Y, NS), np.uint8)
        if T:
            self.t = _feed(rows, self.taus, self.step_s, self.t, self.bar, self.fsum, self.prev,
                           self.hist, self.count, self.cur, env, done, band)
        return env, done, band
