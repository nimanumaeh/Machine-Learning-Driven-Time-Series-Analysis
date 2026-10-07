"""The aspect grammar: everything an agent could attend to, minute by minute.

An aspect is (stream, operator, scale). Streams are read off the market row
(price, order flow, basis, premium, funding, open interest, crowd positioning,
...); operators turn a stream into something comparable across eras:

  mom    sum of returns over ~scale minutes, in units of its typical size
  vol    volatility at this scale relative to four times longer
  dev    deviation of a level from its own recent mean, in standard deviations
  trend  short mean minus long mean of a level, in standard deviations

Scales run from 1 minute to 4 weeks, with exponentially weighted (O(1),
causal) statistics, so an aspect at time t only ever uses data up to t. An
aspect is unavailable (NaN) until its stream has been seen long enough, and
streams that were not recorded in an era (open interest before late 2020,
premium before 2020, ...) simply stay unavailable then.
"""

import numpy as np

from .data import C, EIGHT_HOURS, MINUTE

STREAMS = ("ret", "range", "volume", "trades", "flow", "basis", "spot_flow", "premium",
           "funding", "mark_gap", "open_interest", "top_account", "top_position",
           "accounts", "taker")
RET_SCALES = (1, 5, 15, 60, 240, 1440, 10080)
VOL_SCALES = (5, 15, 60, 240, 1440, 10080)
LVL_SCALES = (5, 15, 60, 240, 1440, 10080)
EMA_SCALES = tuple(sorted(set(RET_SCALES) | {4 * k for k in RET_SCALES} | set(LVL_SCALES)
                          | {4 * k for k in LVL_SCALES}))
_J = {k: j for j, k in enumerate(EMA_SCALES)}
TIME_ASPECTS = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "funding_clock")


def aspect_names():
    names = [f"ret.mom.{k}" for k in RET_SCALES] + [f"ret.vol.{k}" for k in VOL_SCALES]
    for s in STREAMS[1:]:
        names += [f"{s}.dev.{k}" for k in LVL_SCALES] + [f"{s}.trend.{k}" for k in LVL_SCALES]
    return names + [f"time.{n}" for n in TIME_ASPECTS]


NAMES = aspect_names()
N_ASPECTS = len(NAMES)
STREAM_OF = np.array([n.split(".")[0] for n in NAMES])


def _structure():
    """Each aspect as (stream, operator, scale) indices: its 'token' description."""
    streams = list(STREAMS) + ["time"]
    ops = ["mom", "vol", "dev", "trend", "clock"]
    scales = sorted(set(RET_SCALES) | set(VOL_SCALES) | set(LVL_SCALES)) + [0]
    out = []
    for n in NAMES:
        s, op, sc = (n.split(".") + ["0"])[:3]
        if s == "time":
            op, sc = "clock", "0"
        out.append((streams.index(s), ops.index(op), scales.index(int(sc))))
    return np.array(out), (len(streams), len(ops), len(scales))


STRUCTURE, STRUCTURE_SIZES = _structure()


def _log(x):
    return np.log(x) if x > 0 else np.nan


class AspectEngine:
    """Causal, incremental computation of every aspect from market rows."""

    def __init__(self):
        n, m = len(STREAMS), len(EMA_SCALES)
        self.alpha = 1.0 / np.array(EMA_SCALES, dtype=float)
        self.mean = np.zeros((n, m))
        self.var = np.zeros((n, m))
        self.sq = np.zeros(m)                       # EW mean of squared returns
        self.seen = np.zeros(n)                     # valid updates per stream
        self.prev_close = np.nan
        self.funding = np.nan
        self.values = np.full(N_ASPECTS, np.nan)
        self.available = np.zeros(N_ASPECTS, bool)
        self.ret = 0.0
        self.var_1m = np.nan                        # typical squared 1-minute return
        self._lvl = np.arange(1, n)
        self._k = np.array(LVL_SCALES)
        self._jk = np.array([_J[k] for k in LVL_SCALES])
        self._j4k = np.array([_J[4 * k] for k in LVL_SCALES])
        self._jr = np.array([_J[k] for k in RET_SCALES])
        self._jv = np.array([_J[k] for k in VOL_SCALES])
        self._jv4 = np.array([_J[4 * k] for k in VOL_SCALES])
        self._jday = _J[1440]
        self._sqrt_k = np.sqrt(np.array(RET_SCALES, dtype=float))
        self._ret_need = np.maximum(np.array(RET_SCALES), 60)
        self._vol_need = 4 * np.array(VOL_SCALES)

    def streams(self, row):
        close = row[C["close"]]
        ret = np.log(close / self.prev_close) if self.prev_close > 0 else np.nan
        self.prev_close = close
        if row[C["funding_rate"]] != 0:
            self.funding = row[C["funding_rate"]]
        vol, tb = row[C["volume"]], row[C["taker_buy_volume"]]
        svol, stb = row[C["spot_volume"]], row[C["spot_taker_buy_volume"]]
        return np.array([
            ret,
            _log(row[C["high"]] / row[C["low"]]) if row[C["low"]] > 0 else np.nan,
            np.log1p(vol),
            np.log1p(row[C["trades"]]),
            2 * tb / vol - 1 if vol > 0 else np.nan,
            _log(close / row[C["spot_close"]]),
            2 * stb / svol - 1 if svol > 0 else np.nan,
            row[C["premium"]],
            self.funding,
            _log(row[C["mark_close"]] / close),
            _log(row[C["open_interest_value"]]),
            _log(row[C["top_account_ls"]]),
            _log(row[C["top_position_ls"]]),
            _log(row[C["account_ls"]]),
            _log(row[C["taker_ls"]]),
        ])

    def update(self, row):
        x = self.streams(row)
        ok = np.isfinite(x)
        a = self.alpha
        d = x[ok, None] - self.mean[ok]
        self.mean[ok] += a * d
        self.var[ok] = (1 - a) * (self.var[ok] + a * d * d)
        self.seen[ok] += 1
        self.ret = x[0] if ok[0] else 0.0
        if ok[0]:
            self.sq += a * (x[0] * x[0] - self.sq)
        self.var_1m = self.sq[self._jday] if self.seen[0] > 60 else np.nan
        self._assemble(x, ok, int(row[C["open_time"]]) + MINUTE)
        return self.values, self.available

    def _assemble(self, x, ok, t_ms):
        n_ret = len(RET_SCALES) + len(VOL_SCALES)
        vals = self.values
        avail = self.available
        sigma = np.sqrt(self.sq[self._jday]) + 1e-12
        vals[:len(RET_SCALES)] = self.mean[0, self._jr] * self._sqrt_k / sigma
        vals[len(RET_SCALES):n_ret] = 0.5 * np.log((self.sq[self._jv] + 1e-20)
                                                   / (self.sq[self._jv4] + 1e-20))
        avail[:len(RET_SCALES)] = self.seen[0] >= self._ret_need
        avail[len(RET_SCALES):n_ret] = self.seen[0] >= self._vol_need
        m, v, lv = self.mean[1:], self.var[1:], len(LVL_SCALES)
        dev = (x[1:, None] - m[:, self._jk]) / np.sqrt(v[:, self._jk] + 1e-18)
        trend = (m[:, self._jk] - m[:, self._j4k]) / np.sqrt(v[:, self._j4k] + 1e-18)
        block = np.concatenate([dev, trend], axis=1)            # per stream: dev..., trend...
        vals[n_ret:-5] = block.ravel()
        seen = self.seen[1:, None]
        avail[n_ret:-5] = np.concatenate([ok[1:, None] & (seen >= self._k),
                                          np.broadcast_to(seen >= 4 * self._k, (len(m), lv))],
                                         axis=1).ravel()
        day = (t_ms / 86_400_000.0) % 1.0
        week = ((t_ms / 86_400_000.0 + 3) % 7) / 7.0           # epoch was a Thursday
        clock = (t_ms % EIGHT_HOURS) / EIGHT_HOURS
        vals[-5:] = (np.sin(2 * np.pi * day), np.cos(2 * np.pi * day),
                     np.sin(2 * np.pi * week), np.cos(2 * np.pi * week), 2 * clock - 1)
        avail[-5:] = True
        np.clip(vals, -5, 5, out=vals)
        avail &= np.isfinite(vals)
        vals[~avail] = np.nan
