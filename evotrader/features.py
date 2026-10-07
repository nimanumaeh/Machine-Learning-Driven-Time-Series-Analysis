"""Bar aggregation and the market features agents get to see.

Every timeframe niche builds its own candles out of the base feed and turns
the most recent ones into a small, volatility-normalised feature vector. Only
closed candles are ever used, so there is no lookahead.
"""

from collections import deque

import numpy as np

HISTORY = 64            # candles kept per timeframe (also the warmup length)
_RET_LAGS = (1, 2, 4, 8, 16)
FEATURE_NAMES = (
    [f"ret_{k}" for k in _RET_LAGS]
    + ["vol_regime", "dev_ma20", "dev_ma50", "volume_z", "range", "rsi"]
)
N_FEATURES = len(FEATURE_NAMES)


class TimeframeBook:
    """Aggregates base bars into candles of `tf` seconds and keeps history."""

    def __init__(self, tf):
        self.tf = tf
        self.tf_ms = tf * 1000
        self.bucket = None
        self.cur = None                 # [open, high, low, close, volume]
        self.closes = deque(maxlen=HISTORY + 1)
        self.highs = deque(maxlen=HISTORY + 1)
        self.lows = deque(maxlen=HISTORY + 1)
        self.vols = deque(maxlen=HISTORY + 1)

    @property
    def ready(self):
        return len(self.closes) > HISTORY

    def add(self, open_ms, dur_ms, o, h, l, c, v):
        """Feed one base bar. Returns True when a candle of this timeframe closed."""
        bucket = open_ms // self.tf_ms
        closed = False
        if self.bucket is not None and bucket != self.bucket:
            # Data gap: the previous candle never saw its final base bar.
            self._close()
            closed = True
        if self.bucket != bucket:
            self.bucket = bucket
            self.cur = [o, h, l, c, v]
        else:
            cur = self.cur
            cur[1] = max(cur[1], h)
            cur[2] = min(cur[2], l)
            cur[3] = c
            cur[4] += v
        if (open_ms + dur_ms) % self.tf_ms == 0:
            self._close()
            return True
        return closed

    def _close(self):
        _, h, l, c, v = self.cur
        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(l)
        self.vols.append(v)
        self.bucket = None
        self.cur = None

    def features(self):
        closes = np.asarray(self.closes, dtype=np.float64)
        r = np.diff(np.log(closes))                     # HISTORY returns
        vol = r[-32:].std() + 1e-9
        self.vol = vol                                  # per-candle volatility
        f = np.empty(N_FEATURES)
        for j, k in enumerate(_RET_LAGS):
            f[j] = r[-k:].sum() / (vol * np.sqrt(k))
        f[5] = np.log((r[-8:].std() + 1e-9) / vol)
        f[6] = np.log(closes[-1] / closes[-20:].mean()) / (vol * np.sqrt(10))
        f[7] = np.log(closes[-1] / closes[-50:].mean()) / (vol * np.sqrt(25))
        lv = np.log(np.asarray(self.vols, dtype=np.float64)[-32:] + 1e-9)
        f[8] = (lv[-1] - lv.mean()) / (lv.std() + 1e-9)
        f[9] = np.log(self.highs[-1] / self.lows[-1]) / vol - 1.0
        last = r[-14:]
        up, down = last[last > 0].sum(), -last[last < 0].sum()
        f[10] = (up - down) / (up + down + 1e-12)        # RSI rescaled to [-1, 1]
        return np.clip(f, -4.0, 4.0) / 2.0
