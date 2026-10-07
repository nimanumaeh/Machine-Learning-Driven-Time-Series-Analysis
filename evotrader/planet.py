"""A planet made of the BTC market (see docs/planet.md).

Geography
  latitude  = timescale. Band y aggregates the 1-second rows into its own
              bars of TAU[y] seconds: the equator ticks every second, the
              poles once a day. An organism's latitude sets how fast its
              world moves and how long its crops take to grow.
  longitude = which senses exist there. Each regional group of receptors
              (order flow, premium, spot, open interest, positioning,
              funding, mark) has continents, a smooth random geography fixed
              at the planet's birth. Price, volume and clocks exist everywhere.
Weather       each band's bars, read through the same generic receptors as
              selfmade.py (log of a positive quantity, the raw value of a rate).
Water         a place supports more organisms when its band trades more volume
              than usual.
Culture       observatories (an organism's perception made public in a place)
              and markers (traces of how life has gone there, and how crowded
              it is) are extra senses of a place, built and maintained by its
              inhabitants.
"""

import numpy as np

from .data import C, EIGHT_HOURS
from .selfmade import LAGS, NORMS, N_REC, OPS, RECEPTORS, ZERO

TAU = np.array([1, 5, 30, 120, 600, 3600, 21600, 86400])
REGIONS = {
    "flow": ("taker_buy",),
    "premium": ("premium",),
    "spot": ("spot", "spot_volume", "spot_taker_buy"),
    "open_interest": ("open_interest", "open_interest_value"),
    "positioning": ("top_account_ls", "top_position_ls", "account_ls", "taker_ls"),
    "funding": ("funding",),
    "mark": ("mark",),
}
N_OBS = 2                                     # observatory slots per place
CULTURE = N_OBS + 2                           # + growth marker + crowding marker
N_CH = N_REC + 1 + CULTURE                    # every channel a program may read
GROWTH, CROWD = N_REC + 1 + N_OBS, N_REC + 2 + N_OBS
FAST = [RECEPTORS.index(n) for n in ("close", "high", "low", "volume", "quote_volume",
                                     "trades", "taker_buy")]


def _smooth_field(rng, n, waves=4):
    x = np.arange(n) / n
    f = sum(rng.normal() * np.sin(2 * np.pi * (k + 1) * x + rng.uniform(0, 2 * np.pi)) / (k + 1)
            for k in range(waves))
    return (f - f.mean()) / (f.std() + 1e-12)


class Planet:
    def __init__(self, width=48, seed=0, history=512, base_capacity=16, obs_decay=0.005,
                 marker_memory=256):
        rng = np.random.default_rng(seed)
        self.X, self.Y, self.H = width, len(TAU), history
        self.base_capacity = base_capacity
        self.obs_decay, self.marker_memory = obs_decay, marker_memory
        self.avail = np.ones((self.X, self.Y, N_REC + 1), bool)
        self.regions = {}
        for name, receptors in REGIONS.items():
            field = _smooth_field(rng, self.X)[:, None] + 0.3 * rng.normal(size=(1, self.Y))
            here = field > np.quantile(field, 0.45)             # about 55% of the surface
            self.regions[name] = here
            for r in receptors:
                self.avail[:, :, RECEPTORS.index(r)] = here
        # each band's bar under construction
        Y = self.Y
        self.bar = np.full((Y, 8), np.nan)                      # o h l c vol quote trades taker
        self.funding_mark = np.zeros(Y)                         # sum of rate x mark this bar
        self.funding_sum = np.zeros(Y)
        self.last = None                                        # latest 1-second row
        # history in band ticks
        self.rbuf = np.full((Y, self.H, N_REC + 1), np.nan)
        self.pbuf = np.full((Y, self.H, 4), np.nan)             # close, low, high, funding x mark
        self.ticks = np.zeros(Y, np.int64)
        self.vol_fast = np.zeros(Y)
        self.vol_slow = np.zeros(Y)
        self.water = np.ones(Y)
        # culture of each place
        self.obs_prog = np.zeros((self.X, Y, N_OBS, 5), np.int64)
        self.obs_integrity = np.zeros((self.X, Y, N_OBS))
        self.obs_builder = np.full((self.X, Y, N_OBS), -1, np.int64)
        self.obs_state = np.zeros((self.X, Y, N_OBS, 4))        # ema, mean, var, count
        self.obs_value = np.full((self.X, Y, N_OBS), np.nan)
        self.obs_born = np.zeros((self.X, Y, N_OBS))             # when each was built
        self.growth = np.zeros((self.X, Y))
        self.crowding = np.zeros((self.X, Y))
        self.t = 0.0

    # ----------------------------------------------------------------- time
    def step(self, row):
        """Take one 1-second row; return the bands whose tick just completed."""
        self.t = (row[C["open_time"]] + 1000) / 1000.0
        o, h, l, c = row[C["open"]], row[C["high"]], row[C["low"]], row[C["close"]]
        new = np.isnan(self.bar[:, 0])
        self.bar[new, 0] = o
        self.bar[new, 1] = h
        self.bar[new, 2] = l
        self.bar[new, 4:] = 0.0
        self.bar[:, 1] = np.fmax(self.bar[:, 1], h)
        self.bar[:, 2] = np.fmin(self.bar[:, 2], l)
        self.bar[:, 3] = c
        self.bar[:, 4] += np.nan_to_num(row[C["volume"]])
        self.bar[:, 5] += np.nan_to_num(row[C["quote_volume"]])
        self.bar[:, 6] += np.nan_to_num(row[C["trades"]])
        self.bar[:, 7] += np.nan_to_num(row[C["taker_buy_volume"]])
        f = row[C["funding_rate"]]
        if f and f == f:
            self.funding_sum += f
            self.funding_mark += f * row[C["mark_close"]] if row[C["mark_close"]] > 0 else f * c
        self.last = row
        done = np.nonzero(np.round(self.t) % TAU == 0)[0]
        for y in done:
            self._close_bar(y)
        return done

    def _close_bar(self, y):
        b, row = self.bar[y], self.last
        rec = np.full(N_REC + 1, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            fast = np.array([b[3], b[1], b[2], b[4], b[5], b[6], b[7]])
            rec[FAST] = np.where(np.arange(7) < 3, np.log(np.where(fast > 0, fast, np.nan)),
                                 np.log1p(np.maximum(fast, 0)))
            for name, col in (("mark", "mark_close"), ("spot", "spot_close"),
                              ("open_interest", "open_interest"),
                              ("open_interest_value", "open_interest_value"),
                              ("top_account_ls", "top_account_ls"),
                              ("top_position_ls", "top_position_ls"),
                              ("account_ls", "account_ls"), ("taker_ls", "taker_ls")):
                v = row[C[col]]
                rec[RECEPTORS.index(name)] = np.log(v) if v > 0 else np.nan
            for name, col in (("spot_volume", "spot_volume"),
                              ("spot_taker_buy", "spot_taker_buy_volume")):
                v = row[C[col]]
                rec[RECEPTORS.index(name)] = np.log1p(v) if v >= 0 else np.nan
        rec[RECEPTORS.index("premium")] = row[C["premium"]]
        rec[RECEPTORS.index("funding")] = self.funding_sum[y]
        t = self.t * 1000
        day, week = (t / 86_400_000.0) % 1.0, ((t / 86_400_000.0 + 3) % 7) / 7.0
        cyc = (t % EIGHT_HOURS) / EIGHT_HOURS
        for name, v in (("day_sin", np.sin(2 * np.pi * day)), ("day_cos", np.cos(2 * np.pi * day)),
                        ("week_sin", np.sin(2 * np.pi * week)), ("week_cos", np.cos(2 * np.pi * week)),
                        ("cycle_sin", np.sin(2 * np.pi * cyc)), ("cycle_cos", np.cos(2 * np.pi * cyc))):
            rec[RECEPTORS.index(name)] = v
        rec[ZERO] = 0.0
        slot = self.ticks[y] % self.H
        self.rbuf[y, slot] = rec
        self.pbuf[y, slot] = (b[3], b[2], b[1], self.funding_mark[y])
        self.ticks[y] += 1
        # water: this band's volume relative to its own long-run level
        v = b[4]
        self.vol_fast[y] += (v - self.vol_fast[y]) / 16
        self.vol_slow[y] += (v - self.vol_slow[y]) / 512 if self.ticks[y] > 1 else v
        if self.ticks[y] > 16 and self.vol_slow[y] > 0:
            self.water[y] = float(np.clip(self.vol_fast[y] / self.vol_slow[y], 0.5, 2.0))
        self.bar[y] = np.nan
        self.funding_sum[y] = self.funding_mark[y] = 0.0
        # what is built wears away and traces fade, in the band's own time
        self.obs_integrity[:, y] = np.maximum(self.obs_integrity[:, y] - self.obs_decay, 0.0)
        self.growth[:, y] *= 1.0 - 1.0 / self.marker_memory
        self._run_observatories(y)

    def capacity(self):
        """How many organisms each place (x, y) can hold now."""
        per_band = np.maximum(1, np.round(self.base_capacity * self.water)).astype(np.int64)
        return np.broadcast_to(per_band[None, :], (self.X, self.Y)).copy()

    # ------------------------------------------------------------- culture
    def _run_observatories(self, y):
        live = self.obs_integrity[:, y] > 0                       # (X, N_OBS)
        if not live.any():
            self.obs_value[:, y] = np.nan
            return
        p = self.obs_prog[:, y]
        v, st = evaluate(p, self.rbuf[y], self.ticks[y], self.obs_state[:, y])
        self.obs_state[:, y] = st
        self.obs_value[:, y] = np.where(live, v, np.nan)

    def culture(self, x, y):
        """The place's own senses for organisms at cells (x, y): (n, CULTURE)."""
        return np.concatenate([self.obs_value[x, y], self.growth[x, y][:, None],
                               self.crowding[x, y][:, None]], axis=1)

    def build(self, x, y, program, builder, integrity=1.0, replace_below=0.5):
        """Put a program into a place's weakest observatory slot.

        Only a slot that is empty or has worn below `replace_below` can be built
        over: what a place keeps maintained is not torn down by a newcomer.
        """
        weakest = int(np.argmin(self.obs_integrity[x, y]))
        if self.obs_integrity[x, y, weakest] >= replace_below:
            return False
        self.obs_prog[x, y, weakest] = program
        self.obs_integrity[x, y, weakest] = integrity
        self.obs_builder[x, y, weakest] = builder
        self.obs_born[x, y, weakest] = self.t
        self.obs_state[x, y, weakest] = 0.0
        self.obs_value[x, y, weakest] = np.nan
        return True


def evaluate(prog, rbuf, ticks, state):
    """Run market-receptor programs one band tick forward.

    prog: (..., 5) [a, b, op, lag, norm]; rbuf: (H, N_REC + 1) band history;
    state: (..., 4) [ema, mean, var, count]. Returns (values, new state).
    """
    H = len(rbuf)
    a, b, op = prog[..., 0], prog[..., 1], prog[..., 2]
    k, tau = LAGS[prog[..., 3]], NORMS[prog[..., 4]]
    cur = rbuf[(ticks - 1) % H]
    x = cur[np.minimum(a, ZERO)] - cur[np.minimum(b, ZERO)]
    rows = (ticks - 1 - k) % H
    past = np.where((ticks - 1 >= k) & (k < H), rbuf[rows, np.minimum(a, ZERO)]
                    - rbuf[rows, np.minimum(b, ZERO)], np.nan)
    return _normalize(x, x - past, op, k, tau, state)


def _normalize(x, change, op, k, tau, state):
    ema, mean, var, cnt = (state[..., i] for i in range(4))
    fresh = cnt == 0
    ok_x = np.isfinite(x)
    ema = np.where(ok_x, np.where(fresh, x, ema + (x - ema) / k), ema)
    raw = np.select([op == 0, op == 1, op == 2], [x, change, ema], np.abs(change))
    ok = np.isfinite(raw)
    d = raw - mean
    at = 1.0 / tau
    mean = np.where(ok, np.where(fresh, raw, mean + at * d), mean)
    var = np.where(ok, np.where(fresh, 0.0, (1 - at) * (var + at * d * d)), var)
    cnt = cnt + ok
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (raw - mean) / np.sqrt(var + 1e-18)
    live = ok & (cnt >= np.minimum(tau, 32))
    return (np.where(live, np.clip(z, -5, 5), np.nan),
            np.stack([ema, mean, var, cnt], -1))


__all__ = ["Planet", "TAU", "N_CH", "N_OBS", "CULTURE", "GROWTH", "CROWD", "evaluate",
           "_normalize", "OPS"]
