"""A synthetic BTCUSDT perpetual market with every data column, for testing.

A hidden "crowding" variable c (how one-sided leveraged positioning is,
mean-reverting over about a day) shows up, noisily, in the premium, the
funding rate, the basis to spot, open interest and the long/short ratios.
In the planted world, crowding also pushes the price the other way (crowded
longs get squeezed): drift = -effect * c * sigma per minute, which is worth
about 0.3 standard deviations over four hours when |c| = 1, several times the
fees. Order flow carries a small, real but unprofitable one-minute effect.
Everything else (trades, spot flow, account ratios, the time of day) is a
decoy. In the null world the effects are zero and the same streams still
move together, so they look meaningful while predicting nothing.

Only for testing the machinery: nothing learned here transfers to BTC.
"""

import numpy as np

from .data import COLUMNS, C, EIGHT_HOURS, MINUTE


class SyntheticMarket:
    """A synthetic market that can be read in pieces: rows(n) continues where it stopped.

    One row every step_s seconds (60: like the minute store, 1: like the seconds
    store). Effects are scaled with the step so that their strength over hours is
    the same at any resolution.
    """

    REGIMES = [(-0.6, 0.8), (0.0, 0.5), (0.6, 0.6)]            # (annual drift, annual vol)

    def __init__(self, planted=True, seed=0, start_price=60000.0, start_ms=1_700_006_400_000,
                 crowd_effect=0.02, flow_effect=0.1, step_s=60):
        self.rng = np.random.default_rng(seed)
        self.step_s = step_s
        self.crowd_effect = (crowd_effect if planted else 0.0) * np.sqrt(step_s / 60)
        self.flow_effect = flow_effect if planted else 0.0
        self.per_day = 86400 / step_s
        self.dt_y = step_s / (365 * 86400)
        self.phi = 0.997 ** (step_s / 60)
        self.vol_noise = 0.645 * np.sqrt(1 - self.phi * self.phi)
        self.state, self.logvol, self.crowd, self.oi, self.flow = 1, 0.0, 0.0, 0.0, 0.0
        self.price = start_price
        self.premium_8h = []
        self.metrics = None
        self.step = step_s * 1000
        self.t = start_ms // self.step * self.step

    def rows(self, n):
        rng, step_s = self.rng, self.step_s
        out = np.full((n, len(COLUMNS)), np.nan)
        for k in range(n):
            t = self.t
            self.t += self.step
            if rng.random() < 1.0 / (3 * self.per_day):
                self.state = rng.integers(len(self.REGIMES))
            mu, sig = self.REGIMES[self.state]
            self.logvol = self.phi * self.logvol + self.vol_noise * rng.normal()
            s = sig * np.exp(self.logvol) * np.sqrt(self.dt_y)  # this step's volatility
            self.crowd += -self.crowd / self.per_day + np.sqrt(2.0 / self.per_day) * rng.normal()
            crowd = self.crowd
            drift = (mu - 0.5 * sig * sig) * self.dt_y - self.crowd_effect * crowd * s
            r = drift + self.flow_effect * self.flow * s + s * rng.standard_t(5) / np.sqrt(5 / 3)
            o, c = self.price, self.price * np.exp(r)
            h = max(o, c) * np.exp(abs(rng.normal(0, 0.3 * s)))
            lo = min(o, c) * np.exp(-abs(rng.normal(0, 0.3 * s)))
            self.price = c
            flow = self.flow = float(np.clip(rng.normal(0, 0.25), -1, 1))   # next step's taker imbalance
            vol = 4.0 * (step_s / 60) * np.exp(rng.normal(0, 0.4)) * (1 + 150 * abs(r) * np.sqrt(60 / step_s))
            prem = 0.0004 * crowd + 0.0001 * rng.normal()
            self.premium_8h.append(prem)
            funding = 0.0
            if (t + self.step) % EIGHT_HOURS == 0:
                p = float(np.mean(self.premium_8h))
                funding = p + float(np.clip(0.0001 - p, -0.0005, 0.0005))
                self.premium_8h = []
            self.oi += (-self.oi / 20000 + 0.003 * rng.normal()) * np.sqrt(step_s / 60)
            if t % (5 * MINUTE) == 0:
                oi_value = np.exp(22.5 + 0.25 * abs(crowd) + self.oi)
                self.metrics = (oi_value / c, oi_value, np.exp(0.2 * crowd + 0.15 * rng.normal()),
                                np.exp(0.3 * crowd + 0.1 * rng.normal()), np.exp(0.2 * rng.normal()),
                                np.exp(0.8 * flow + 0.1 * rng.normal()))
            spot = c * np.exp(-(0.0002 * crowd + 0.00005 * rng.normal()))
            row = out[k]
            row[:9] = (t, o, h, lo, c, vol, vol * c, vol * 40 * np.exp(rng.normal(0, 0.3)),
                       vol * (0.5 + 0.5 * flow))
            row[C["mark_high"]] = max(o, c) + 0.5 * (h - max(o, c))
            row[C["mark_low"]] = min(o, c) - 0.5 * (min(o, c) - lo)
            row[C["mark_close"]] = c
            row[C["funding_rate"]] = funding
            row[C["premium"]] = prem
            sv = 0.6 * vol * np.exp(rng.normal(0, 0.3))
            row[C["spot_close"]], row[C["spot_volume"]] = spot, sv
            row[C["spot_taker_buy_volume"]] = sv * float(np.clip(0.5 + 0.2 * rng.normal(), 0, 1))
            if self.metrics is not None:
                for name, value in zip(("open_interest", "open_interest_value", "top_account_ls",
                                        "top_position_ls", "account_ls", "taker_ls"), self.metrics):
                    row[C[name]] = value
            row[C["perp"]] = 1.0
        return out

    def stream(self, n=None, chunk=86_400):
        """Yield rows one at a time, forever or n of them, generated a chunk at a time."""
        for block in self.blocks(n, chunk):
            yield from block

    def blocks(self, n=None, chunk=86_400):
        """Yield arrays of up to `chunk` rows, forever or n rows in all."""
        done = 0
        while n is None or done < n:
            block = np.asarray(self.rows(chunk if n is None else min(chunk, n - done)))
            done += len(block)
            yield block


def synthetic_rows(n, planted=True, seed=0, start_price=60000.0,
                   start_ms=1_700_006_400_000, crowd_effect=0.02, flow_effect=0.1, step_s=60):
    """n rows, one every step_s seconds (60: the minute store, 1: the seconds store)."""
    return SyntheticMarket(planted, seed, start_price, start_ms, crowd_effect, flow_effect,
                           step_s).rows(n)
