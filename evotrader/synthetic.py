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


def synthetic_rows(n, planted=True, seed=0, start_price=60000.0,
                   start_ms=1_700_006_400_000, crowd_effect=0.02, flow_effect=0.1):
    rng = np.random.default_rng(seed)
    crowd_effect = crowd_effect if planted else 0.0
    flow_effect = flow_effect if planted else 0.0
    rows = np.full((n, len(COLUMNS)), np.nan)
    dt_y = 1.0 / (365 * 1440)
    regimes = [(-0.6, 0.8), (0.0, 0.5), (0.6, 0.6)]          # (annual drift, annual vol)
    state, logvol, crowd, oi = 1, 0.0, 0.0, 0.0
    price = start_price
    flow = 0.0
    premium_8h = []
    metrics = None
    t0 = start_ms // MINUTE * MINUTE
    for k in range(n):
        t = t0 + k * MINUTE
        if rng.random() < 1.0 / (3 * 1440):
            state = rng.integers(len(regimes))
        mu, sig = regimes[state]
        logvol = 0.997 * logvol + 0.05 * rng.normal()
        s = sig * np.exp(logvol) * np.sqrt(dt_y)              # this minute's volatility
        crowd += -crowd / 1440 + np.sqrt(2.0 / 1440) * rng.normal()
        drift = (mu - 0.5 * sig * sig) * dt_y - crowd_effect * crowd * s
        r = drift + flow_effect * flow * s + s * rng.standard_t(5) / np.sqrt(5 / 3)
        o, c = price, price * np.exp(r)
        h = max(o, c) * np.exp(abs(rng.normal(0, 0.3 * s)))
        lo = min(o, c) * np.exp(-abs(rng.normal(0, 0.3 * s)))
        price = c
        flow = float(np.clip(rng.normal(0, 0.25), -1, 1))     # next minute's taker imbalance
        vol = 4.0 * np.exp(rng.normal(0, 0.4)) * (1 + 150 * abs(r))
        prem = 0.0004 * crowd + 0.0001 * rng.normal()
        premium_8h.append(prem)
        settle = (t + MINUTE) % EIGHT_HOURS == 0
        funding = 0.0
        if settle:
            p = float(np.mean(premium_8h))
            funding = p + float(np.clip(0.0001 - p, -0.0005, 0.0005))
            premium_8h = []
        oi += -oi / 20000 + 0.003 * rng.normal()
        if t % (5 * MINUTE) == 0:
            oi_value = np.exp(22.5 + 0.25 * abs(crowd) + oi)
            metrics = (oi_value / c, oi_value, np.exp(0.2 * crowd + 0.15 * rng.normal()),
                       np.exp(0.3 * crowd + 0.1 * rng.normal()), np.exp(0.2 * rng.normal()),
                       np.exp(0.8 * flow + 0.1 * rng.normal()))
        spot = c * np.exp(-(0.0002 * crowd + 0.00005 * rng.normal()))
        row = rows[k]
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
        if metrics is not None:
            for name, value in zip(("open_interest", "open_interest_value", "top_account_ls",
                                    "top_position_ls", "account_ls", "taker_ls"), metrics):
                row[C[name]] = value
        row[C["perp"]] = 1.0
    return rows
