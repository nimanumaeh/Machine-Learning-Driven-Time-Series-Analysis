"""BTCUSDT perpetual futures accounts, vectorised over many agents.

Modelled on Binance USDⓈ-M futures in one-way mode with isolated margin and
market orders. Nothing else is simulated:

* a buy fills at price + half_spread, a sell at price - half_spread;
* every fill pays ``taker_fee`` on its notional, out of the wallet;
* opening or adding locks notional / leverage of the wallet as margin;
* orders come in multiples of ``qty_step`` and must be worth at least
  ``min_notional`` to open or add (closing is always allowed);
* at each funding settlement the position pays (or receives)
  position size * mark price * funding rate, out of (into) its margin;
* when margin + unrealised PnL at the mark price falls to the maintenance
  margin, the position is liquidated and its margin is lost. The rest of the
  wallet is untouched (that is what isolated margin means).
"""

import numpy as np

_EPS = 1e-9


class Brackets:
    """Leverage/maintenance tiers by position notional."""

    def __init__(self, brackets):
        caps, levs, rates = (np.array(x, dtype=float) for x in zip(*brackets))
        self.caps, self.levs, self.rates = caps, levs, rates
        # Maintenance amount: keeps maintenance margin continuous across tiers.
        self.cum = np.concatenate(([0.0], np.cumsum(caps[:-1] * np.diff(rates))))

    def tier(self, notional):
        k = np.minimum(np.searchsorted(self.caps, notional), len(self.caps) - 1)
        return self.rates[k], self.cum[k]

    def max_notional(self, leverage):
        """Largest position (USDT) allowed at each leverage."""
        lev = np.asarray(leverage, dtype=float)[..., None]
        return np.where(self.levs >= lev, self.caps, 0.0).max(axis=-1)


class Accounts:
    def __init__(self, n, cfg):
        self.cfg = cfg
        self.brackets = Brackets(cfg.brackets)
        self.wallet = np.zeros(n)        # USDT not locked as margin
        self.q = np.zeros(n)             # position in BTC: + long, - short
        self.entry = np.zeros(n)         # average entry price
        self.margin = np.zeros(n)        # isolated margin of the position
        self.fees = np.zeros(n)          # lifetime fees paid
        self.funding = np.zeros(n)       # lifetime funding paid (negative = received)
        self.liquidations = np.zeros(n, np.int64)
        self.trades = np.zeros(n, np.int64)

    # ----------------------------------------------------------- lifecycle
    def open(self, i, capital):
        self.close(i)
        self.wallet[i] = capital

    def close(self, i):
        for arr in (self.wallet, self.q, self.entry, self.margin, self.fees,
                    self.funding, self.liquidations, self.trades):
            arr[i] = 0

    def split(self, parent, child):
        """Give `child` exactly half of everything `parent` holds."""
        self.close(child)
        for arr in (self.wallet, self.q, self.margin):
            arr[parent] /= 2.0
            arr[child] = arr[parent]
        self.entry[child] = self.entry[parent]

    # ------------------------------------------------------------- reading
    def equity(self, mark, idx=slice(None)):
        return self.wallet[idx] + self.margin[idx] + self.q[idx] * (mark - self.entry[idx])

    def liquidation_price(self, idx):
        q, e, m = self.q[idx], self.entry[idx], self.margin[idx]
        mmr, cum = self.brackets.tier(np.abs(q) * e)
        with np.errstate(divide="ignore", invalid="ignore"):
            long_p = (q * e - m - cum) / (q * (1 - mmr))
            short_p = (m - q * e + cum) / (-q * (1 + mmr))
        return np.where(q > 0, long_p, np.where(q < 0, short_p, np.nan))

    # ------------------------------------------------------------- trading
    def execute(self, idx, target_q, price, leverage):
        """Market orders taking positions `idx` toward `target_q` BTC at `price`.

        Returns the fees paid by each account.
        """
        cfg = self.cfg
        step, fee, hs = cfg.qty_step, cfg.taker_fee, cfg.half_spread
        q0 = self.q[idx]
        E, M, W = self.entry[idx].copy(), self.margin[idx].copy(), self.wallet[idx].copy()
        s0, st = np.sign(q0), np.sign(target_q)
        a0, at = np.abs(q0), np.abs(target_q)

        # 1) reduce, close, or close before flipping
        full = (s0 != 0) & (st != s0)
        part = (s0 != 0) & (st == s0) & (at < a0)
        c = np.where(full, a0, np.where(part, np.floor((a0 - at) / step + _EPS) * step, 0.0))
        c = np.minimum(c, a0)
        pc = price - s0 * hs
        share = np.divide(c, a0, out=np.zeros_like(c), where=a0 > 0)
        release = M * share
        fee_c = c * pc * fee
        W += release + c * (pc - E) * s0 - fee_c
        M -= release
        q = q0 - s0 * c
        gone = np.abs(q) < _EPS
        W += np.where(gone, M, 0.0)
        q, E, M = np.where(gone, 0.0, q), np.where(gone, 0.0, E), np.where(gone, 0.0, M)

        # 2) open or add, limited by what the wallet can margin
        a = np.abs(q)
        same = (st != 0) & ((np.sign(q) == st) | (a == 0))
        o = np.where(same, np.floor(np.maximum(at - a, 0.0) / step + _EPS) * step, 0.0)
        po = price + st * hs
        per_btc = po * (1.0 / leverage + fee)
        o = np.minimum(o, np.floor(np.maximum(W, 0.0) / per_btc / step + _EPS) * step)
        o = np.where(o * po >= cfg.min_notional, o, 0.0)
        new_a = a + o
        E = np.where(o > 0, (a * E + o * po) / np.maximum(new_a, _EPS), E)
        fee_o = o * po * fee
        M += o * po / leverage
        W -= o * po / leverage + fee_o
        q = np.where(o > 0, st * new_a, q)

        self.q[idx], self.entry[idx], self.margin[idx], self.wallet[idx] = q, E, M, W
        paid = fee_c + fee_o
        self.fees[idx] += paid
        self.trades[idx] += (c > 0) | (o > 0)
        return paid

    def liquidate(self, mark_high, mark_low):
        """Liquidate every position the mark price range touched. Returns indices."""
        idx = np.nonzero(self.q)[0]
        if len(idx) == 0:
            return idx
        p = self.liquidation_price(idx)
        q = self.q[idx]
        hit = idx[((q > 0) & (mark_low <= p)) | ((q < 0) & (mark_high >= p))]
        self.wallet[hit] += np.minimum(self.margin[hit], 0.0)   # funding debt, if any
        self.q[hit] = self.entry[hit] = self.margin[hit] = 0.0
        self.liquidations[hit] += 1
        return hit

    def settle_funding(self, rate, mark):
        """Funding settlement: longs pay shorts when the rate is positive."""
        idx = np.nonzero(self.q)[0]
        pay = self.q[idx] * mark * rate
        self.margin[idx] -= pay
        self.funding[idx] += pay
        return idx, pay
