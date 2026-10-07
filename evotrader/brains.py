"""NetBrain: the baseline mind. A small fixed neural net per agent, shaped only by
evolution (no learning within a lifetime).

Each agent reads 11 candle features of its own timeframe plus its situation
(position, unrealised PnL, distance to liquidation) and outputs a target
position between full short and full long, acted on only when it differs from
the current one by more than the agent's dead-band.
"""

import numpy as np

from .data import C, MINUTE
from .features import N_FEATURES, TimeframeBook

N_SELF = 3


class NetBrain:
    def __init__(self, cfg):
        self.cfg = cfg

    def attach(self, world):
        self.world = world
        cfg, S = self.cfg, world.S
        self.n_in = N_FEATURES + N_SELF
        h = cfg.hidden
        self.n_params = self.n_in * h + h + h + 1
        self.params = np.zeros((S, self.n_params))
        self.sigma = np.zeros(S)
        self.deadband = np.zeros(S)
        self.books = {tf: TimeframeBook(tf) for tf in world.timeframes}
        self.closed = []

    # -------------------------------------------------------------- perceive
    def observe(self, row, t):
        self.closed = []
        args = (int(row[C["open_time"]]), MINUTE, row[C["open"]], row[C["high"]],
                row[C["low"]], row[C["close"]], row[C["volume"]])
        for tf, book in self.books.items():
            if book.add(*args) and book.ready:
                self.closed.append(tf)

    # ---------------------------------------------------------------- decide
    def _forward(self, idx, X):
        n, ni, h = len(idx), self.n_in, self.cfg.hidden
        p = self.params[idx]
        W1 = p[:, : ni * h].reshape(n, ni, h)
        b1 = p[:, ni * h: ni * h + h]
        W2 = p[:, ni * h + h: ni * h + 2 * h]
        b2 = p[:, -1]
        hid = np.tanh(np.einsum("ni,nih->nh", X, W1) + b1)
        return np.tanh(np.einsum("nh,nh->n", hid, W2) + b2)

    def decide(self, t):
        w = self.world
        out_i, out_x = [], []
        for tf in self.closed:
            idx = np.nonzero(w.alive & (w.tf == tf))[0]
            if len(idx) == 0:
                continue
            book = self.books[tf]
            feats = book.features()
            price = book.closes[-1]
            a = w.acct
            eq = np.maximum(a.equity(price, idx), 1e-9)
            budget = w.frac[idx] * w.lev[idx] * eq
            pos = np.clip(a.q[idx] * price / budget, -1, 1)
            upnl = np.divide(a.q[idx] * (price - a.entry[idx]), a.margin[idx],
                             out=np.zeros(len(idx)), where=a.margin[idx] > 0)
            liq = a.liquidation_price(idx)
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.abs(np.log(price / liq)) / book.vol
            dist = np.where(np.isfinite(dist), np.minimum(dist, 8.0), 8.0)
            X = np.empty((len(idx), self.n_in))
            X[:, :N_FEATURES] = feats
            X[:, N_FEATURES] = pos
            X[:, N_FEATURES + 1] = np.clip(upnl, -1, 3)
            X[:, N_FEATURES + 2] = dist / 4.0 - 1.0
            u = self._forward(idx, X)
            go = np.abs(u - pos) > self.deadband[idx]
            out_i.append(idx[go])
            out_x.append(u[go] * w.frac[idx[go]] * w.lev[idx[go]])
        if not out_i:
            return np.empty(0, int), np.empty(0)
        return np.concatenate(out_i), np.concatenate(out_x)

    # --------------------------------------------------------------- genomes
    def new_genome(self, i):
        ni, h, rng = self.n_in, self.cfg.hidden, self.world.rng
        p = np.zeros(self.n_params)
        p[: ni * h] = rng.normal(0, 1 / np.sqrt(ni), ni * h)
        p[ni * h + h: ni * h + 2 * h] = rng.normal(0, 1 / np.sqrt(h), h)
        p[-1] = rng.normal(0, 0.5)
        self.params[i] = p
        self.sigma[i] = self.cfg.init_sigma
        self.deadband[i] = np.exp(rng.uniform(np.log(0.02), np.log(0.3)))

    def inherit(self, parent, child):
        w, rng, cfg = self.world, self.world.rng, self.cfg
        genome = self.params[parent].copy()
        mates = np.nonzero(w.alive & ~w.control & (w.tf == w.tf[parent])
                           & (w.equity() > w.ref))[0]
        mates = mates[(mates != parent) & (mates != child)]
        if len(mates) and rng.random() < cfg.crossover_prob:
            mate = rng.choice(mates)
            mask = rng.random(self.n_params) < 0.5
            genome[mask] = self.params[mate][mask]
        sigma = float(np.clip(self.sigma[parent] * np.exp(0.2 * rng.normal()), 1e-3, 1.0))
        self.params[child] = genome + sigma * rng.normal(size=self.n_params)
        self.sigma[child] = sigma
        self.deadband[child] = float(np.clip(
            self.deadband[parent] * np.exp(0.2 * rng.normal()), 0.005, 0.5))

    def clear(self, i):
        pass

    def genome_bytes(self, i):
        return self.params[i].astype(np.float32).tobytes()

    def snapshot(self):
        return {}

    def describe(self, i):
        return {"deadband": float(self.deadband[i])}
