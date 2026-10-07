"""The ecosystem: play-money BTC perpetual traders that live or die by their equity.

Each living agent owns one BTCUSDT perpetual account (see ``exchange.py``) and
a genome:

* a tiny neural net mapping market features plus its own situation
  (position, unrealised PnL, distance to liquidation) to a target position
  between full short and full long;
* a timeframe it thinks in, a leverage setting, the share of its equity it
  is willing to put up as margin, a rebalance dead-band, and its own mutation
  step size.

Decisions are made on a candle's close and filled at the next bar's open.
The ecology only ever reads equity: falling below ``death_ratio`` of the
reference equity kills an agent; reaching ``repro_ratio`` of it splits the
account exactly in half with a mutated child. When the population is full a
newborn takes the slot of the weakest agent, if that agent is doing worse
than the parent. A separate control group of random agents that never
reproduce (and are replaced when they die) shows what luck alone earns.
"""

import numpy as np

from .exchange import Accounts
from .features import N_FEATURES, TimeframeBook

DAY = 86400.0
N_SELF = 3      # position, unrealised PnL, distance to liquidation


class World:
    def __init__(self, cfg, base_seconds):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n_in = N_FEATURES + N_SELF
        h = cfg.hidden
        self.n_params = self.n_in * h + h + h + 1

        S = cfg.capacity + cfg.n_control
        self.S = S
        self.acct = Accounts(S, cfg)
        self.alive = np.zeros(S, bool)
        self.control = np.zeros(S, bool)
        self.control[cfg.capacity:] = True
        self.agent_id = np.full(S, -1, np.int64)
        self.parent_id = np.full(S, -1, np.int64)
        self.root_id = np.full(S, -1, np.int64)
        self.generation = np.zeros(S, np.int64)
        self.tf = np.zeros(S, np.int64)
        self.params = np.zeros((S, self.n_params))
        self.sigma = np.zeros(S)
        self.deadband = np.zeros(S)
        self.lev = np.ones(S)                 # leverage setting (integer valued)
        self.frac = np.ones(S)                # share of equity put up as margin
        self.max_notional = np.zeros(S)
        self.pending = np.full(S, np.nan)     # target decided, filled next bar
        self.ref = np.ones(S)
        self.growth = np.ones(S)              # growth banked at previous splits
        self.born_t = np.zeros(S)
        self.last_split_t = np.zeros(S)
        self.children = np.zeros(S, np.int64)

        self.t = 0.0
        self.price = None                     # last trade price
        self.mark = None                      # last mark price
        self.next_id = 0
        self.injected = 0.0                   # play money handed to newcomers
        self.withdrawn = 0.0                  # equity left when agents died
        self.control_injected = 0.0
        self.control_withdrawn = 0.0
        self.bnh_units = None                 # 1x buy-and-hold yardstick
        self.bnh_start_t = None
        self.totals = {"fees": 0.0, "funding": 0.0, "liquidations": 0}
        self.counts = {"births": 0, "immigrants": 0, "starved": 0, "outcompeted": 0}
        self.events = []                      # drained by the store
        self._last_snapshot_t = None
        self.seeded = False
        self.reset_market(base_seconds)

    # ------------------------------------------------------------------ market
    def reset_market(self, base_seconds):
        """Start fresh candle histories for a feed with `base_seconds` bars."""
        self.base_seconds = base_seconds
        self.timeframes = [t for t in self.cfg.timeframes if t % base_seconds == 0]
        if not self.timeframes:
            raise ValueError(f"no configured timeframe is a multiple of {base_seconds}s")
        self.books = {tf: TimeframeBook(tf) for tf in self.timeframes}
        for i in np.nonzero(self.alive & ~np.isin(self.tf, self.timeframes))[0]:
            self.tf[i] = min(self.timeframes, key=lambda x: abs(x - self.tf[i]))

    def warm(self, open_ms, dur_ms, o, h, l, c, v):
        """Feed history into candle books without trading (e.g. before going live)."""
        for tf, book in self.books.items():
            if (tf * 1000) % dur_ms == 0:
                book.add(open_ms, dur_ms, o, h, l, c, v)

    def step(self, open_ms, o, h, l, c, v, mark_h, mark_l, mark_c, funding):
        """Advance by one closed base bar (trade OHLCV, mark range, funding rate)."""
        dur_ms = self.base_seconds * 1000
        self.t = (open_ms + dur_ms) / 1000.0
        if not self.seeded:
            self._seed()
        if self.bnh_units is None:
            self.bnh_units = self.cfg.initial_capital / o
            self.bnh_start_t = self.t

        self._execute(o)                               # orders decided last bar
        hit = self.acct.liquidate(mark_h, mark_l)
        self.totals["liquidations"] += int((~self.control[hit]).sum())
        if funding and funding == funding:             # skip 0 and NaN
            idx, pay = self.acct.settle_funding(funding, mark_c)
            self.totals["funding"] += float(pay[~self.control[idx]].sum())
        self.price, self.mark = c, mark_c

        for tf, book in self.books.items():
            if book.add(open_ms, dur_ms, o, h, l, c, v) and book.ready:
                self._decide(tf, book)

        self._ecology()
        every = self.cfg.snapshot_every_s
        if self._last_snapshot_t is None or self.t - self._last_snapshot_t >= every:
            self._last_snapshot_t = self.t
            self.events.append(("snapshot", self.snapshot()))

    # ---------------------------------------------------------------- policies
    def _forward(self, idx, X):
        n, ni, h = len(idx), self.n_in, self.cfg.hidden
        p = self.params[idx]
        W1 = p[:, : ni * h].reshape(n, ni, h)
        b1 = p[:, ni * h: ni * h + h]
        W2 = p[:, ni * h + h: ni * h + 2 * h]
        b2 = p[:, -1]
        hid = np.tanh(np.einsum("ni,nih->nh", X, W1) + b1)
        return np.tanh(np.einsum("nh,nh->n", hid, W2) + b2)

    def _decide(self, tf, book):
        idx = np.nonzero(self.alive & (self.tf == tf))[0]
        if len(idx) == 0:
            return
        feats = book.features()
        price = book.closes[-1]
        a = self.acct
        eq = np.maximum(a.equity(price, idx), 1e-9)
        budget = self.frac[idx] * self.lev[idx] * eq
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
        target = self._forward(idx, X)
        go = np.abs(target - pos) > self.deadband[idx]
        self.pending[idx[go]] = target[go]

    def _execute(self, price):
        idx = np.nonzero(self.alive & ~np.isnan(self.pending))[0]
        if len(idx) == 0:
            return
        u = self.pending[idx]
        self.pending[idx] = np.nan
        eq = np.maximum(self.acct.equity(price, idx), 0.0)
        notional = np.clip(u * self.frac[idx] * self.lev[idx] * eq,
                           -self.max_notional[idx], self.max_notional[idx])
        paid = self.acct.execute(idx, notional / price, price, self.lev[idx])
        self.totals["fees"] += float(paid[~self.control[idx]].sum())

    # ----------------------------------------------------------------- ecology
    def equity(self, idx=slice(None)):
        return self.acct.equity(self.mark, idx)

    def _ecology(self):
        cfg = self.cfg
        ratio = self.equity() / self.ref
        dying = self.alive & (ratio < cfg.death_ratio)
        for i in np.nonzero(dying & ~self.control)[0]:
            self._kill(i, "starved")
        for i in np.nonzero(dying & self.control)[0]:
            self._kill(i, "starved")
            self._spawn_random(int(self.tf[i]), slot=i)

        ready = self.alive & ~self.control & (ratio >= cfg.repro_ratio)
        ready &= (self.t - self.last_split_t) >= cfg.min_repro_age_days * DAY
        queue = sorted(np.nonzero(ready)[0], key=lambda j: -ratio[j])
        ids = {i: self.agent_id[i] for i in queue}
        for i in queue:
            if self.alive[i] and self.agent_id[i] == ids[i]:   # not displaced meanwhile
                self._reproduce(i)

        wild = self.alive & ~self.control
        sizes = {tf: int((wild & (self.tf == tf)).sum()) for tf in self.timeframes}
        short = max(0, cfg.min_population - sum(sizes.values()))
        while True:                        # newcomers go to the emptiest niche
            tf = min(sizes, key=sizes.get)
            if sizes[tf] >= cfg.min_per_niche and short == 0:
                break
            if self._spawn_random(tf) is None:
                break
            sizes[tf] += 1
            short = max(0, short - 1)

    def _free_slot(self):
        free = np.nonzero(~self.alive[: self.cfg.capacity])[0]
        return int(free[0]) if len(free) else None

    def _new_identity(self, i, parent=None):
        self.agent_id[i] = self.next_id
        self.next_id += 1
        self.alive[i] = True
        self.born_t[i] = self.last_split_t[i] = self.t
        self.growth[i] = 1.0
        self.children[i] = 0
        self.pending[i] = np.nan
        if parent is None:
            self.parent_id[i] = -1
            self.root_id[i] = self.agent_id[i]
            self.generation[i] = 0
        else:
            self.parent_id[i] = self.agent_id[parent]
            self.root_id[i] = self.root_id[parent]
            self.generation[i] = self.generation[parent] + 1

    def _set_leverage(self, i, lev):
        self.lev[i] = float(np.clip(round(lev), 1, self.cfg.max_leverage))
        self.max_notional[i] = self.acct.brackets.max_notional(self.lev[i])

    def _random_genome(self, i, tf):
        ni, h, rng = self.n_in, self.cfg.hidden, self.rng
        p = np.zeros(self.n_params)
        p[: ni * h] = rng.normal(0, 1 / np.sqrt(ni), ni * h)
        p[ni * h + h: ni * h + 2 * h] = rng.normal(0, 1 / np.sqrt(h), h)
        p[-1] = rng.normal(0, 0.5)
        self.params[i] = p
        self.tf[i] = tf
        self.sigma[i] = self.cfg.init_sigma
        self.deadband[i] = np.exp(rng.uniform(np.log(0.02), np.log(0.3)))
        self._set_leverage(i, np.exp(rng.uniform(0, np.log(self.cfg.max_leverage))))
        self.frac[i] = np.exp(rng.uniform(np.log(0.05), 0.0))

    def _spawn_random(self, tf, slot=None):
        i = self._free_slot() if slot is None else slot
        if i is None:
            return None
        self._random_genome(i, tf)
        self._new_identity(i)
        cap = self.cfg.initial_capital
        self.acct.open(i, cap)
        self.ref[i] = cap
        if self.control[i]:
            self.control_injected += cap
        else:
            self.injected += cap
            self.counts["immigrants"] += 1
        self.events.append(("birth", self._record(i, "immigrant")))
        return i

    def _seed(self):
        self.seeded = True
        tfs = self.timeframes
        for k in range(self.cfg.capacity // 2):
            self._spawn_random(tfs[k % len(tfs)])
        for k, slot in enumerate(range(self.cfg.capacity, self.S)):
            self._spawn_random(tfs[k % len(tfs)], slot=slot)

    def _reproduce(self, i):
        ratio = self.equity() / self.ref
        slot = self._free_slot()
        if slot is None:
            wild = np.nonzero(self.alive & ~self.control)[0]
            wild = wild[wild != i]
            if len(wild) == 0:
                return
            victim = wild[np.argmin(ratio[wild])]
            if ratio[victim] >= ratio[i]:
                return
            self._kill(victim, "outcompeted")
            slot = victim

        self._mutate_into(i, slot)
        self._new_identity(slot, parent=i)
        self.growth[i] *= ratio[i]
        self.acct.split(i, slot)
        self.ref[i] = self.ref[slot] = self.equity(i)
        self.last_split_t[i] = self.t
        self.children[i] += 1
        self.counts["births"] += 1
        self.events.append(("birth", self._record(slot, "born")))

    def _mutate_into(self, parent, child):
        cfg, rng = self.cfg, self.rng
        genome = self.params[parent].copy()
        mates = np.nonzero(self.alive & ~self.control & (self.tf == self.tf[parent])
                           & (self.equity() > self.ref))[0]
        mates = mates[mates != parent]
        if len(mates) and rng.random() < cfg.crossover_prob:
            mate = rng.choice(mates)
            mask = rng.random(self.n_params) < 0.5
            genome[mask] = self.params[mate][mask]
        sigma = float(np.clip(self.sigma[parent] * np.exp(0.2 * rng.normal()), 1e-3, 1.0))
        self.params[child] = genome + sigma * rng.normal(size=self.n_params)
        self.sigma[child] = sigma
        self.deadband[child] = float(np.clip(
            self.deadband[parent] * np.exp(0.2 * rng.normal()), 0.005, 0.5))
        self._set_leverage(child, self.lev[parent] * np.exp(0.25 * rng.normal()))
        self.frac[child] = float(np.clip(self.frac[parent] * np.exp(0.2 * rng.normal()), 0.02, 1.0))
        tf = int(self.tf[parent])
        if rng.random() < cfg.timeframe_mutation_prob:
            k = self.timeframes.index(tf) + rng.choice((-1, 1))
            tf = self.timeframes[int(np.clip(k, 0, len(self.timeframes) - 1))]
        self.tf[child] = tf

    def _kill(self, i, cause):
        eq = float(self.equity(i))
        a = self.acct
        if self.control[i]:
            self.control_withdrawn += eq
        else:
            self.withdrawn += eq
            self.counts[cause] += 1
        self.events.append(("death", {
            "id": int(self.agent_id[i]), "t": self.t, "cause": cause,
            "growth": float(self.growth[i] * eq / self.ref[i]),
            "trades": int(a.trades[i]), "fees": float(a.fees[i]),
            "funding": float(a.funding[i]), "liquidations": int(a.liquidations[i]),
            "children": int(self.children[i]),
        }))
        a.close(i)
        self.alive[i] = False
        self.pending[i] = np.nan

    # -------------------------------------------------------------- reporting
    def _record(self, i, kind):
        return {
            "id": int(self.agent_id[i]), "parent": int(self.parent_id[i]),
            "root": int(self.root_id[i]), "generation": int(self.generation[i]),
            "timeframe": int(self.tf[i]), "t": self.t, "kind": kind,
            "control": bool(self.control[i]), "leverage": float(self.lev[i]),
            "margin_frac": float(self.frac[i]), "deadband": float(self.deadband[i]),
            "sigma": float(self.sigma[i]),
            "genome": self.params[i].astype(np.float32).tobytes(),
        }

    def lifetime_growth(self):
        return np.maximum(self.growth * self.equity() / self.ref, 1e-12)

    def daily_log_growth(self, mask):
        age = np.maximum((self.t - self.born_t[mask]) / DAY, 1e-9)
        return np.log(self.lifetime_growth()[mask]) / age

    def snapshot(self):
        wild = self.alive & ~self.control
        ctrl = self.alive & self.control
        eq = self.equity()
        q = self.acct.q
        mature = wild & ((self.t - self.born_t) >= DAY)
        bnh = self.bnh_units * self.mark
        niches = {}
        for tf in self.timeframes:
            m = wild & (self.tf == tf)
            niches[str(tf)] = {
                "n": int(m.sum()),
                "equity": float(eq[m].sum()),
                "median_ratio": float(np.median(eq[m] / self.ref[m])) if m.any() else None,
            }
        levs = self.lev[wild]
        held = float(eq[wild].sum())
        ctrl_held = float(eq[ctrl].sum())
        return {
            "t": self.t, "price": self.price, "mark": self.mark,
            "population": int(wild.sum()),
            "long": int((wild & (q > 0)).sum()), "short": int((wild & (q < 0)).sum()),
            "equity": held, "injected": self.injected, "withdrawn": self.withdrawn,
            "net_pnl": held + self.withdrawn - self.injected,
            "control_net_pnl": ctrl_held + self.control_withdrawn - self.control_injected,
            "control_injected": self.control_injected,
            "bnh_return": bnh / self.cfg.initial_capital - 1.0,
            "max_generation": int(self.generation[wild].max()) if wild.any() else 0,
            "median_leverage": float(np.median(levs)) if len(levs) else None,
            "leverage_bands": {band: int(((levs >= lo) & (levs <= hi)).sum()) for band, lo, hi in
                               (("1-2x", 1, 2), ("3-5x", 3, 5), ("6-10x", 6, 10),
                                ("11-25x", 11, 25), ("26-50x", 26, 50), ("51-125x", 51, 125))},
            "mature_daily_log_growth": (float(np.median(self.daily_log_growth(mature)))
                                        if mature.any() else None),
            "totals": dict(self.totals),
            "niches": niches,
            "counts": dict(self.counts),
        }

    def leaderboard(self, n=10, min_age_days=1.0):
        wild = self.alive & ~self.control & ((self.t - self.born_t) >= min_age_days * DAY)
        idx = np.nonzero(wild)[0]
        if len(idx) == 0:
            return []
        dlg = self.daily_log_growth(wild)
        eq = self.equity()
        a = self.acct
        rows = []
        for k in np.argsort(-dlg)[:n]:
            i = idx[k]
            rows.append({
                "id": int(self.agent_id[i]), "root": int(self.root_id[i]),
                "generation": int(self.generation[i]), "timeframe": int(self.tf[i]),
                "age_days": (self.t - self.born_t[i]) / DAY,
                "lifetime_growth": float(self.lifetime_growth()[i]),
                "daily_log_growth": float(dlg[k]), "equity": float(eq[i]),
                "leverage": float(self.lev[i]), "margin_frac": float(self.frac[i]),
                "position_btc": float(a.q[i]), "trades": int(a.trades[i]),
                "liquidations": int(a.liquidations[i]), "children": int(self.children[i]),
            })
        return rows

    def reset_economy(self):
        """Give every living agent a fresh, flat account (e.g. when going live)."""
        cap = self.cfg.initial_capital
        for i in np.nonzero(self.alive)[0]:
            self.acct.open(i, cap)
        self.ref[self.alive] = cap
        self.growth[self.alive] = 1.0
        self.pending[:] = np.nan
        self.born_t[self.alive] = self.last_split_t[self.alive] = self.t
        self.injected = cap * float((self.alive & ~self.control).sum())
        self.control_injected = cap * float((self.alive & self.control).sum())
        self.withdrawn = self.control_withdrawn = 0.0
        self.totals = {"fees": 0.0, "funding": 0.0, "liquidations": 0}
        self.bnh_units = None
        self._last_snapshot_t = None
