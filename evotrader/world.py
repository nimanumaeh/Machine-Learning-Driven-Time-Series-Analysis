"""The ecosystem: a population of play-money BTC traders living or dying by PnL.

State is kept as struct-of-arrays over a fixed number of slots so the whole
population can be stepped with numpy. Each living agent owns:

* a genome: a tiny neural net (features + own exposure -> target exposure),
  a timeframe it trades on, a rebalance dead-band and its own mutation size;
* a wallet of cash and BTC, taxed by a small metabolism every bar;
* a reference capital. Falling below `death_ratio` of it kills the agent;
  reaching `repro_ratio` of it splits the wallet in half with a mutated child.

When the population is full a newborn takes the slot of the weakest agent,
but only if that agent is doing worse than the parent. A separate control
group of random, immortal, never-selected agents shows what luck alone earns.
"""

import numpy as np

from .features import N_FEATURES, TimeframeBook

DAY = 86400.0


class World:
    def __init__(self, cfg, base_seconds):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n_in = N_FEATURES + 1
        h = cfg.hidden
        self.n_params = self.n_in * h + h + h + 1

        S = cfg.capacity + cfg.n_control
        self.S = S
        self.alive = np.zeros(S, bool)
        self.control = np.zeros(S, bool)
        self.control[cfg.capacity:] = True
        self.agent_id = np.full(S, -1, np.int64)
        self.parent_id = np.full(S, -1, np.int64)
        self.root_id = np.full(S, -1, np.int64)
        self.generation = np.zeros(S, np.int64)
        self.tf = np.zeros(S, np.int64)
        self.params = np.zeros((S, self.n_params), np.float64)
        self.sigma = np.zeros(S)
        self.deadband = np.zeros(S)
        self.cash = np.zeros(S)
        self.btc = np.zeros(S)
        self.ref = np.ones(S)
        self.growth = np.ones(S)          # growth banked at previous splits
        self.born_t = np.zeros(S)
        self.last_split_t = np.zeros(S)
        self.fees = np.zeros(S)
        self.trades = np.zeros(S, np.int64)
        self.children = np.zeros(S, np.int64)

        self.t = 0.0
        self.price = None
        self.next_id = 0
        self.injected = 0.0               # play money handed to immigrants
        self.withdrawn = 0.0              # equity of agents when they died
        self.bnh_units = None             # buy-and-hold benchmark
        self.bnh_start_t = None
        self.counts = {"births": 0, "immigrants": 0, "starved": 0, "outcompeted": 0}
        self.events = []                  # drained by the store
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
        live = self.alive & (~np.isin(self.tf, self.timeframes))
        for i in np.nonzero(live)[0]:     # e.g. a 30s agent moved onto a 1m feed
            self.tf[i] = min(self.timeframes, key=lambda x: abs(x - self.tf[i]))

    def warm(self, open_ms, dur_ms, o, h, l, c, v):
        """Feed history into candle books without trading (e.g. before going live)."""
        for tf, book in self.books.items():
            if (tf * 1000) % dur_ms == 0:
                book.add(open_ms, dur_ms, o, h, l, c, v)

    def step(self, open_ms, o, h, l, c, v):
        """Advance the world by one closed base bar."""
        dur_ms = self.base_seconds * 1000
        self.t = (open_ms + dur_ms) / 1000.0
        self.price = c
        if not self.seeded:
            self._seed()
        if self.bnh_units is None:
            self.bnh_units = self.cfg.initial_capital / c
            self.bnh_start_t = self.t

        tax = 1.0 - self.cfg.metabolism_bps_per_day * 1e-4 * self.base_seconds / DAY
        self.cash[self.alive] *= tax
        self.btc[self.alive] *= tax

        for tf, book in self.books.items():
            if book.add(open_ms, dur_ms, o, h, l, c, v) and book.ready:
                self._act(tf, book.features(), c)

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
        out = np.einsum("nh,nh->n", hid, W2) + b2
        if self.cfg.allow_short:
            return np.tanh(out)
        return 1.0 / (1.0 + np.exp(-out))

    def _act(self, tf, feats, price):
        idx = np.nonzero(self.alive & (self.tf == tf))[0]
        if len(idx) == 0:
            return
        eq = self.cash[idx] + self.btc[idx] * price
        x = self.btc[idx] * price / eq
        X = np.empty((len(idx), self.n_in))
        X[:, :-1] = feats
        X[:, -1] = x
        target = self._forward(idx, X)
        delta = target - x
        go = np.abs(delta) > self.deadband[idx]
        if not go.any():
            return
        idx, delta, eq = idx[go], delta[go], eq[go]
        fee = self.cfg.fee_bps * 1e-4
        slip = self.cfg.slippage_bps * 1e-4

        buy = delta > 0
        if buy.any():
            b = idx[buy]
            notional = np.minimum(delta[buy] * eq[buy], self.cash[b] / (1 + fee))
            self.cash[b] -= notional * (1 + fee)
            self.btc[b] += notional / (price * (1 + slip))
            self.fees[b] += notional * (fee + slip)
        sell = ~buy
        if sell.any():
            s = idx[sell]
            qty = -delta[sell] * eq[sell] / price
            if not self.cfg.allow_short:
                qty = np.minimum(qty, self.btc[s])
            proceeds = qty * price * (1 - slip)
            self.btc[s] -= qty
            self.cash[s] += proceeds * (1 - fee)
            self.fees[s] += qty * price * (fee + slip)
        self.trades[idx] += 1

    # ----------------------------------------------------------------- ecology
    def equity(self, idx=None):
        if idx is None:
            return self.cash + self.btc * self.price
        return self.cash[idx] + self.btc[idx] * self.price

    def _ecology(self):
        cfg = self.cfg
        wild = self.alive & ~self.control
        ratio = self.equity() / self.ref

        for i in np.nonzero(wild & (ratio < cfg.death_ratio))[0]:
            self._kill(i, "starved")

        min_age = cfg.min_repro_age_days * DAY
        ready = self.alive & ~self.control & (ratio >= cfg.repro_ratio)
        ready &= (self.t - self.last_split_t) >= min_age
        queue = sorted(np.nonzero(ready)[0], key=lambda j: -ratio[j])
        ids = {i: self.agent_id[i] for i in queue}
        for i in queue:
            if self.alive[i] and self.agent_id[i] == ids[i]:   # not displaced meanwhile
                self._reproduce(i)

        wild = self.alive & ~self.control
        for tf in self.timeframes:
            missing = cfg.min_per_niche - int((wild & (self.tf == tf)).sum())
            for _ in range(max(0, missing)):
                if self._spawn_random(tf) is None:
                    break

    def _free_slot(self):
        free = np.nonzero(~self.alive[: self.cfg.capacity])[0]
        return int(free[0]) if len(free) else None

    def _new_identity(self, i, parent=None):
        self.agent_id[i] = self.next_id
        self.next_id += 1
        self.alive[i] = True
        self.born_t[i] = self.last_split_t[i] = self.t
        self.growth[i] = 1.0
        self.fees[i] = 0.0
        self.trades[i] = 0
        self.children[i] = 0
        if parent is None:
            self.parent_id[i] = -1
            self.root_id[i] = self.agent_id[i]
            self.generation[i] = 0
        else:
            self.parent_id[i] = self.agent_id[parent]
            self.root_id[i] = self.root_id[parent]
            self.generation[i] = self.generation[parent] + 1

    def _random_genome(self, i, tf):
        ni, h = self.n_in, self.cfg.hidden
        p = np.zeros(self.n_params)
        p[: ni * h] = self.rng.normal(0, 1 / np.sqrt(ni), ni * h)
        p[ni * h + h: ni * h + 2 * h] = self.rng.normal(0, 1 / np.sqrt(h), h)
        p[-1] = self.rng.normal(0, 1)
        self.params[i] = p
        self.tf[i] = tf
        self.sigma[i] = self.cfg.init_sigma
        self.deadband[i] = np.exp(self.rng.uniform(np.log(0.02), np.log(0.3)))

    def _spawn_random(self, tf, slot=None):
        i = self._free_slot() if slot is None else slot
        if i is None:
            return None
        self._random_genome(i, tf)
        self._new_identity(i)
        cap = self.cfg.initial_capital
        self.cash[i], self.btc[i], self.ref[i] = cap, 0.0, cap
        if not self.control[i]:
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
        for arr in (self.cash, self.btc):
            arr[i] /= 2.0
            arr[slot] = arr[i]
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
        tf = int(self.tf[parent])
        if rng.random() < cfg.timeframe_mutation_prob:
            k = self.timeframes.index(tf) + rng.choice((-1, 1))
            tf = self.timeframes[int(np.clip(k, 0, len(self.timeframes) - 1))]
        self.tf[child] = tf

    def _kill(self, i, cause):
        eq = float(self.equity(i))
        self.withdrawn += eq
        self.counts[cause] += 1
        self.alive[i] = False
        self.events.append(("death", {
            "id": int(self.agent_id[i]), "t": self.t, "cause": cause,
            "growth": float(self.growth[i] * eq / self.ref[i]),
            "trades": int(self.trades[i]), "fees": float(self.fees[i]),
            "children": int(self.children[i]),
        }))

    # -------------------------------------------------------------- reporting
    def _record(self, i, kind):
        return {
            "id": int(self.agent_id[i]), "parent": int(self.parent_id[i]),
            "root": int(self.root_id[i]), "generation": int(self.generation[i]),
            "timeframe": int(self.tf[i]), "t": self.t, "kind": kind,
            "control": bool(self.control[i]), "deadband": float(self.deadband[i]),
            "sigma": float(self.sigma[i]), "genome": self.params[i].astype(np.float32).tobytes(),
        }

    def lifetime_growth(self):
        return self.growth * self.equity() / self.ref

    def daily_log_growth(self, mask):
        age = np.maximum((self.t - self.born_t[mask]) / DAY, 1e-9)
        return np.log(self.lifetime_growth()[mask]) / age

    def snapshot(self):
        wild = self.alive & ~self.control
        ctrl = self.alive & self.control
        eq = self.equity()
        mature = wild & ((self.t - self.born_t) >= DAY)
        bnh_days = max((self.t - self.bnh_start_t) / DAY, 1e-9)
        bnh_value = self.bnh_units * self.price
        niches = {}
        for tf in self.timeframes:
            m = wild & (self.tf == tf)
            niches[str(tf)] = {
                "n": int(m.sum()),
                "equity": float(eq[m].sum()),
                "median_ratio": float(np.median(eq[m] / self.ref[m])) if m.any() else None,
            }
        return {
            "t": self.t, "price": self.price,
            "population": int(wild.sum()),
            "equity": float(eq[wild].sum()),
            "injected": self.injected, "withdrawn": self.withdrawn,
            "net_pnl": float(eq[wild].sum()) + self.withdrawn - self.injected,
            "max_generation": int(self.generation[wild].max()) if wild.any() else 0,
            "mature_daily_log_growth": (float(np.median(self.daily_log_growth(mature)))
                                        if mature.any() else None),
            "control_daily_log_growth": (float(np.median(self.daily_log_growth(ctrl)))
                                         if ctrl.any() else None),
            "bnh_daily_log_growth": float(np.log(bnh_value / self.cfg.initial_capital) / bnh_days),
            "niches": niches,
            "counts": dict(self.counts),
        }

    def leaderboard(self, n=10, min_age_days=1.0):
        wild = self.alive & ~self.control & ((self.t - self.born_t) >= min_age_days * DAY)
        idx = np.nonzero(wild)[0]
        if len(idx) == 0:
            return []
        dlg = self.daily_log_growth(wild)
        order = np.argsort(-dlg)[:n]
        eq = self.equity()
        rows = []
        for k in order:
            i = idx[k]
            rows.append({
                "id": int(self.agent_id[i]), "root": int(self.root_id[i]),
                "generation": int(self.generation[i]), "timeframe": int(self.tf[i]),
                "age_days": (self.t - self.born_t[i]) / DAY,
                "lifetime_growth": float(self.lifetime_growth()[i]),
                "daily_log_growth": float(dlg[k]), "equity": float(eq[i]),
                "exposure": float(self.btc[i] * self.price / eq[i]),
                "trades": int(self.trades[i]), "children": int(self.children[i]),
                "deadband": float(self.deadband[i]),
            })
        return rows

    def reset_economy(self):
        """Give every living agent a fresh play wallet (e.g. when going live)."""
        cap = self.cfg.initial_capital
        live = self.alive
        self.cash[live], self.btc[live], self.ref[live] = cap, 0.0, cap
        self.growth[live] = 1.0
        self.fees[live] = 0.0
        self.trades[live] = 0
        self.born_t[live] = self.last_split_t[live] = self.t
        self.injected = cap * float((live & ~self.control).sum())
        self.withdrawn = 0.0
        self.bnh_units = None
        self._last_snapshot_t = None
