"""The ecosystem: play-money BTCUSDT perpetual traders that live or die by equity.

The world owns what is common to every kind of agent: one exchange account
each (``exchange.py``), an identity (the cadence it decides at, its leverage,
the share of equity it will put up as margin), the order flow (a target
exposure decided on a minute's close fills at the next minute's open) and the
ecology. Perception and decisions belong to a pluggable brain:

* ``brains.NetBrain``: a small fixed neural net per agent, evolved only;
* ``mind.RRBrain``: relevance-realizing agents (see docs/).

The ecology only ever reads equity. Falling below ``death_ratio`` of the
reference equity kills an agent. Reaching ``repro_ratio`` of it splits the
account exactly in half with a mutated child. When the population is full a
newborn takes the slot of the weakest agent, if that agent is doing worse
than the parent. Control agents never reproduce and are replaced when they
die; they show what the same machinery earns without selection.
"""

import numpy as np

from .data import C, MINUTE
from .exchange import Accounts

DAY = 86400.0


class World:
    def __init__(self, cfg, brain):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
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
        self.tf = np.zeros(S, np.int64)            # decision cadence, seconds
        self.lev = np.ones(S)                      # leverage setting (integer valued)
        self.frac = np.ones(S)                     # share of equity put up as margin
        self.max_notional = np.zeros(S)
        self.pending = np.full(S, np.nan)          # target exposure, filled next minute
        self.ref = np.ones(S)
        self.growth = np.ones(S)                   # growth banked at previous splits
        self.born_t = np.zeros(S)
        self.last_split_t = np.zeros(S)
        self.children = np.zeros(S, np.int64)
        self.timeframes = list(cfg.timeframes)

        self.t = 0.0
        self.price = None
        self.mark = None
        self.next_id = 0
        self.injected = 0.0
        self.withdrawn = 0.0
        self.control_injected = 0.0
        self.control_withdrawn = 0.0
        self.bnh_units = None
        self.bnh_start_t = None
        self.totals = {"fees": 0.0, "funding": 0.0, "liquidations": 0}
        self.counts = {"births": 0, "immigrants": 0, "starved": 0, "outcompeted": 0}
        self.events = []
        self._last_snapshot_t = None
        self.seeded = False
        self.brain = brain
        brain.attach(self)

    # ------------------------------------------------------------------ market
    def warm(self, row):
        """Perceive a minute without acting (e.g. history before going live)."""
        self.brain.observe(row, (row[C["open_time"]] + MINUTE) / 1000.0)

    def step(self, row):
        """Advance by one closed minute (a row of data.COLUMNS)."""
        self.t = (row[C["open_time"]] + MINUTE) / 1000.0
        o, c = row[C["open"]], row[C["close"]]
        if not self.seeded:
            self.mark = row[C["mark_close"]]
            self._seed()
        if self.bnh_units is None:
            self.bnh_units = self.cfg.initial_capital / o
            self.bnh_start_t = self.t

        self._execute(o)
        hit = self.acct.liquidate(row[C["mark_high"]], row[C["mark_low"]])
        self.totals["liquidations"] += int((~self.control[hit]).sum())
        funding = row[C["funding_rate"]]
        if funding and funding == funding:
            idx, pay = self.acct.settle_funding(funding, row[C["mark_close"]])
            self.totals["funding"] += float(pay[~self.control[idx]].sum())
        self.price, self.mark = c, row[C["mark_close"]]

        self.brain.observe(row, self.t)
        idx, x = self.brain.decide(self.t)
        if len(idx):
            self._target(idx, x)

        self._ecology()
        every = self.cfg.snapshot_every_s
        if self._last_snapshot_t is None or self.t - self._last_snapshot_t >= every:
            self._last_snapshot_t = self.t
            self.events.append(("snapshot", self.snapshot()))

    def exposure(self, idx):
        """Current signed exposure (position value / equity) of agents `idx`."""
        eq = np.maximum(self.acct.equity(self.mark, idx), 1e-9)
        return self.acct.q[idx] * self.mark / eq

    def max_exposure(self, idx):
        eq = np.maximum(self.acct.equity(self.mark, idx), 1e-9)
        return np.minimum(self.frac[idx] * self.lev[idx], self.max_notional[idx] / eq)

    def _target(self, idx, x):
        lim = self.max_exposure(idx)
        self.pending[idx] = np.clip(x, -lim, lim)

    def _execute(self, price):
        idx = np.nonzero(self.alive & ~np.isnan(self.pending))[0]
        if len(idx) == 0:
            return
        x = self.pending[idx]
        self.pending[idx] = np.nan
        eq = np.maximum(self.acct.equity(price, idx), 0.0)
        notional = np.clip(x * eq, -self.max_notional[idx], self.max_notional[idx])
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

    def _spawn_random(self, tf, slot=None):
        i = self._free_slot() if slot is None else slot
        if i is None:
            return None
        rng = self.rng
        self.tf[i] = tf
        self._set_leverage(i, np.exp(rng.uniform(0, np.log(self.cfg.max_leverage))))
        self.frac[i] = np.exp(rng.uniform(np.log(0.05), 0.0))
        self._new_identity(i)
        self.brain.new_genome(i)
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
        for k in range(self.cfg.min_population):
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

        rng = self.rng
        tf = int(self.tf[i])
        if rng.random() < self.cfg.timeframe_mutation_prob:
            k = self.timeframes.index(tf) + rng.choice((-1, 1))
            tf = self.timeframes[int(np.clip(k, 0, len(self.timeframes) - 1))]
        self.tf[slot] = tf
        self._set_leverage(slot, self.lev[i] * np.exp(0.25 * rng.normal()))
        self.frac[slot] = float(np.clip(self.frac[i] * np.exp(0.2 * rng.normal()), 0.02, 1.0))
        self._new_identity(slot, parent=i)
        self.brain.inherit(i, slot)
        self.growth[i] *= ratio[i]
        self.acct.split(i, slot)
        self.ref[i] = self.ref[slot] = self.equity(i)
        self.last_split_t[i] = self.t
        self.children[i] += 1
        self.counts["births"] += 1
        self.events.append(("birth", self._record(slot, "born")))

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
        self.brain.clear(i)
        self.alive[i] = False
        self.pending[i] = np.nan

    # -------------------------------------------------------------- reporting
    def _record(self, i, kind):
        return {
            "id": int(self.agent_id[i]), "parent": int(self.parent_id[i]),
            "root": int(self.root_id[i]), "generation": int(self.generation[i]),
            "timeframe": int(self.tf[i]), "t": self.t, "kind": kind,
            "control": bool(self.control[i]), "leverage": float(self.lev[i]),
            "margin_frac": float(self.frac[i]), "genome": self.brain.genome_bytes(i),
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
        return {
            "t": self.t, "price": self.price, "mark": self.mark,
            "population": int(wild.sum()),
            "long": int((wild & (q > 0)).sum()), "short": int((wild & (q < 0)).sum()),
            "equity": held, "injected": self.injected, "withdrawn": self.withdrawn,
            "net_pnl": held + self.withdrawn - self.injected,
            "control_net_pnl": float(eq[ctrl].sum()) + self.control_withdrawn - self.control_injected,
            "control_injected": self.control_injected,
            "bnh_return": self.bnh_units * self.mark / self.cfg.initial_capital - 1.0,
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
            "mind": self.brain.snapshot(),
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
                "mind": self.brain.describe(i),
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
