"""Life on the planet: very many minimal Vervaekean organisms (docs/planet.md, section 3).

Every organism is a few hundred numbers, kept as columns of arrays so that a
whole band of the planet thinks at once.

Body       one exact BTCUSDT perpetual account (exchange.py) and a place: a cell
           whose latitude is its timescale. Wealth is life: death below
           death_ratio of its reference wealth, division above repro_ratio.
Organs     K self-made programs over the senses of its place (market
           receptors that exist there, plus the place's observatories and
           markers). Each organ has an integrity that decays every tick and
           is restored only by being salient. Organs that stop mattering
           dissolve and are replaced. The organism keeps remaking its perception.
Values     what each of its nine crop intensities would have done to its own
           wealth, learned from the world's actual path run through its own
           account (recursive least squares, believed only on evidence).
Salience   how much its own values move when one of its own organs moves.
Dials      explore-exploit and efficiency-resiliency, each the balance of two
           opposed rates driven by the organism's own signals.
Modes      habitual (every tick: act on values, keep organs) and higher-order
           (opened by a learned gate when surprise, stagnation, conflict or
           novelty are high): rebuild organs deliberately, break frames,
           migrate, build an observatory. Both modes act on the organism's
           values: it learns what all nine moves would have done on every tick,
           so a random move would teach it nothing and only pay fees. The gate
           is an opponent process: every episode costs a little (it raises the
           bar), and episodes that improved grip lower it.
"""

import numpy as np

from .data import C
from .exchange import Accounts
from .planet import CULTURE, GROWTH, N_CH, N_OBS, _normalize
from .selfmade import ACTIONS, LAGS, MMR, N_ACT, N_REC, NORMS, OPS, RECEPTORS, ZERO

K = 6                                         # organs per organism
D = 3 * K + 1                                 # features of a perceived situation
Q = 10                                        # crops growing at once
R = 16                                        # recent situations kept for re-landscaping
N_GATE = 4                                    # signals the gate reads
N_BODY = 6                                    # interoceptive features
FIELDS = (N_CH, N_CH, len(OPS), len(LAGS), len(NORMS))
OFFSETS = np.cumsum((0,) + FIELDS[:-1])
D_KEY = 8                                     # width of anticipation's queries and keys
YOUTH = 64                                    # band ticks a new organ is spared for being unsalient
GATE_COST = 0.01                              # what each conscious episode raises the gate's bar by


def describe(p):
    a, b, op, k, tau = (int(v) for v in p)
    name = lambda c: RECEPTORS[c] if c < ZERO else ("" if c == ZERO else
                                                     ("observatory%d" % (c - ZERO - 1) if c < GROWTH
                                                      else "harvest-marker" if c == GROWTH
                                                      else "crowding-marker"))
    series = name(a) if b == ZERO else f"{name(a)} - {name(b)}"
    inner = series if op == 0 else f"{OPS[op]}{LAGS[k]}({series})"
    return f"z{NORMS[tau]}({inner})"


class Life:
    # ablations (class defaults, so worlds saved before they existed load unchanged)
    higher_order = True        # False: the gate never opens; habits and blind generate-and-test only
    culture = True             # False: no observatories, no markers to perceive or follow

    def __init__(self, cfg, planet, capacity=8192, seed=0, min_population=None,
                 higher_order=True, culture=True):
        self.higher_order, self.culture = higher_order, culture
        self.cfg, self.planet = cfg, planet
        self.N = N = capacity
        self.rng = np.random.default_rng(seed)
        self.min_population = min_population or planet.X * planet.Y * 2
        self.acct = Accounts(N, cfg)
        z = lambda *s: np.zeros((N,) + s)
        self.alive = np.zeros(N, bool)
        self.oid = np.full(N, -1, np.int64)
        self.root = np.full(N, -1, np.int64)
        self.gen = np.zeros(N, np.int64)
        self.x = np.zeros(N, np.int64)
        self.y = np.zeros(N, np.int64)
        self.born = z()
        self.last_split = z()
        self.ref = np.ones(N)
        self.growth = np.ones(N)
        self.lev = np.ones(N)
        self.frac = np.ones(N)
        self.max_notional = z()
        self.pending = np.full(N, np.nan)
        # genes
        self.hmult = np.ones(N, np.int64)
        self.memory = np.ones(N)
        self.p_mutate = np.full(N, 0.5)
        self.p_build = z()
        self.gate_w = z(N_GATE)
        self.gate_theta = z()
        # organs
        self.prog = np.zeros((N, K, 5), np.int64)
        self.pstate = z(K, 4)
        self.value = np.full((N, K), np.nan)
        self.integrity = np.ones((N, K))
        self.age = np.zeros((N, K), np.int64)
        # values
        self.W = z(D, N_ACT)
        self.P = z(D, D)
        self.s2 = np.ones((N, N_ACT))
        self.nupd = np.zeros(N, np.int64)
        self.q_phi = z(Q, D)
        self.q_due = np.full((N, Q), np.iinfo(np.int64).max)
        self.q_val = z(Q, 6)                       # tick, price, xmax, leverage, action, predicted
        self.q_head = np.zeros(N, np.int64)
        self.q_len = np.zeros(N, np.int64)
        self.recent = z(R, K)
        self.recent_n = np.zeros(N, np.int64)
        self.salience = np.full((N, K), 1.0 / K)
        # signals, dials, modes
        self.grip = z()
        self.err_fast = np.ones(N)
        self.err_slow = np.ones(N)
        self.turbulence = z()
        self.conflict = z()
        self.explore = np.full(N, 0.3)
        self.resilience = np.full(N, 0.5)
        self.refractory = np.zeros(N, np.int64)
        self.episode_grip = np.full(N, np.nan)
        self.episode_due = np.full(N, -1, np.int64)
        self.drift = z()                           # how grip moves without episodes
        self.decisions = np.zeros(N, np.int64)
        self.episodes = np.zeros(N, np.int64)
        # anticipation: each organism's own query-key model of which organs will matter to it
        self.a_E = np.zeros((N, sum(FIELDS), D_KEY))
        self.a_Wq = np.zeros((N, N_BODY, D_KEY))
        self.a_n = np.zeros(N, np.int64)
        # bookkeeping
        self.next_id = 0
        self.injected = self.withdrawn = 0.0
        self.fees_dead = self.funding_dead = 0.0              # paid by organisms now dead
        self.liq_dead = 0
        self.counts = {"born": 0, "seeded": 0, "died": 0, "displaced": 0, "migrated": 0,
                       "built": 0, "episodes": 0, "organs_made": 0, "breakdowns": 0}
        self._senses = None
        # moving is a slow affair in planetary time: a 1-second organism gets as many
        # chances to move per hour as a 10-minute one, not 600 times as many
        self.roam = np.minimum(1.0, planet.taus / 600.0)
        self.t = 0.0
        self.price = self.mark = self.price0 = np.nan

    # ------------------------------------------------------------- one second
    def step(self, row, done):
        """One second: fills, liquidations, funding, then every band that ticked."""
        self.t = self.planet.t
        if not self.alive.any():
            self.mark = row[C["close"]]
            if self.price0 != self.price0:
                self.price0 = row[C["open"]]
            self._seed(self.min_population)
        self._execute(row[C["open"]])
        hi, lo = row[C["mark_high"]], row[C["mark_low"]]
        if not (hi > 0 and lo > 0):
            hi, lo = row[C["high"]], row[C["low"]]
        self.acct.liquidate(hi, lo)
        f = row[C["funding_rate"]]
        if f and f == f:
            self.acct.settle_funding(f, row[C["mark_close"]] if row[C["mark_close"]] > 0 else row[C["close"]])
        self.price = row[C["close"]]
        self.mark = row[C["mark_close"]] if row[C["mark_close"]] > 0 else self.price
        for y in done:
            self._band_tick(int(y))
        if int(round(self.t)) % 10 == 0:
            self._ecology()

    def equity(self, idx=slice(None)):
        return self.acct.equity(self.mark, idx)

    def exposure(self, idx):
        eq = np.maximum(self.equity(idx), 1e-9)
        return self.acct.q[idx] * self.mark / eq

    def max_exposure(self, idx):
        eq = np.maximum(self.equity(idx), 1e-9)
        return np.minimum(self.frac[idx] * self.lev[idx], self.max_notional[idx] / eq)

    def _execute(self, price):
        idx = np.nonzero(self.alive & ~np.isnan(self.pending))[0]
        if len(idx) == 0:
            return
        x = self.pending[idx]
        self.pending[idx] = np.nan
        eq = np.maximum(self.acct.equity(price, idx), 0.0)
        notional = np.clip(x * eq, -self.max_notional[idx], self.max_notional[idx])
        self.acct.execute(idx, notional / price, price, self.lev[idx])

    # ------------------------------------------------------------- a band ticks
    def _band_tick(self, y):
        idx = np.nonzero(self.alive & (self.y == y))[0]
        if len(idx) == 0:
            return
        pl = self.planet
        tick = pl.ticks[y]
        self._sense(idx, y)
        self._mature(idx, y, tick)
        v = np.nan_to_num(self.value[idx])
        phi = _phi(v)
        Wf = self._shrunk(idx)
        Qv = (phi[:, None, :] @ Wf)[:, 0]
        xmax, lam = self.max_exposure(idx), self.exposure(idx)
        cost = self.cfg.taker_fee + self.cfg.half_spread / self.price
        score, x_all, stay = _scores(Qv, xmax, lam, cost)
        sd = np.sqrt(np.maximum(self.s2[idx], 1e-12)).mean(1)
        top2 = np.sort(score, axis=1)[:, -2:]
        self.conflict[idx] += (np.exp(-(top2[:, 1] - top2[:, 0]) / sd) - self.conflict[idx]) / 8
        conscious = self._gate(idx, tick)
        act = np.argmax(score, axis=1)
        best = score[np.arange(len(idx)), act]
        keep = stay >= best                                  # move only when it is worth its cost
        chosen = np.where(keep, _nearest(lam, xmax), act)
        move = ~keep
        self.pending[idx[move]] = x_all[move, act[move]]
        self._queue(idx, phi, tick, xmax, chosen, Qv[np.arange(len(idx)), chosen])
        pos = self.decisions[idx] % R
        self.recent[idx, pos] = v
        self.recent_n[idx] = np.minimum(self.recent_n[idx] + 1, R)
        self.decisions[idx] += 1
        self._maintain(idx, y)
        if conscious.any():
            self._reorganize(idx[conscious], y)

    def _channels(self, idx, y, ch, cur):
        """Values of channels `ch` (n, K) for organisms idx at their places."""
        pl = self.planet
        xi = self.x[idx][:, None]
        market = np.where(pl.avail[xi, y, np.minimum(ch, ZERO)], cur[np.minimum(ch, ZERO)], np.nan)
        culture = pl.culture(self.x[idx], np.full(len(idx), y))            # (n, CULTURE)
        cult = np.take_along_axis(culture, np.clip(ch - ZERO - 1, 0, CULTURE - 1), axis=1)
        return np.where(ch <= ZERO, market, cult)

    def _sense(self, idx, y):
        pl = self.planet
        H, tick = pl.H, pl.ticks[y]
        p = self.prog[idx]
        a, b, op = p[..., 0], p[..., 1], p[..., 2]
        k, tau = LAGS[p[..., 3]], NORMS[p[..., 4]]
        cur = pl.rbuf[y, (tick - 1) % H]
        x = self._channels(idx, y, a, cur) - self._channels(idx, y, b, cur)
        past_row = pl.rbuf[y][(tick - 1 - k) % H]                          # (n, K, channels)
        xi = self.x[idx][:, None]
        am, bm = np.minimum(a, ZERO), np.minimum(b, ZERO)
        past = (np.take_along_axis(past_row, am[..., None], -1)[..., 0]
                - np.take_along_axis(past_row, bm[..., None], -1)[..., 0])
        ok_past = ((tick - 1 >= k) & (k < H) & (a <= ZERO) & (b <= ZERO)
                   & pl.avail[xi, y, am] & pl.avail[xi, y, bm])
        change = x - np.where(ok_past, past, np.nan)
        val, st = _normalize(x, change, op, k, tau, self.pstate[idx])
        self.pstate[idx] = st
        self.value[idx] = val

    # ------------------------------------------------- learning from harvests
    def _mature(self, idx, y, tick):
        """Harvest every ripe crop: the world's actual path run through the organism's
        own account, at each of its nine intensities (full counterfactual experience)."""
        pl = self.planet
        for _ in range(Q):
            live = idx[self.q_len[idx] > 0]
            if len(live) == 0:
                return
            h = self.q_head[live]
            ripe = self.q_due[live, h] <= tick
            live, h = live[ripe], h[ripe]
            if len(live) == 0:
                return
            m0, p0, xmax, lev, chosen, pred = self.q_val[live, h].T
            starts, inv = np.unique(m0, return_inverse=True)  # crops planted together share a path
            path = np.empty((len(starts), 4))
            for j, start in enumerate(starts):
                seg = pl.pbuf[y, np.arange(int(start) + 1, tick) % pl.H]
                path[j] = (seg[-1, 0], np.fmin.reduce(seg[:, 1]), np.fmax.reduce(seg[:, 2]),
                           np.nansum(seg[:, 3]))
            Y = _harvest(ACTIONS * xmax[:, None], lev, p0, *path[inv.ravel()].T)
            c = chosen.astype(np.int64)
            yc = Y[np.arange(len(live)), c]
            s2c = np.maximum(self.s2[live, c], 1e-12)
            err = np.abs(yc - pred) / np.sqrt(s2c)
            self.err_fast[live] += (err - self.err_fast[live]) / 8
            self.err_slow[live] += (err - self.err_slow[live]) / 128
            gain = (yc * yc - (yc - pred) ** 2) / s2c
            old = self.grip[live].copy()
            self.grip[live] += (np.clip(gain, -5, 5) - self.grip[live]) / 32
            self.turbulence[live] += (np.abs(self.grip[live] - old) * 32 - self.turbulence[live]) / 32
            self.drift[live] += ((self.grip[live] - old) - self.drift[live]) / 64
            self._mark_growth(self.x[live], y, yc * 86400.0 / ((tick - 1 - m0) * self.planet.taus[y]))
            self._rls(live, self.q_phi[live, h], Y)
            self.q_head[live] = (h + 1) % Q
            self.q_len[live] -= 1

    def _mark_growth(self, xs, y, rate):
        """Leave a trace of how harvests went at each place (log growth per day)."""
        pl = self.planet
        n = np.bincount(xs, minlength=pl.X)
        hit = n > 0
        mean = np.bincount(xs, weights=rate, minlength=pl.X)[hit] / n[hit]
        w = 1.0 - (1.0 - 1.0 / 16) ** n[hit]
        pl.growth[hit, y] += w * (mean - pl.growth[hit, y])

    def _rls(self, idx, phi, Y):
        """Recursive least squares with forgetting, one step for each organism."""
        lam = 1.0 - 1.0 / self.memory[idx]
        P, W = self.P[idx], self.W[idx]
        Pphi = (P @ phi[:, :, None])[:, :, 0]
        k = Pphi / (lam + np.maximum((phi * Pphi).sum(1), 0.0))[:, None]
        E = Y - (phi[:, None, :] @ W)[:, 0]
        W += k[:, :, None] * E[:, None, :]
        P -= k[:, :, None] * Pphi[:, None, :]
        P = 0.5 * (P + P.transpose(0, 2, 1))              # rounding must not unbalance it over
        P /= lam[:, None, None]                           # hundreds of thousands of updates
        P *= np.minimum(1.0, 4.0 * D * 0.05 / np.maximum(np.trace(P, axis1=1, axis2=2), 1e-12))[:, None, None]
        self.P[idx], self.W[idx] = P, W
        rate = np.maximum(1.0 / (self.nupd[idx] + 2.0), 1.0 / 500)[:, None]
        self.s2[idx] += rate * (E * E - self.s2[idx])
        self.nupd[idx] += 1
        broken = ~(np.isfinite(W).all((1, 2)) & np.isfinite(P).all((1, 2)) & np.isfinite(self.s2[idx]).all(1))
        if broken.any():
            self._recover(idx[broken])

    def _recover(self, idx):
        """A mind that broke down numerically starts its learning over; body and organs remain."""
        self.counts["breakdowns"] = self.counts.get("breakdowns", 0) + len(idx)
        self.W[idx] = 0.0
        self.P[idx] = np.eye(D) * 0.05
        self.s2[idx] = 1e-4
        self.nupd[idx] = 0
        self.q_len[idx] = 0
        self.grip[idx] = self.turbulence[idx] = self.conflict[idx] = self.drift[idx] = 0.0
        self.err_fast[idx] = self.err_slow[idx] = 1.0
        self.explore[idx], self.resilience[idx] = 0.3, 0.5
        self.salience[idx] = 1.0 / K
        self.integrity[idx] = np.where(np.isfinite(self.integrity[idx]), self.integrity[idx], 1.0)
        for arr in (self.a_E, self.a_Wq):
            bad = idx[~np.isfinite(arr[idx]).reshape(len(idx), -1).all(1)]
            arr[bad] = self.rng.normal(0, 0.1, (len(bad),) + arr.shape[1:])

    def _shrunk(self, idx):
        """Values believed only as far as the evidence goes: w max(0, 1 - Var(w) / w^2)."""
        d = np.arange(D)
        var_w = self.s2[idx][:, None, :] * self.P[idx[:, None], d, d][:, :, None]
        W = self.W[idx]
        return W * np.maximum(0.0, 1.0 - var_w / (W * W + 1e-30))

    def _queue(self, idx, phi, tick, xmax, chosen, pred):
        slot = (self.q_head[idx] + self.q_len[idx]) % Q
        room = self.q_len[idx] < Q
        i, s = idx[room], slot[room]
        self.q_phi[i, s] = phi[room]
        self.q_due[i, s] = tick + self.hmult[i]
        self.q_val[i, s] = np.stack([np.full(len(i), tick - 1.0), np.full(len(i), self.price),
                                     xmax[room], self.lev[i], chosen[room], pred[room]], 1)
        self.q_len[i] += 1

    # ------------------------------------------------ keeping the organs alive
    def _maintain(self, idx, y):
        """Habitual upkeep. Salient organs are restored and the rest wear away; a worn-out
        organ dissolves and a blind variant takes its place (generate and test). How
        hard the organism prunes is set by its efficiency-resiliency dial."""
        due = idx[(self.decisions[idx] % 8 == 0) & (self.recent_n[idx] >= R // 2)]
        if len(due):
            tr = self.perspective(due)
            share = self.salience[due] * K
            I = self.integrity[due]
            step = 0.1 * (np.minimum(share, 1.0) - I)
            grown = self.age[due] >= YOUTH                  # the young are not judged yet
            step = np.where(grown | (step > 0), step, 0.0) * (tr > 0)[:, None]
            self.integrity[due] = np.clip(I + step, 0.0, 1.0)
            self._repair_observatories(due, y, share)
            self._dials(due)
        r = self.resilience[idx]
        self.integrity[idx] -= (0.002 * (1.5 - r))[:, None]
        rows, slots = np.nonzero(self.integrity[idx] < (0.5 * (1.0 - r))[:, None])
        if len(rows):
            for i, slot, prog in zip(idx[rows], slots, self._blind_programs(idx[rows])):
                self._replace(i, slot, prog)
        self.age[idx] += 1

    def perspective(self, idx, delta=1.0):
        """Salience: how much what the world offers the organism moves when one of its
        own organs moves, on its own values at its own recent situations (central
        differences, averaged as an expected gradient outer product). Returns the
        total sensitivity; zero means nothing it perceives matters to it yet."""
        n = len(idx)
        O = self.recent[idx]                                         # (n, R, K)
        mask = (np.arange(R)[None, :] < self.recent_n[idx][:, None]).astype(float)
        Wf = self._shrunk(idx)
        base = _phi(O) @ Wf                                          # (n, R, A)
        Wk = Wf[:, :3 * K].reshape(n, 3, K, N_ACT).transpose(0, 2, 1, 3)   # (n, K, 3, A)
        relu = lambda u: np.maximum(u, 0.0)
        cost = self.cfg.taker_fee + self.cfg.half_spread / self.price
        xmax, lam = self.max_exposure(idx), self.exposure(idx)
        V = []
        for sign in (1.0, -1.0):                                     # moving one organ moves only
            v1 = O + sign * delta                                    # its own three features
            d = np.stack([v1 - O, relu(v1 - 1) - relu(O - 1), relu(-v1 - 1) - relu(-O - 1)], -1)
            Qv = base[:, :, None, :] + (d.transpose(0, 2, 1, 3) @ Wk).transpose(0, 2, 1, 3)
            V.append(_best_value(Qv, xmax, lam, cost))
        dV = (V[0] - V[1]) / (2 * delta) * mask[..., None]           # (n, R, K)
        diag = (dV * dV).sum(1) / np.maximum(mask.sum(1), 1)[:, None]
        tr = diag.sum(1)
        self.salience[idx] = np.where(tr[:, None] > 0, diag / np.maximum(tr, 1e-300)[:, None], 1.0 / K)
        seasoned = (self.age[idx] > 16) & (tr[:, None] > 0)
        if seasoned.any():
            rows, slots = np.nonzero(seasoned)
            org = idx[rows]
            self._learn_anticipation(org, self.body_state(org), self.prog[org, slots],
                                     np.log(K * self.salience[org, slots] + 0.05))
        return tr

    # ------------------------------------------------ anticipation (its own)
    def _anticipate(self, i, body, tokens):
        """Organism i's anticipated log-salience of candidate organs: a query from its
        body, keys from the organs' parts, scored by dot-product attention."""
        k = self.a_E[i][tokens + OFFSETS].sum(1)
        return k @ (body @ self.a_Wq[i]) / np.sqrt(D_KEY)

    def _anticipate_many(self, orgs, tokens):
        """Each organism's anticipated log-salience of its own candidates (m, n, 5)."""
        k = self.a_E[orgs[:, None, None], tokens + OFFSETS].sum(2)          # (m, n, d)
        q = (self.body_state(orgs)[:, None, :] @ self.a_Wq[orgs])           # (m, 1, d)
        return (k @ q.transpose(0, 2, 1))[..., 0] / np.sqrt(D_KEY)

    def _learn_anticipation(self, org, body, tokens, target, lr=0.03, decay=1e-4):
        """One gradient step of each organism's anticipation toward the salience it measured."""
        rows = tokens + OFFSETS
        k = self.a_E[org[:, None], rows].sum(1)
        q = (body[:, None, :] @ self.a_Wq[org])[:, 0]
        per = np.bincount(org, minlength=self.N)[org]
        g = (((q * k).sum(1) / np.sqrt(D_KEY) - target) / (np.sqrt(D_KEY) * per))[:, None]
        u = np.unique(org)
        self.a_Wq[u] *= 1 - lr * decay
        self.a_E[u] *= 1 - lr * decay
        _scatter_add(self.a_Wq.reshape(self.N, -1), org,
                     (-lr * body[:, :, None] * (g * k)[:, None, :]).reshape(len(org), -1))
        F = self.a_E.shape[1]
        _scatter_add(self.a_E.reshape(self.N * F, D_KEY), (org[:, None] * F + rows).ravel(),
                     np.repeat(-lr * g * q, rows.shape[1], axis=0))
        self.a_n[u] += 1

    def _repair_observatories(self, idx, y, share):
        """Organisms keep up the observatories they find salient."""
        pl = self.planet
        for j in range(N_OBS):
            ch = ZERO + 1 + j
            reads = (self.prog[idx, :, 0] == ch) | (self.prog[idx, :, 1] == ch)
            gain = (share * reads).sum(1) * 0.05
            col = pl.obs_integrity[:, y, j]
            np.add.at(col, self.x[idx], gain * (col[self.x[idx]] > 0))
            np.minimum(col, 1.0, out=col)

    def _dials(self, idx):
        """Opponent processing: each dial balances two opposed pressures."""
        surprise = np.maximum(self.err_fast[idx] / np.maximum(self.err_slow[idx], 1e-6) - 1, 0)
        stall = np.maximum(-self.drift[idx] * 64, 0)
        settled = np.maximum(self.grip[idx], 0) + np.maximum(self.drift[idx] * 64, 0)
        up, down = surprise + stall + 0.05, settled + 0.05
        self.explore[idx] += 0.1 * (up * (1 - self.explore[idx]) - down * self.explore[idx])
        turb = self.turbulence[idx]
        self.resilience[idx] += 0.1 * ((turb + 0.05) * (1 - self.resilience[idx]) - 0.3 * self.resilience[idx])
        np.clip(self.explore, 0, 1, out=self.explore)
        np.clip(self.resilience, 0, 1, out=self.resilience)

    # ------------------------------------------------------ the higher order
    def _gate(self, idx, tick):
        """Which organisms open the higher-order mode now; also learns the gate."""
        s = np.stack([self.err_fast[idx] / np.maximum(self.err_slow[idx], 1e-6) - 1,
                      np.clip(-self.drift[idx] * 64, -1, 3), self.conflict[idx],
                      np.nanmean(np.abs(np.nan_to_num(self.value[idx])), 1) - 1], 1)
        g = 1 / (1 + np.exp(-((s * self.gate_w[idx]).sum(1) - self.gate_theta[idx])))
        due = self.episode_due[idx] == tick
        if due.any():                                       # did the last episode pay?
            d = idx[due]
            paid = (self.grip[d] - self.episode_grip[d]) > self.drift[d] * 16
            self.gate_theta[d] += np.where(paid, -0.05, 0.05)
            self.episode_due[d] = -1
        self.refractory[idx] = np.maximum(self.refractory[idx] - 1, 0)
        open_ = (g > 0.5) & (self.refractory[idx] == 0) & (self.nupd[idx] > 16) & self.higher_order
        o = idx[open_]
        self.gate_theta[o] += GATE_COST                     # consciousness is costly
        self.refractory[o] = 8
        self.episode_grip[o] = self.grip[o]
        self.episode_due[o] = tick + 16
        self.episodes[o] += 1
        self.counts["episodes"] += len(o)
        return open_

    def _reorganize(self, idx, y):
        """What a conscious moment can do: rebuild organs deliberately, migrate, build."""
        rng = self.rng
        n_new = 1 + np.round(2 * self.explore[idx]).astype(np.int64)   # more when exploring
        weakest = np.argsort(self.integrity[idx] * self.salience[idx], 1)
        rows, ranks = np.nonzero(np.arange(K)[None, :] < n_new[:, None])
        orgs, slots = idx[rows], weakest[rows, ranks]
        progs, novel = self._constructed_programs(orgs)
        for i, slot, prog in zip(orgs[novel], slots[novel], progs[novel]):
            if not (self.prog[i] == prog).all(1).any():
                self._replace(i, slot, prog)
        move = idx[rng.random(len(idx)) < (0.3 * self.explore[idx] + 0.1) * self.roam[y]]
        build = idx[rng.random(len(idx)) < self.p_build[idx]]
        if len(move):
            dens, cap = self._density(), self.planet.capacity()
            for i in move:
                self._migrate(i, dens, cap)
        for i in build:
            self._build(i)

    def _migrate(self, i, dens, cap):
        """Move to a neighboring place with room: where harvests have gone better than
        here or, when exploring, anywhere."""
        pl, rng = self.planet, self.rng
        x, y = int(self.x[i]), int(self.y[i])
        cells = [((x + 1) % pl.X, y), ((x - 1) % pl.X, y)]
        cells += [(x, y + 1)] if y + 1 < pl.Y else []
        cells += [(x, y - 1)] if y > 0 else []
        cells = [c for c in cells if dens[c] < cap[c]]
        if not cells:
            return
        if rng.random() < self.explore[i] or not self.culture:   # without markers, a blind move
            cx, cy = cells[rng.integers(len(cells))]
        else:
            g = np.array([pl.growth[c] for c in cells])
            best = int(np.argmax(g))
            if g[best] <= pl.growth[x, y]:
                return
            cx, cy = cells[best]
        dens[x, y] -= 1
        dens[cx, cy] += 1
        self.counts["migrated"] += 1
        self.x[i] = cx
        if cy != y:                                         # a new climate: perception starts over
            self.y[i] = cy
            self.pstate[i] = 0.0
            self.value[i] = np.nan
            self.q_len[i] = 0
            self.recent_n[i] = 0

    def _build(self, i):
        """Externalize its most salient organ into its place, as an observatory that
        everyone living there can perceive. Sometimes a telescope: the same program
        aimed at a stream chosen at random, which the place may not sense itself."""
        rng, pl = self.rng, self.planet
        top = int(np.argmax(self.salience[i]))
        if not self.culture or self.salience[i, top] <= 1.5 / K:
            return
        prog = self.prog[i, top].copy()
        if rng.random() < 0.3:
            prog[0] = rng.integers(N_REC)
            if prog[1] > ZERO:
                prog[1] = ZERO
        if prog[0] > ZERO or prog[1] > ZERO:                # culture is not re-externalized
            return
        if pl.build(self.x[i], self.y[i], prog, self.oid[i]):
            self.counts["built"] += 1

    # ------------------------------------------------------------- organs
    def _sense_table(self):
        """Every channel an organ can read at each place, padded (X, Y, S), and how many."""
        if self._senses is None:                           # geography is fixed at the planet's birth
            pl = self.planet
            culture = ZERO + 1 + np.arange(CULTURE if self.culture else 0)
            lists = [[np.concatenate([np.nonzero(pl.avail[x, y, :ZERO])[0], culture])
                      for y in range(pl.Y)] for x in range(pl.X)]
            table = np.full((pl.X, pl.Y, max(len(c) for r in lists for c in r)), ZERO, np.int64)
            count = np.zeros((pl.X, pl.Y), np.int64)
            for x in range(pl.X):
                for y in range(pl.Y):
                    table[x, y, :len(lists[x][y])] = lists[x][y]
                    count[x, y] = len(lists[x][y])
            self._senses = (table, count)
        return self._senses

    def _draw_senses(self, orgs, n):
        """(len(orgs), n) channels drawn at random from what each organism's place offers."""
        table, count = self._sense_table()
        x, y = self.x[orgs][:, None], self.y[orgs][:, None]
        return table[x, y, (self.rng.random((len(orgs), n)) * count[x, y]).astype(np.int64)]

    def _random_programs(self, orgs, n):
        """n blind variants for each organism: random programs over the senses of its place."""
        rng = self.rng
        m = len(orgs)
        a = self._draw_senses(orgs, n)
        b = np.where(rng.random((m, n)) < 0.5, ZERO, self._draw_senses(orgs, n))
        b[b == a] = ZERO
        op = rng.integers(len(OPS), size=(m, n))
        no_past = ((a > ZERO) | (b > ZERO)) & ((op == 1) | (op == 3))  # culture has no history
        op[no_past] = 2 * rng.integers(2, size=int(no_past.sum()))
        return np.stack([a, b, op, rng.integers(len(LAGS), size=(m, n)),
                         rng.integers(len(NORMS), size=(m, n))], -1)

    def _mutants(self, orgs, parents):
        """One small change to each parent (len(orgs), n, 5): a part swapped or a time
        constant nudged."""
        rng = self.rng
        p = np.array(parents, dtype=np.int64)
        m, n = p.shape[:2]
        f = rng.integers(5, size=(m, n))
        new_a = self._draw_senses(orgs, n)
        new_b = np.where(rng.random((m, n)) < 0.3, ZERO, self._draw_senses(orgs, n))
        step = 2 * rng.integers(2, size=(m, n)) - 1
        p[..., 0] = np.where(f == 0, new_a, p[..., 0])
        p[..., 1] = np.where(f == 1, new_b, p[..., 1])
        p[..., 2] = np.where(f == 2, rng.integers(len(OPS), size=(m, n)), p[..., 2])
        p[..., 3] = np.where(f == 3, np.clip(p[..., 3] + step, 0, len(LAGS) - 1), p[..., 3])
        p[..., 4] = np.where(f == 4, np.clip(p[..., 4] + step, 0, len(NORMS) - 1), p[..., 4])
        p[..., 1] = np.where(p[..., 1] == p[..., 0], ZERO, p[..., 1])
        culture = (p[..., 0] > ZERO) | (p[..., 1] > ZERO)
        p[..., 2] = np.where(culture & ((p[..., 2] == 1) | (p[..., 2] == 3)), 2, p[..., 2])
        return p

    def _random_program(self, i):
        return self._random_programs(np.array([i]), 1)[0, 0]

    def _mutate(self, i, p):
        return self._mutants(np.array([i]), np.asarray(p)[None, None])[0, 0]

    def _novel(self, orgs, pool):
        """Which candidates (m, n, 5) are not already organs of their organism."""
        return ~(pool[:, :, None, :] == self.prog[orgs][:, None, :, :]).all(-1).any(-1)

    def _blind_programs(self, orgs):
        """Habitual replacement: one random new organ for each entry of `orgs`."""
        pool = self._random_programs(orgs, 4)
        pick = np.argmax(self._novel(orgs, pool), 1)
        return pool[np.arange(len(orgs)), pick]

    def _constructed_programs(self, orgs):
        """Deliberate construction, one new organ for each entry of `orgs`: variants of
        what matters to it (with probability p_mutate) or new ideas, weighed by its own
        anticipation of what will matter once it has learned some. Returns the
        programs and which of them are new to their organism."""
        rng = self.rng
        m = len(orgs)
        ranked = np.argsort(-self.salience[orgs], 1)[:, :2]
        top = self.prog[orgs[:, None], ranked]                                  # (m, 2, 5)
        parents = top[np.arange(m)[:, None], rng.integers(2, size=(m, 16))]
        pool = np.concatenate([self._random_programs(orgs, 8), self._mutants(orgs, parents)], 1)
        pm = self.p_mutate[orgs][:, None]
        p = np.concatenate([np.broadcast_to((1 - pm) / 8, (m, 8)), np.broadcast_to(pm / 16, (m, 16))], 1)
        seasoned = self.a_n[orgs] > 50
        if seasoned.any():
            z = self._anticipate_many(orgs[seasoned], pool[seasoned])
            a = np.exp(z - z.max(1, keepdims=True))
            p[seasoned] *= 0.8 * a / a.sum(1, keepdims=True) + 0.2 / pool.shape[1]
        novel = self._novel(orgs, pool)
        p = np.where(novel, p, 0.0)
        p /= np.maximum(p.sum(1, keepdims=True), 1e-300)
        pick = np.minimum((np.cumsum(p, 1) < rng.random(m)[:, None]).sum(1), pool.shape[1] - 1)
        return pool[np.arange(m), pick], novel[np.arange(m), pick]

    def _replace(self, i, slot, program):
        """A new organ in `slot`: it starts cold, and what was learned through the old one is let go."""
        self.counts["organs_made"] += 1
        f = [slot, K + slot, 2 * K + slot]
        self.prog[i, slot] = program
        self.pstate[i, slot] = 0.0
        self.value[i, slot] = np.nan
        self.integrity[i, slot] = 1.0
        self.age[i, slot] = 0
        self.P[i][f, :] = 0.0
        self.P[i][:, f] = 0.0
        self.P[i][f, f] = 0.05
        self.W[i][f, :] = 0.0
        self.q_phi[i][:, f] = 0.0
        self.recent[i][:, slot] = 0.0
        self.salience[i, slot] = 1.0 / K

    def body_state(self, idx):
        x = self.exposure(idx)
        return np.stack([np.ones(len(idx)), np.tanh(x / 3), np.tanh(np.abs(x) / 3),
                         np.log(self.lev[idx]) / 5, self.y[idx] / self.planet.Y,
                         np.clip(self.grip[idx], -1, 1)], 1)

    # ------------------------------------------------------------- ecology
    def _density(self):
        pl = self.planet
        live = np.nonzero(self.alive)[0]
        d = np.zeros((pl.X, pl.Y), np.int64)
        np.add.at(d, (self.x[live], self.y[live]), 1)
        return d

    def _ecology(self):
        cfg, pl = self.cfg, self.planet
        ratio = self.equity() / self.ref
        for i in np.nonzero(self.alive & (ratio < cfg.death_ratio))[0]:
            self._die(i, "died")
        min_age = np.maximum(3600.0, 50.0 * self.planet.taus[self.y])
        ready = np.nonzero(self.alive & (ratio >= cfg.repro_ratio) & (self.t - self.last_split >= min_age))[0]
        dens, cap = self._density(), pl.capacity()
        for i in ready[np.argsort(-ratio[ready])]:
            self._divide(i, ratio, dens, cap)
        pl.crowding = dens / np.maximum(cap, 1)
        alive = int(self.alive.sum())
        if alive < self.min_population:
            self._seed(min(self.min_population - alive, max(8, self.min_population // 20)))

    def _seed(self, n):
        """Panspermia: new random organisms at random places with room, all over the planet."""
        pl = self.planet
        free = np.nonzero(~self.alive)[0][:n]
        dens, cap = self._density(), pl.capacity()
        for i in free:
            room = np.flatnonzero(dens < cap)
            if len(room) == 0:
                return
            self.x[i], self.y[i] = divmod(int(room[self.rng.integers(len(room))]), pl.Y)
            dens[self.x[i], self.y[i]] += 1
            self._new_genome(i)
            self._new_identity(i, None)
            self.acct.open(i, self.cfg.initial_capital)
            self.ref[i] = self.cfg.initial_capital
            self.injected += self.cfg.initial_capital
            self.counts["seeded"] += 1

    def _divide(self, i, ratio, dens, cap):
        pl = self.planet
        x, y = self.x[i], self.y[i]
        places = [(x, y), ((x + 1) % pl.X, y), ((x - 1) % pl.X, y)]
        places += [(x, y + 1)] if y + 1 < pl.Y else []
        places += [(x, y - 1)] if y > 0 else []
        free = [(cx, cy) for cx, cy in places if dens[cx, cy] < cap[cx, cy]]
        slots = np.nonzero(~self.alive)[0]
        if free and len(slots):
            cx, cy = free[self.rng.integers(len(free))]
            c = slots[0]
        else:                                               # crowded out: displace the weakest here
            here = np.nonzero(self.alive & (self.x == x) & (self.y == y))[0]
            here = here[here != i]
            if len(here) == 0:
                return
            c = here[np.argmin(ratio[here])]
            if ratio[c] >= ratio[i]:
                return
            self._die(c, "displaced")
            cx, cy = x, y
        dens[cx, cy] += 1
        self.x[c], self.y[c] = cx, cy
        self._inherit(i, c)
        self._new_identity(c, i)
        self.growth[i] *= ratio[i]
        self.acct.split(i, c)
        self.ref[i] = self.ref[c] = self.equity(i)
        self.last_split[i] = self.t
        self.counts["born"] += 1

    def _new_identity(self, i, parent):
        self.alive[i] = True
        self.oid[i] = self.next_id
        self.next_id += 1
        self.born[i] = self.last_split[i] = self.t
        self.growth[i] = 1.0
        self.pending[i] = np.nan
        self.root[i] = self.oid[i] if parent is None else self.root[parent]
        self.gen[i] = 0 if parent is None else self.gen[parent] + 1

    def _set_leverage(self, i, lev):
        self.lev[i] = float(np.clip(round(lev), 1, self.cfg.max_leverage))
        self.max_notional[i] = self.acct.brackets.max_notional(self.lev[i])

    def _new_genome(self, i):
        rng = self.rng
        self._set_leverage(i, np.exp(rng.uniform(0, np.log(25))))
        self.frac[i] = np.exp(rng.uniform(np.log(0.05), 0.0))
        self.hmult[i] = rng.choice((1, 2, 4, 8))
        self.memory[i] = np.exp(rng.uniform(np.log(100), np.log(3000)))
        self.p_mutate[i] = rng.uniform(0.2, 0.8)
        self.p_build[i] = rng.uniform(0.0, 0.3)
        self.gate_w[i] = rng.normal(1.0, 0.3, N_GATE)
        self.gate_theta[i] = rng.normal(1.5, 0.5)
        for slot in range(K):
            self.prog[i, slot] = self._random_program(i)
        self.pstate[i] = 0.0
        self.value[i] = np.nan
        self.integrity[i] = rng.uniform(0.6, 1.0, K)       # staggered, so organs do not all wear out at once
        self.age[i] = 0
        self.W[i] = 0.0
        self.P[i] = np.eye(D) * 0.05
        self.s2[i] = 1e-4
        self.nupd[i] = 0
        self.q_len[i] = 0
        self.recent_n[i] = 0
        self.salience[i] = 1.0 / K
        self.grip[i] = self.turbulence[i] = self.conflict[i] = self.drift[i] = 0.0
        self.err_fast[i] = self.err_slow[i] = 1.0
        self.explore[i], self.resilience[i] = 0.3, 0.5
        self.refractory[i] = 0
        self.episode_due[i] = -1
        self.decisions[i] = self.episodes[i] = 0
        self.a_E[i] = rng.normal(0, 0.1, self.a_E.shape[1:])
        self.a_Wq[i] = rng.normal(0, 0.1, self.a_Wq.shape[1:])
        self.a_n[i] = 0

    def _inherit(self, parent, child):
        rng = self.rng
        for arr in (self.prog, self.pstate, self.value, self.integrity, self.age, self.W, self.P,
                    self.s2, self.nupd, self.recent, self.recent_n, self.salience, self.grip,
                    self.err_fast, self.err_slow, self.turbulence, self.conflict, self.drift,
                    self.explore, self.resilience, self.hmult, self.memory, self.p_mutate,
                    self.p_build, self.gate_w, self.gate_theta, self.frac,
                    self.a_E, self.a_Wq, self.a_n):
            arr[child] = arr[parent]
        self._set_leverage(child, self.lev[parent] * np.exp(0.2 * rng.normal()))
        self.frac[child] = float(np.clip(self.frac[child] * np.exp(0.2 * rng.normal()), 0.02, 1.0))
        self.memory[child] = float(np.clip(self.memory[child] * np.exp(0.2 * rng.normal()), 50, 10000))
        self.gate_w[child] += 0.1 * rng.normal(size=N_GATE)
        self.gate_theta[child] += 0.1 * rng.normal()
        self.p_build[child] = float(np.clip(self.p_build[child] + 0.03 * rng.normal(), 0, 1))
        if rng.random() < 0.1:
            self.hmult[child] = rng.choice((1, 2, 4, 8))
        if self.y[child] != self.y[parent]:
            self.pstate[child] = 0.0
            self.value[child] = np.nan
            self.recent_n[child] = 0
        self.q_len[child] = 0
        self.refractory[child] = 0
        self.episode_due[child] = -1
        self.decisions[child] = self.episodes[child] = 0
        if rng.random() < 0.5:
            slot = rng.integers(K)
            self._replace(child, slot, self._mutate(child, self.prog[child, slot]))

    def _die(self, i, cause):
        self.withdrawn += float(self.equity(i))
        self.fees_dead += float(self.acct.fees[i])
        self.funding_dead += float(self.acct.funding[i])
        self.liq_dead += int(self.acct.liquidations[i])
        self.acct.close(i)
        self.alive[i] = False
        self.pending[i] = np.nan
        self.q_len[i] = 0
        self.counts[cause] += 1

    # ------------------------------------------------------------- watching
    def census(self, n_organs=12, n_lineages=8, n_observatories=12):
        """A snapshot of life on the planet, for watching it over time."""
        pl = self.planet
        live = np.nonzero(self.alive)[0]
        ys = self.y[live]
        dens = self._density()
        eq = self.equity(live)
        log_ratio = np.log(np.maximum(eq / self.ref[live], 1e-9))
        wealth = np.zeros((pl.X, pl.Y))
        np.add.at(wealth, (self.x[live], ys), log_ratio)
        by_band = np.bincount(ys, minlength=pl.Y)

        def band_mean(v):
            ok = np.isfinite(v)
            s = np.bincount(ys[ok], weights=v[ok], minlength=pl.Y)
            return (s / np.maximum(np.bincount(ys[ok], minlength=pl.Y), 1)).round(4).tolist()

        organs, lineages = [], []
        if len(live):
            progs = self.prog[live].reshape(-1, 5)
            u, inv = np.unique(progs, axis=0, return_inverse=True)
            inv = inv.ravel()
            w = np.bincount(inv, weights=self.salience[live].ravel())
            cnt = np.bincount(inv)
            organs = [(describe(u[j]), round(float(w[j] / len(live)), 5), int(cnt[j]))
                      for j in np.argsort(-w)[:n_organs]]
            roots, rc = np.unique(self.root[live], return_counts=True)
            for j in np.argsort(-rc)[:n_lineages]:
                m = live[self.root[live] == roots[j]]
                lineages.append({"root": int(roots[j]), "members": int(rc[j]),
                                 "max_generation": int(self.gen[m].max()),
                                 "bands": np.bincount(self.y[m], minlength=pl.Y).tolist()})
        obs = []
        xs, yy, js = np.nonzero(pl.obs_integrity > 0)
        order = np.argsort(pl.obs_born[xs, yy, js])[:n_observatories]
        for x, y, j in zip(xs[order], yy[order], js[order]):
            obs.append({"x": int(x), "y": int(y), "program": describe(pl.obs_prog[x, y, j]),
                        "age_s": round(float(self.t - pl.obs_born[x, y, j])),
                        "integrity": round(float(pl.obs_integrity[x, y, j]), 3)})
        held = float(eq.sum())
        fees = self.fees_dead + float(self.acct.fees[live].sum())
        funding = self.funding_dead + float(self.acct.funding[live].sum())
        return {
            "t": self.t, "price": self.price, "price0": self.price0,
            "population": int(len(live)), "by_band": by_band.tolist(),
            "density": dens.tolist(), "wealth": (wealth / np.maximum(dens, 1)).round(4).tolist(),
            "growth": pl.growth.round(5).tolist(), "water": pl.water.round(3).tolist(),
            "observatories": (pl.obs_integrity > 0).sum(2).tolist(),
            "conscious": band_mean((self.refractory[live] > 0).astype(float)),
            "explore": band_mean(self.explore[live]), "resilience": band_mean(self.resilience[live]),
            "exposure": band_mean(np.abs(self.exposure(live))), "leverage": band_mean(self.lev[live]),
            "log_wealth": band_mean(log_ratio), "grip": band_mean(self.grip[live]),
            "max_generation": int(self.gen[live].max()) if len(live) else 0,
            "net": held + self.withdrawn - self.injected, "injected": self.injected,
            "fees": fees, "funding": funding,
            "liquidations": self.liq_dead + int(self.acct.liquidations[live].sum()),
            "counts": dict(self.counts), "organs": organs, "lineages": lineages,
            "oldest_observatories": obs,
        }


def _phi(v):
    return np.concatenate([v, np.maximum(v - 1, 0), np.maximum(-v - 1, 0),
                           np.ones(v.shape[:-1] + (1,))], axis=-1)


def _scores(Qv, xmax, lam, cost):
    """Value of each move net of the cost of moving, the value of staying, and the moves."""
    x = ACTIONS * xmax[..., None]
    score = Qv - cost * np.abs(x - lam[..., None])
    u = np.clip(lam / np.maximum(xmax, 1e-12), -1, 1)
    j = np.clip(np.searchsorted(ACTIONS, u) - 1, 0, N_ACT - 2)
    w = (u - ACTIONS[j]) / (ACTIONS[j + 1] - ACTIONS[j])
    stay = ((1 - w) * np.take_along_axis(Qv, j[..., None], -1)[..., 0]
            + w * np.take_along_axis(Qv, (j + 1)[..., None], -1)[..., 0])
    return score, x, stay


def _best_value(Qv, xmax, lam, cost):
    """What the best of moving or staying is worth, for values Qv (n, ..., A) of an
    organism with reach xmax and exposure lam (n,)."""
    n = len(xmax)
    shape = (n,) + (1,) * (Qv.ndim - 2)
    move = (cost * np.abs(ACTIONS * xmax[:, None] - lam[:, None])).reshape(shape + (N_ACT,))
    u = np.clip(lam / np.maximum(xmax, 1e-12), -1, 1)
    j = np.clip(np.searchsorted(ACTIONS, u) - 1, 0, N_ACT - 2)
    w = ((u - ACTIONS[j]) / (ACTIONS[j + 1] - ACTIONS[j])).reshape(shape)
    lo = np.take_along_axis(Qv, np.broadcast_to(j.reshape(shape + (1,)), Qv.shape[:-1] + (1,)), -1)[..., 0]
    hi = np.take_along_axis(Qv, np.broadcast_to((j + 1).reshape(shape + (1,)), Qv.shape[:-1] + (1,)), -1)[..., 0]
    return np.maximum((Qv - move).max(-1), (1 - w) * lo + w * hi)


def _scatter_add(target, rows, values):
    """target[rows] += values, adding up repeated rows (a fast np.add.at for 2-D)."""
    order = np.argsort(rows, kind="stable")
    r = rows[order]
    starts = np.r_[0, np.nonzero(r[1:] != r[:-1])[0] + 1]
    target[r[starts]] += np.add.reduceat(values[order], starts, axis=0)


def _nearest(lam, xmax):
    u = np.clip(lam / np.maximum(xmax, 1e-12), -1, 1)
    return np.abs(ACTIONS[None, :] - u[:, None]).argmin(1)


def _harvest(x, lev, p0, last, low, high, fund_mark):
    """What exposures x (n, A) would have done to accounts with leverage lev, entered at
    p0, over a stretch of the path that ended at `last`, reached `low` and `high`, and
    paid funding rate x mark summing to `fund_mark` (all (n,)): log growth, with
    liquidation (isolated margin lost) when the stretch crossed the liquidation price."""
    lev, p0 = lev[:, None], p0[:, None]
    drift = last[:, None] / p0 - 1.0
    paid = fund_mark[:, None] / p0
    hit_long = low[:, None] / p0 <= (1 - 1 / lev) / (1 - MMR)
    hit_short = high[:, None] / p0 >= (1 + 1 / lev) / (1 + MMR)
    liq = np.where(x > 0, hit_long, np.where(x < 0, hit_short, False))
    mult = np.where(liq, 1.0 - np.abs(x) / lev, 1.0 + x * (drift - paid))
    return np.log(np.maximum(mult, 1e-6))
