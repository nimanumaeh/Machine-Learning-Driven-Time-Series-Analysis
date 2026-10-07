"""SelfMadeBrain: agents that build their own perception and learn what matters to them.

Nothing about markets is defined for these agents. They are given only what
any organism is given:

* receptors: the raw market columns, each transduced generically (the log of
  a positive quantity, the raw value of a rate), plus clocks for the day, the
  week and the exchange's 8-hour cycle;
* a body: their exchange account, whose physics (margin, liquidation,
  funding, the cost of moving) is the world's, not a theory about the world;
* capacities: to compose features out of receptors with generic operations,
  to keep only a few of them (attention is scarce), to learn, to act, to die.

What they make themselves:

* their perception: features are small programs over receptors,
  z_tau(op_k(a - b)), with op one of level, change over k, smoothing over k,
  swing over k, and z_tau an adaptive normalization. Agents generate, mutate,
  keep and drop them (generate-and-test, after Mahmood & Sutton 2013);
* their sense of what each move is worth: after every decision the market's
  actual path is run through their own account for each of nine exposures,
  and they learn, per perceived situation, what each would have done to them
  (full-information learning from their own counterfactual participation;
  nothing is assumed about drift, volatility or risk);
* salience: how much what the world offers them changes when one of their own
  features moves, measured on their own learned values.
"""

import numpy as np

from .data import C, EIGHT_HOURS, MINUTE

RECEPTORS = ("close", "high", "low", "mark", "spot", "volume", "quote_volume", "trades",
             "taker_buy", "spot_volume", "spot_taker_buy", "open_interest",
             "open_interest_value", "funding", "premium", "top_account_ls", "top_position_ls",
             "account_ls", "taker_ls", "day_sin", "day_cos", "week_sin", "week_cos",
             "cycle_sin", "cycle_cos")
_SOURCE = {"close": "close", "high": "high", "low": "low", "mark": "mark_close",
           "spot": "spot_close", "volume": "volume", "quote_volume": "quote_volume",
           "trades": "trades", "taker_buy": "taker_buy_volume", "spot_volume": "spot_volume",
           "spot_taker_buy": "spot_taker_buy_volume", "open_interest": "open_interest",
           "open_interest_value": "open_interest_value", "funding": "funding_rate",
           "premium": "premium", "top_account_ls": "top_account_ls",
           "top_position_ls": "top_position_ls", "account_ls": "account_ls",
           "taker_ls": "taker_ls"}
_LOG = [RECEPTORS.index(n) for n in ("close", "high", "low", "mark", "spot", "open_interest",
                                     "open_interest_value", "top_account_ls",
                                     "top_position_ls", "account_ls", "taker_ls")]
_LOG1P = [RECEPTORS.index(n) for n in ("volume", "quote_volume", "trades", "taker_buy",
                                       "spot_volume", "spot_taker_buy")]
_RAW = [RECEPTORS.index(n) for n in ("funding", "premium")]
_COLS = np.array([C[_SOURCE[n]] for n in RECEPTORS if n in _SOURCE])
N_REC = len(RECEPTORS)
ZERO = N_REC                                 # the silent receptor: a - ZERO is just a
OPS = ("level", "change", "smooth", "swing")
LAGS = np.array([1, 4, 16, 64, 256, 1024])
NORMS = np.array([64, 256, 1024, 4096, 16384])
FIELDS = (N_REC, N_REC + 1, len(OPS), len(LAGS), len(NORMS))   # a, b, op, lag, norm
ACTIONS = np.array([-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0])
N_ACT = len(ACTIONS)
MMR = 0.004
BUF = 2048                                   # minutes of receptor and path history kept
N_BODY = 6


def describe_program(p):
    a, b, op, k, tau = (int(x) for x in p)
    series = RECEPTORS[a] if b == ZERO else f"{RECEPTORS[a]} - {RECEPTORS[b]}"
    inner = series if op == 0 else f"{OPS[op]}{LAGS[k]}({series})"
    return f"z{NORMS[tau]}({inner})"


class TokenAttention:
    """Query-key attention over a space of composite tokens (here: programs).

    key(token) = sum of one embedding per field; query = the agent's body; the
    score q.k / sqrt(d) is trained to predict the log-salience agents measure.
    """

    def __init__(self, sizes, rng, d=8, lr=0.03, decay=1e-4):
        self.E = [rng.normal(0, 0.1, (n, d)) for n in sizes]
        self.Wq = rng.normal(0, 0.1, (N_BODY, d))
        self.d, self.lr, self.decay, self.updates = d, lr, decay, 0

    def keys(self, tokens):
        return sum(E[tokens[:, f]] for f, E in enumerate(self.E))

    def scores(self, body, tokens):
        return (body @ self.Wq) @ self.keys(tokens).T / np.sqrt(self.d)

    def learn(self, body, tokens, target):
        k, q = self.keys(tokens), body @ self.Wq
        g = ((q * k).sum(1) / np.sqrt(self.d) - target)[:, None] / np.sqrt(self.d) / len(target)
        self.Wq -= self.lr * (body.T @ (g * k) + self.decay * self.Wq)
        for f, E in enumerate(self.E):
            E *= 1 - self.lr * self.decay
            np.add.at(E, tokens[:, f], -self.lr * g * q)
        self.updates += 1


class SelfMadeBrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mc = cfg.mind

    # ----------------------------------------------------------------- setup
    def attach(self, world):
        self.world = world
        S, mc = world.S, self.mc
        B = mc.attention
        self.B, self.D, self.P0 = B, 3 * B + 1, 1.0 / mc.ridge
        Q = max(mc.horizon_mults) + 2
        self.Q, self.R = Q, mc.recent
        # receptors and the market path, shared by everyone
        self.rbuf = np.full((BUF, N_REC + 1), np.nan)
        self.path = np.full((BUF, 5), np.nan)       # close, mark low, mark high, mark close, funding
        self.n = 0
        self.close = np.nan
        self.var_1m = np.nan
        self._sq = 0.0
        # programs: each agent's perceptual organs
        self.prog = np.zeros((S, B, 5), np.int64)
        self.p_ema = np.zeros((S, B))
        self.p_mean = np.zeros((S, B))
        self.p_var = np.zeros((S, B))
        self.p_cnt = np.zeros((S, B))
        self.value = np.full((S, B), np.nan)
        self.age = np.zeros((S, B), np.int64)
        # what each exposure is worth to me, per perceived situation
        self.W = np.zeros((S, self.D, N_ACT))
        self.P = np.zeros((S, self.D, self.D))
        self.s2 = np.ones((S, N_ACT))
        self.nupd = np.zeros(S, np.int64)
        self.q_phi = np.zeros((S, Q, self.D))
        self.q_due = np.full((S, Q), np.inf)
        self.q_val = np.zeros((S, Q, 6))            # minute, price, xmax, leverage, action, predicted
        self.q_head = np.zeros(S, np.int64)
        self.q_len = np.zeros(S, np.int64)
        self.recent = np.zeros((S, self.R, B))
        self.recent_n = np.zeros(S, np.int64)
        self.recent_pos = np.zeros(S, np.int64)
        self.salience = np.full((S, B), 1.0 / B)
        self.grip = np.zeros(S)
        self.decisions = np.zeros(S, np.int64)
        self.hmult = np.ones(S, np.int64)
        self.explore = np.zeros(S)
        self.memory = np.ones(S)
        self.p_mutate = np.full(S, 0.5)
        self.anticipation = TokenAttention(FIELDS, world.rng)

    # -------------------------------------------------------------- perceive
    def _receive(self, row):
        r = np.full(N_REC + 1, np.nan)
        raw = row[_COLS]
        with np.errstate(divide="ignore", invalid="ignore"):
            r[:len(_COLS)] = raw
            r[_LOG] = np.where(raw[_LOG] > 0, np.log(raw[_LOG]), np.nan)
            r[_LOG1P] = np.where(raw[_LOG1P] >= 0, np.log1p(raw[_LOG1P]), np.nan)
        t = row[C["open_time"]] + MINUTE
        day, week = (t / 86_400_000.0) % 1.0, ((t / 86_400_000.0 + 3) % 7) / 7.0
        cyc = (t % EIGHT_HOURS) / EIGHT_HOURS
        r[len(_COLS):N_REC] = (np.sin(2 * np.pi * day), np.cos(2 * np.pi * day),
                               np.sin(2 * np.pi * week), np.cos(2 * np.pi * week),
                               np.sin(2 * np.pi * cyc), np.cos(2 * np.pi * cyc))
        r[ZERO] = 0.0
        return r

    def observe(self, row, t):
        rec = self._receive(row)
        slot = self.n % BUF
        prev = self.close
        self.close = row[C["close"]]
        if prev > 0:
            ret = np.log(self.close / prev)
            self._sq += (ret * ret - self._sq) / min(self.n + 1, 1440)
            self.var_1m = self._sq if self.n > 60 else np.nan
        self.rbuf[slot] = rec
        self.path[slot] = (self.close, row[C["mark_low"]], row[C["mark_high"]],
                           row[C["mark_close"]], row[C["funding_rate"]])
        self.n += 1
        self._sense(rec)
        self._mature(t)

    def _sense(self, rec):
        """Run every agent's feature programs one minute forward."""
        a, b, op = self.prog[..., 0], self.prog[..., 1], self.prog[..., 2]
        k, tau = LAGS[self.prog[..., 3]], NORMS[self.prog[..., 4]]
        x = rec[a] - rec[b]
        rows = (self.n - 1 - k) % BUF
        past = np.where(self.n - 1 >= k, self.rbuf[rows, a] - self.rbuf[rows, b], np.nan)
        change = x - past
        fresh = self.p_cnt == 0
        ok_x = np.isfinite(x)
        self.p_ema = np.where(ok_x, np.where(fresh, x, self.p_ema + (x - self.p_ema) / k), self.p_ema)
        raw = np.select([op == 0, op == 1, op == 2], [x, change, self.p_ema], np.abs(change))
        ok = np.isfinite(raw)
        d = raw - self.p_mean
        at = 1.0 / tau
        self.p_mean = np.where(ok, np.where(fresh, raw, self.p_mean + at * d), self.p_mean)
        self.p_var = np.where(ok, np.where(fresh, 0.0, (1 - at) * (self.p_var + at * d * d)), self.p_var)
        self.p_cnt += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (raw - self.p_mean) / np.sqrt(self.p_var + 1e-18)
        live = ok & (self.p_cnt >= np.minimum(tau, 240))
        self.value = np.where(live, np.clip(z, -5, 5), np.nan)

    # ------------------------------------------------- learning from outcomes
    def _outcomes(self, i, h):
        """What each of the nine exposures would have done to agent i's equity."""
        m0, p0, xmax, lev = self.q_val[i, h, :4]
        rows = np.arange(int(m0) + 1, self.n) % BUF
        if len(rows) == 0:
            return None
        close, low, high, mark, fund = self.path[rows].T
        x = ACTIONS * xmax
        drift = close[-1] / p0 - 1.0
        paid = np.nansum(fund * mark) / p0                     # funding per unit long exposure
        hit_long = np.nanmin(low) / p0 <= (1 - 1 / lev) / (1 - MMR)
        hit_short = np.nanmax(high) / p0 >= (1 + 1 / lev) / (1 + MMR)
        liq = np.where(x > 0, hit_long, np.where(x < 0, hit_short, False))
        mult = np.where(liq, 1.0 - np.abs(x) / lev, 1.0 + x * drift - x * paid)
        return np.log(np.maximum(mult, 1e-6))

    def _mature(self, t):
        live = np.nonzero(self.world.alive & (self.q_len > 0))[0]
        while len(live):
            idx = live[self.q_due[live, self.q_head[live]] <= t + 1e-6]
            if len(idx) == 0:
                break
            h = self.q_head[idx]
            Y = np.array([self._outcomes(i, hh) for i, hh in zip(idx, h)])
            chosen = self.q_val[idx, h, 4].astype(int)
            pred = self.q_val[idx, h, 5]
            y = Y[np.arange(len(idx)), chosen]
            s2 = self.s2[idx, chosen]
            self.grip[idx] += (((y * y) - (y - pred) ** 2) / (s2 + 1e-12) - self.grip[idx]) / 100
            self._rls(idx, self.q_phi[idx, h], Y)
            self.q_head[idx] = (h + 1) % self.Q
            self.q_len[idx] -= 1
            live = idx[self.q_len[idx] > 0]

    def _rls(self, idx, phi, Y):
        lam = 1.0 - 1.0 / self.memory[idx]
        P, W = self.P[idx], self.W[idx]
        Pphi = np.einsum("nij,nj->ni", P, phi)
        k = Pphi / (lam + np.einsum("ni,ni->n", phi, Pphi))[:, None]
        E = Y - np.einsum("nd,ndh->nh", phi, W)
        W += k[:, :, None] * E[:, None, :]
        P -= k[:, :, None] * Pphi[:, None, :]
        P /= lam[:, None, None]
        tr = np.trace(P, axis1=1, axis2=2)
        P *= np.minimum(1.0, 4.0 * self.D * self.P0 / np.maximum(tr, 1e-12))[:, None, None]
        self.P[idx], self.W[idx] = P, W
        rate = np.maximum(1.0 / (self.nupd[idx] + 2.0), 1.0 / 500)[:, None]
        self.s2[idx] += rate * (E * E - self.s2[idx])
        self.nupd[idx] += 1

    # ---------------------------------------------------------------- decide
    def _phi(self, v):
        return np.concatenate([v, np.maximum(v - 1, 0), np.maximum(-v - 1, 0),
                               np.ones(v.shape[:-1] + (1,))], axis=-1)

    def _shrunk(self, idx):
        var_w = self.s2[idx][:, None, :] * np.diagonal(self.P[idx], axis1=1, axis2=2)[:, :, None]
        W = self.W[idx]
        return W * np.maximum(0.0, 1.0 - var_w / (W * W + 1e-30))

    def _horizon(self, idx):
        return (self.world.tf[idx] // 60) * self.hmult[idx]

    def _choose(self, Q, xmax, lam, cost):
        """Best move given values Q (..., N_ACT): (value, exposure, action index, stay?)."""
        x = ACTIONS * xmax[..., None]
        score = Q - cost * np.abs(x - lam[..., None])
        best = np.argmax(score, axis=-1)
        best_v = np.take_along_axis(score, best[..., None], -1)[..., 0]
        u = np.clip(lam / np.maximum(xmax, 1e-12), -1, 1)
        j = np.clip(np.searchsorted(ACTIONS, u) - 1, 0, N_ACT - 2)
        w = (u - ACTIONS[j]) / (ACTIONS[j + 1] - ACTIONS[j])
        stay = ((1 - w) * np.take_along_axis(Q, j[..., None], -1)[..., 0]
                + w * np.take_along_axis(Q, (j + 1)[..., None], -1)[..., 0])
        keep = stay >= best_v
        nearest = np.where(w < 0.5, j, j + 1)
        return (np.where(keep, stay, best_v), np.take_along_axis(x, best[..., None], -1)[..., 0],
                np.where(keep, nearest, best), keep)

    def decide(self, t):
        w, mc = self.world, self.mc
        empty = (np.empty(0, int), np.empty(0))
        if not np.isfinite(self.var_1m):
            return empty
        minute = int(round(t / 60))
        idx = np.nonzero(w.alive & (minute % np.maximum(w.tf // 60, 1) == 0))[0]
        if len(idx) == 0:
            return empty
        v = np.nan_to_num(self.value[idx])
        phi = self._phi(v)
        Q = np.einsum("nd,ndh->nh", phi, self._shrunk(idx))
        xmax, lam = w.max_exposure(idx), w.exposure(idx)
        cost = self.cfg.taker_fee + self.cfg.half_spread / self.close
        _, x, act, keep = self._choose(Q, xmax, lam, cost)
        hmin = self._horizon(idx)
        slot = (self.q_head[idx] + self.q_len[idx]) % self.Q
        room = self.q_len[idx] < self.Q
        ii, ss = idx[room], slot[room]
        self.q_phi[ii, ss] = phi[room]
        self.q_due[ii, ss] = (t + hmin * 60.0)[room]
        self.q_val[ii, ss] = np.stack([np.full(len(ii), self.n - 1.0), np.full(len(ii), self.close),
                                       xmax[room], w.lev[ii], act[room],
                                       Q[room][np.arange(len(ii)), act[room]]], 1)
        self.q_len[ii] += 1
        pos = self.recent_pos[idx]
        self.recent[idx, pos] = v
        self.recent_pos[idx] = (pos + 1) % self.R
        self.recent_n[idx] = np.minimum(self.recent_n[idx] + 1, self.R)
        self.decisions[idx] += 1
        due = idx[(self.decisions[idx] % mc.relandscape_every == 0)
                  & (self.recent_n[idx] >= self.R // 2)]
        if len(due):
            self.perspective(due)
            self._attend(due)
        return idx[~keep], x[~keep]

    # ----------------------------------------------------------- perspective
    def perspective(self, idx, delta=1.0):
        """Salience of each of the agent's own features, from its own learned values."""
        B, w = self.B, self.world
        O = self.recent[idx]
        mask = (np.arange(self.R)[None, :] < self.recent_n[idx][:, None]).astype(float)
        probe = np.concatenate([O[:, :, None, :] + delta * np.eye(B),
                                O[:, :, None, :] - delta * np.eye(B)], axis=2)
        Q = np.einsum("nrpd,ndh->nrph", self._phi(probe), self._shrunk(idx))
        cost = self.cfg.taker_fee + self.cfg.half_spread / self.close
        xmax = np.broadcast_to(w.max_exposure(idx)[:, None, None], Q.shape[:-1])
        lam = np.broadcast_to(w.exposure(idx)[:, None, None], Q.shape[:-1])
        V, _, _, _ = self._choose(Q, xmax, lam, cost)
        dV = (V[..., :B] - V[..., B:]) / (2 * delta) * mask[..., None]
        Cm = np.einsum("nrb,nrc->nbc", dV, dV) / np.maximum(mask.sum(1), 1)[:, None, None]
        tr = np.trace(Cm, axis1=1, axis2=2)
        self.salience[idx] = np.where(tr[:, None] > 0, np.diagonal(Cm, axis1=1, axis2=2)
                                      / np.maximum(tr, 1e-300)[:, None], 1.0 / B)
        self.age[idx] += 1
        seasoned = (self.age[idx] > self.mc.grace) & (tr[:, None] > 0)
        if seasoned.any():
            rows, slots = np.nonzero(seasoned)
            target = np.log(B * self.salience[idx][rows, slots] + 0.05)
            self.anticipation.learn(self.body_state(idx)[rows], self.prog[idx][rows, slots], target)
        return Cm

    def body_state(self, idx):
        w = self.world
        x = w.exposure(idx)
        return np.stack([np.ones(len(idx)), np.tanh(x / 3), np.tanh(np.abs(x) / 3),
                         np.log(w.lev[idx]) / 5, np.log(self._horizon(idx)) / 6,
                         np.clip(self.grip[idx], -1, 1)], axis=1)

    # ------------------------------------------------- making new perceptions
    def _random_program(self, rng):
        a = rng.integers(N_REC)
        b = ZERO if rng.random() < 0.5 else rng.integers(N_REC)
        if b == a:
            b = ZERO
        return np.array([a, b, rng.integers(len(OPS)), rng.integers(len(LAGS)),
                         rng.integers(len(NORMS))])

    def _mutate(self, p, rng):
        p = p.copy()
        f = rng.integers(5)
        if f == 0:
            p[0] = rng.integers(N_REC)
        elif f == 1:
            p[1] = ZERO if rng.random() < 0.3 else rng.integers(N_REC)
        elif f == 2:
            p[2] = rng.integers(len(OPS))
        else:
            n = len(LAGS) if f == 3 else len(NORMS)
            p[f] = int(np.clip(p[f] + rng.choice((-1, 1)), 0, n - 1))
        if p[1] == p[0]:
            p[1] = ZERO
        return p

    def _new_program(self, i, rng, use_anticipation):
        mine = {tuple(p) for p in self.prog[i]}
        top = self.prog[i][np.argsort(-self.salience[i])[:3]]
        pool = [self._random_program(rng) for _ in range(16)]
        pool += [self._mutate(top[rng.integers(len(top))], rng)
                 for _ in range(int(16 * self.p_mutate[i]))]
        pool = [p for p in pool if tuple(p) not in mine] or [self._random_program(rng)]
        pool = np.array(pool)
        if use_anticipation and self.anticipation.updates > 50:
            z = self.anticipation.scores(self.body_state(np.array([i])), pool)[0]
            p = np.exp(z - z.max())
            p = 0.8 * p / p.sum() + 0.2 / len(pool)
            return pool[rng.choice(len(pool), p=p)]
        return pool[rng.integers(len(pool))]

    def _replace(self, i, slot, program):
        B = self.B
        f = [slot, B + slot, 2 * B + slot]
        self.prog[i, slot] = program
        self.p_ema[i, slot] = self.p_mean[i, slot] = self.p_var[i, slot] = self.p_cnt[i, slot] = 0.0
        self.value[i, slot] = np.nan
        self.age[i, slot] = 0
        self.P[i][f, :] = 0.0
        self.P[i][:, f] = 0.0
        self.P[i][f, f] = self.P0
        self.W[i][f, :] = 0.0
        self.q_phi[i][:, f] = 0.0
        self.recent[i][:, slot] = 0.0
        self.salience[i, slot] = 0.0

    def _attend(self, idx):
        rng, mc = self.world.rng, self.mc
        for i in idx:
            mode = mc.control_attention if self.world.control[i] else mc.attention_mode
            if mode == "fixed" or rng.random() >= self.explore[i]:
                continue
            eligible = np.nonzero(self.age[i] >= mc.grace)[0]
            if len(eligible) == 0:
                continue
            if mode == "salience":
                slot = eligible[np.argmin(self.salience[i, eligible])]
                new = self._new_program(i, rng, mc.anticipate)
            else:
                slot = rng.choice(eligible)
                new = self._random_program(rng)
            self._replace(i, slot, new)

    # --------------------------------------------------------------- genomes
    def _reset_mind(self, i):
        self.W[i] = 0.0
        self.P[i] = np.eye(self.D) * self.P0
        self.s2[i] = 1e-3
        self.nupd[i] = 0
        self.age[i] = 0
        self.q_len[i] = 0
        self.recent[i] = 0.0
        self.recent_n[i] = self.recent_pos[i] = 0
        self.salience[i] = 1.0 / self.B
        self.grip[i] = 0.0
        self.decisions[i] = 0

    def new_genome(self, i):
        rng, mc = self.world.rng, self.mc
        self.hmult[i] = rng.choice(mc.horizon_mults)
        self.explore[i] = np.exp(rng.uniform(*np.log(mc.explore)))
        self.memory[i] = np.exp(rng.uniform(*np.log(mc.memory)))
        self.p_mutate[i] = rng.uniform(0.2, 0.8)
        programs = []
        while len(programs) < self.B:
            p = self._random_program(rng)
            if all(tuple(p) != tuple(q) for q in programs):
                programs.append(p)
        for slot, p in enumerate(programs):
            self.prog[i, slot] = p
        self.p_ema[i] = self.p_mean[i] = self.p_var[i] = self.p_cnt[i] = 0.0
        self.value[i] = np.nan
        self._reset_mind(i)

    def inherit(self, parent, child):
        rng, mc = self.world.rng, self.mc
        mults = list(mc.horizon_mults)
        k = mults.index(int(self.hmult[parent]))
        if rng.random() < 0.1:
            k = int(np.clip(k + rng.choice((-1, 1)), 0, len(mults) - 1))
        self.hmult[child] = mults[k]
        self.explore[child] = float(np.clip(self.explore[parent] * np.exp(0.2 * rng.normal()), 0.01, 1.0))
        self.memory[child] = float(np.clip(self.memory[parent] * np.exp(0.2 * rng.normal()), 100.0, 20000.0))
        self.p_mutate[child] = float(np.clip(self.p_mutate[parent] + 0.1 * rng.normal(), 0.0, 1.0))
        for arr in (self.prog, self.p_ema, self.p_mean, self.p_var, self.p_cnt, self.value, self.age):
            arr[child] = arr[parent]
        if mc.inherit_mind:
            for arr in (self.W, self.P, self.s2, self.nupd, self.recent, self.recent_n,
                        self.recent_pos, self.salience, self.grip):
                arr[child] = arr[parent]
            self.q_len[child] = 0
            self.decisions[child] = 0
        else:
            self._reset_mind(child)
        if rng.random() < 0.5:
            slot = rng.integers(self.B)
            self._replace(child, slot, self._mutate(self.prog[child, slot], rng))

    def clear(self, i):
        self.q_len[i] = 0

    def genome_bytes(self, i):
        genes = [self.hmult[i], self.explore[i], self.memory[i], self.p_mutate[i]]
        return np.concatenate([genes, self.prog[i].ravel()]).astype(np.float32).tobytes()

    # -------------------------------------------------------------- reporting
    def snapshot(self):
        w = self.world
        wild = np.nonzero(w.alive & ~w.control)[0]
        if len(wild) == 0:
            return {}
        sal = self.salience[wild].ravel()
        progs = self.prog[wild].reshape(-1, 5)
        by_receptor = np.zeros(N_REC + 1)
        np.add.at(by_receptor, progs[:, 0], sal)
        np.add.at(by_receptor, progs[:, 1], sal)
        by_receptor[ZERO] = 0.0
        by_receptor /= max(by_receptor.sum(), 1e-12)
        totals = {}
        for p, s in zip(map(tuple, progs), sal):
            totals[p] = totals.get(p, 0.0) + s
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:8]
        norm = max(sal.sum(), 1e-12)
        learned = wild[self.nupd[wild] > 50]
        return {
            "grip": float(np.median(self.grip[learned])) if len(learned) else None,
            "salience_by_receptor": {RECEPTORS[k]: float(by_receptor[k])
                                     for k in np.argsort(-by_receptor)[:10] if by_receptor[k] > 0},
            "top_programs": [(describe_program(p), float(s / norm)) for p, s in top],
        }

    def describe(self, i):
        order = np.argsort(-self.salience[i])[:3]
        return {"horizon_min": int(self._horizon(np.array([i]))[0]), "grip": float(self.grip[i]),
                "attends": [(describe_program(self.prog[i, k]), float(self.salience[i, k]))
                            for k in order]}
