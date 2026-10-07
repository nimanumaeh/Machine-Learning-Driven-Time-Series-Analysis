"""RRBrain: relevance-realizing agents (steps 1-3 of docs/relevance-realization-design.md).

Each agent attends to a few aspects of the market (scarcity), learns online
what they say about the drift and the volatility of its horizon (the arena
model), and reads from that model what the market offers its own account
right now (the affordance value, ``body.py``). Its perspective is derived
from that, not learned separately:

    salience of aspect i  =  how much the affordance value moves when aspect i
                             moves, averaged over the recent present
    frame                 =  top eigenvectors of the expected gradient outer
                             product C = E[grad V grad V^T]

Because V depends on the agent's position, leverage and risk weight, the same
market yields different salience for different bodies: perspective follows
from participation. Attention is then reallocated by salience: aspects that
do not matter to this agent are dropped and new ones tried, so the loop
attention -> model -> affordance -> salience -> attention realizes relevance.

The arena model is recursive least squares with forgetting on features
[o, relu(o - 1), relu(-o - 1), 1] of the attended aspects o, one head for the
horizon's log return and one for its log realised volatility. Salience uses
weights shrunk toward zero by their own uncertainty, so relevance has to be
earned by evidence.
"""

import numpy as np

from .aspects import AspectEngine, N_ASPECTS, NAMES, STREAM_OF
from .body import affordance
from .data import C

EIGHT_HOURS_S = 8 * 3600


def _logpdf(y, mu, sd):
    return -0.5 * ((y - mu) / sd) ** 2 - np.log(sd) - 0.9189385332046727


class RRBrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mc = cfg.mind

    def attach(self, world):
        self.world = world
        S, mc = world.S, self.mc
        B = mc.attention
        self.B, self.D, self.K = B, 3 * B + 1, mc.frame_k
        self.P0 = 1.0 / mc.ridge
        Q = max(mc.horizon_mults) + 2
        self.Q, self.R = Q, mc.recent
        self.engine = AspectEngine()
        self.att = np.zeros((S, B), np.int64)
        self.age = np.zeros((S, B), np.int64)          # perspective updates since attended
        self.missing = np.zeros((S, B), np.int64)      # decisions the aspect was unavailable
        self.w = np.zeros((S, 2, self.D))
        self.P = np.zeros((S, 2, self.D, self.D))
        self.s2 = np.ones((S, 2))
        self.nupd = np.zeros(S, np.int64)
        self.q_phi = np.zeros((S, Q, self.D))
        self.q_due = np.full((S, Q), np.inf)
        self.q_val = np.zeros((S, Q, 6))               # close, cum_rv, mu, sd, naive sd, model sig
        self.q_head = np.zeros(S, np.int64)
        self.q_len = np.zeros(S, np.int64)
        self.recent = np.zeros((S, self.R, B))
        self.recent_n = np.zeros(S, np.int64)
        self.recent_pos = np.zeros(S, np.int64)
        self.salience = np.full((S, B), 1.0 / B)
        self.frame = np.zeros((S, B, self.K))
        self.vcal = np.ones(S)                         # realised / modelled variance
        self.grip = np.zeros(S)                        # grip_drift + grip_vol
        self.grip_drift = np.zeros(S)                  # does my sense of direction hold?
        self.grip_vol = np.zeros(S)                    # is the world as wild as I think?
        self.loss_fast = np.zeros(S)
        self.loss_slow = np.zeros(S)
        self.lp = np.zeros(S)
        self.decisions = np.zeros(S, np.int64)
        self.hmult = np.ones(S, np.int64)
        self.explore = np.zeros(S)
        self.kappa = np.ones(S)
        self.memory = np.ones(S)
        self.cum_rv = 0.0
        self.close = np.nan
        self.funding = 0.0
        self.values = np.full(N_ASPECTS, np.nan)
        self.avail = np.zeros(N_ASPECTS, bool)

    # -------------------------------------------------------------- perceive
    def observe(self, row, t):
        values, avail = self.engine.update(row)
        self.values, self.avail = values, avail
        self.cum_rv += self.engine.ret ** 2
        self.close = row[C["close"]]
        if np.isfinite(self.engine.funding):
            self.funding = self.engine.funding
        self._mature(t)

    def _mature(self, t):
        live = np.nonzero(self.world.alive & (self.q_len > 0))[0]
        while len(live):
            heads = self.q_head[live]
            idx = live[self.q_due[live, heads] <= t + 1e-6]
            if len(idx) == 0:
                break
            h = self.q_head[idx]
            phi = self.q_phi[idx, h]
            close0, rv0, mu, sd, naive, sig = self.q_val[idx, h].T
            y_mu = np.log(self.close / close0)
            y_sg = 0.5 * np.log(np.maximum(self.cum_rv - rv0, 1e-12))
            ratio = np.minimum((y_mu - mu) ** 2 / (sig * sig), 25.0)
            rate = np.maximum(1.0 / (self.nupd[idx] + 2.0), 1.0 / 300)
            self.vcal[idx] += rate * (ratio - self.vcal[idx])
            loss = -_logpdf(y_mu, mu, sd)
            first = self.nupd[idx] == 0
            self.loss_fast[idx] = np.where(first, loss, self.loss_fast[idx] + (loss - self.loss_fast[idx]) / 30)
            self.loss_slow[idx] = np.where(first, loss, self.loss_slow[idx] + (loss - self.loss_slow[idx]) / 300)
            self.lp[idx] = self.loss_slow[idx] - self.loss_fast[idx]
            flat = _logpdf(y_mu, 0.0, sd)               # same volatility, no direction
            self.grip_drift[idx] += ((-loss - flat) - self.grip_drift[idx]) / 100
            self.grip_vol[idx] += ((flat - _logpdf(y_mu, 0.0, naive)) - self.grip_vol[idx]) / 100
            self.grip[idx] = self.grip_drift[idx] + self.grip_vol[idx]
            self._rls(idx, phi, np.stack([y_mu, y_sg], 1))
            self.q_head[idx] = (h + 1) % self.Q
            self.q_len[idx] -= 1
            live = idx[self.q_len[idx] > 0]

    def _rls(self, idx, phi, Y):
        """One recursive-least-squares step (with forgetting) for both heads at once."""
        lam = 1.0 - 1.0 / self.memory[idx]
        P, w = self.P[idx], self.w[idx]                          # (n, 2, D, D), (n, 2, D)
        Pphi = np.matmul(P, phi[:, None, :, None])[..., 0]       # (n, 2, D)
        k = Pphi / (lam[:, None] + np.einsum("nhd,nd->nh", Pphi, phi))[..., None]
        e = Y - np.einsum("nhd,nd->nh", w, phi)
        w += k * e[..., None]
        P -= k[..., :, None] * Pphi[..., None, :]
        P /= lam[:, None, None, None]
        tr = np.einsum("nhii->nh", P)
        P *= np.minimum(1.0, 4.0 * self.D * self.P0 / np.maximum(tr, 1e-12))[..., None, None]
        if self.nupd[idx[0]] % 64 == 0:
            P = 0.5 * (P + np.swapaxes(P, -1, -2))
        self.P[idx], self.w[idx] = P, w
        rate = np.maximum(1.0 / (self.nupd[idx] + 2.0), 1.0 / 500)[:, None]
        self.s2[idx] += rate * (e * e - self.s2[idx])
        self.nupd[idx] += 1

    def _shrunk(self, idx):
        """Weights shrunk toward zero by their own uncertainty: only evidence counts."""
        var_w = self.s2[idx][:, :, None] * np.diagonal(self.P[idx], axis1=2, axis2=3)
        w = self.w[idx]
        shrunk = w * np.maximum(0.0, 1.0 - var_w / (w * w + 1e-30))
        shrunk[:, 1, -1] = w[:, 1, -1]                           # keep the volatility level
        return shrunk

    # ---------------------------------------------------------------- decide
    def _phi(self, o):
        return np.concatenate([o, np.maximum(o - 1, 0), np.maximum(-o - 1, 0),
                               np.ones(o.shape[:-1] + (1,))], axis=-1)

    def _horizon(self, idx):
        return (self.world.tf[idx] // 60) * self.hmult[idx]          # minutes

    def _predict(self, idx, phi, w=None, raw=False):
        """Drift and predictive sd of the horizon's log return for agents idx.

        phi has shape (n, ..., D). Volatility comes from the log-vol head,
        rescaled by how large outcomes have actually been (vcal), plus the
        uncertainty of the drift estimate itself.
        """
        w = self._shrunk(idx) if w is None else w
        n, shape = len(idx), phi.shape[1:-1]
        f = phi.reshape(n, -1, self.D)
        mu = np.einsum("nd,nmd->nm", w[:, 0], f)
        sig = np.exp(np.einsum("nd,nmd->nm", w[:, 1], f))
        pv = self.s2[idx, 0][:, None] * np.einsum("nmd,nde,nme->nm", f, self.P[idx, 0], f)
        cal = sig * np.sqrt(self.vcal[idx])[:, None]
        sd = np.sqrt(cal * cal + pv)
        out = (mu.reshape(n, *shape), sd.reshape(n, *shape))
        return out + (sig.reshape(n, *shape),) if raw else out

    def _body(self, idx, t):
        """Everything about the agents' bodies the affordance value needs."""
        w, cfg = self.world, self.cfg
        hmin = self._horizon(idx)
        settle = (np.floor((t + hmin * 60) / EIGHT_HOURS_S) - np.floor(t / EIGHT_HOURS_S))
        return dict(fund=self.funding * settle, lam=w.exposure(idx), L=w.lev[idx],
                    xmax=w.max_exposure(idx), cost=cfg.taker_fee + cfg.half_spread / self.close,
                    kappa=self.kappa[idx])

    def decide(self, t):
        w, mc = self.world, self.mc
        empty = (np.empty(0, int), np.empty(0))
        if not np.isfinite(self.engine.var_1m):
            return empty
        minute = int(round(t / 60))
        cadence = np.maximum(w.tf // 60, 1)
        idx = np.nonzero(w.alive & (minute % cadence == 0))[0]
        if len(idx) == 0:
            return empty
        self._forced_attention(idx)
        o = self.values[self.att[idx]]
        o = np.where(self.avail[self.att[idx]], o, 0.0)
        phi = self._phi(o)
        mu, sd, sig = self._predict(idx, phi, raw=True)
        b = self._body(idx, t)
        _, x = affordance(mu, sd, b["fund"], b["lam"], b["L"], b["xmax"], b["cost"], b["kappa"],
                          mc.grid)
        hmin = self._horizon(idx)
        self._push(idx, phi, t + hmin * 60.0, mu, sd, np.sqrt(self.engine.var_1m * hmin), sig)
        pos = self.recent_pos[idx]
        self.recent[idx, pos] = o
        self.recent_pos[idx] = (pos + 1) % self.R
        self.recent_n[idx] = np.minimum(self.recent_n[idx] + 1, self.R)
        self.decisions[idx] += 1
        due = idx[(self.decisions[idx] % mc.relandscape_every == 0)
                  & (self.recent_n[idx] >= self.R // 2)]
        if len(due):
            self.perspective(due, t)
            self._attend(due)
        move = np.abs(x - b["lam"]) > 1e-9
        return idx[move], x[move]

    def _push(self, idx, phi, due, mu, sd, naive, sig):
        slot = (self.q_head[idx] + self.q_len[idx]) % self.Q
        full = self.q_len[idx] >= self.Q
        idx, slot = idx[~full], slot[~full]
        self.q_phi[idx, slot] = phi[~full]
        self.q_due[idx, slot] = due[~full]
        self.q_val[idx, slot] = np.stack([np.full(len(idx), self.close), np.full(len(idx), self.cum_rv),
                                          mu[~full], sd[~full], naive[~full], sig[~full]], 1)
        self.q_len[idx] += 1

    # ----------------------------------------------------------- perspective
    def perspective(self, idx, t, delta=1.0):
        """Salience and frame of agents idx, derived from their bodies (emanation).

        For each recent state o and attended aspect i, how much does what the
        market offers this agent change when aspect i moves one typical step
        (aspects are in standard units)?  dV_i = (V(o + d e_i) - V(o - d e_i)) / 2d.
        C = E[dV dV^T] over the recent present; salience = diag(C) / trace(C);
        frame = top eigenvectors of C. For smooth V this is the expected
        gradient outer product; unlike a gradient it still sees what matters
        at the edge of a no-trade zone or when the best move is to step out.
        """
        B = self.B
        O = self.recent[idx]                                     # (n, R, B)
        mask = (np.arange(self.R)[None, :] < self.recent_n[idx][:, None]).astype(float)
        shift = delta * np.eye(B)
        probe = np.concatenate([O[:, :, None, :] + shift, O[:, :, None, :] - shift], axis=2)
        mu, sd = self._predict(idx, self._phi(probe))           # (n, R, 2B)
        b = {k: (v[:, None, None] if np.ndim(v) else v) for k, v in self._body(idx, t).items()}
        V, _ = affordance(mu, sd, b["fund"], b["lam"], b["L"], b["xmax"], b["cost"], b["kappa"],
                          self.mc.grid)
        dV = (V[..., :B] - V[..., B:]) / (2 * delta) * mask[..., None]
        Cm = np.einsum("nrb,nrc->nbc", dV, dV) / np.maximum(mask.sum(1), 1)[:, None, None]
        tr = np.trace(Cm, axis1=1, axis2=2)
        diag = np.diagonal(Cm, axis1=1, axis2=2)
        self.salience[idx] = np.where(tr[:, None] > 0, diag / np.maximum(tr, 1e-300)[:, None],
                                      1.0 / B)
        _, vecs = np.linalg.eigh(Cm)
        self.frame[idx] = vecs[:, :, ::-1][:, :, :self.K]
        self.age[idx] += 1
        return Cm

    # ------------------------------------------------------------- attention
    def _candidates(self, i):
        pool = np.nonzero(self.avail)[0]
        return pool[~np.isin(pool, self.att[i])]

    def _replace(self, i, slot, aspect):
        B = self.B
        f = [slot, B + slot, 2 * B + slot]
        self.att[i, slot] = aspect
        self.age[i, slot] = 0
        self.missing[i, slot] = 0
        for j in (0, 1):
            self.P[i, j][f, :] = 0.0
            self.P[i, j][:, f] = 0.0
            self.P[i, j][f, f] = self.P0
            self.w[i, j][f] = 0.0
        self.q_phi[i][:, f] = 0.0
        self.recent[i][:, slot] = 0.0
        self.salience[i, slot] = 0.0

    def _forced_attention(self, idx):
        gone = ~self.avail[self.att[idx]]
        self.missing[idx] = np.where(gone, self.missing[idx] + 1, 0)
        for k in np.nonzero((self.missing[idx] >= 3).any(1))[0]:
            i = idx[k]
            for slot in np.nonzero(self.missing[i] >= 3)[0]:
                cand = self._candidates(i)
                if len(cand):
                    self._replace(i, slot, self.world.rng.choice(cand))

    def _attend(self, idx):
        rng, mc = self.world.rng, self.mc
        for i in idx:
            mode = mc.control_attention if self.world.control[i] else mc.attention_mode
            if mode == "fixed" or rng.random() >= self.explore[i]:
                continue
            eligible = np.nonzero(self.age[i] >= mc.grace)[0]
            cand = self._candidates(i)
            if len(eligible) == 0 or len(cand) == 0:
                continue
            if mode == "salience":
                slot = eligible[np.argmin(self.salience[i, eligible])]
            else:
                slot = rng.choice(eligible)
            self._replace(i, slot, rng.choice(cand))

    # --------------------------------------------------------------- genomes
    def _reset_mind(self, i):
        B = self.B
        hmin = float(self._horizon(np.array([i]))[0])
        var = self.engine.var_1m if np.isfinite(self.engine.var_1m) else 1e-6
        self.w[i] = 0.0
        self.w[i, 1, -1] = 0.5 * np.log(var * hmin) - 0.635 / np.sqrt(hmin)
        self.P[i] = np.eye(self.D) * self.P0
        self.s2[i] = (var * hmin, 0.3)
        self.nupd[i] = 0
        self.age[i] = 0
        self.missing[i] = 0
        self.q_len[i] = 0
        self.recent[i] = 0.0
        self.recent_n[i] = self.recent_pos[i] = 0
        self.salience[i] = 1.0 / B
        self.frame[i] = 0.0
        self.grip[i] = self.grip_drift[i] = self.grip_vol[i] = 0.0
        self.loss_fast[i] = self.loss_slow[i] = self.lp[i] = 0.0
        self.vcal[i] = 1.0
        self.decisions[i] = 0

    def new_genome(self, i):
        rng, mc = self.world.rng, self.mc
        self.hmult[i] = rng.choice(mc.horizon_mults)
        self.explore[i] = np.exp(rng.uniform(*np.log(mc.explore)))
        self.kappa[i] = np.exp(rng.uniform(*np.log(mc.kappa)))
        self.memory[i] = np.exp(rng.uniform(*np.log(mc.memory)))
        pool = np.nonzero(self.avail)[0]
        if len(pool) < self.B:
            pool = np.arange(N_ASPECTS)
        self.att[i] = rng.choice(pool, self.B, replace=False)
        self._reset_mind(i)

    def inherit(self, parent, child):
        rng, mc, w = self.world.rng, self.mc, self.world
        old_h = float(w.tf[parent] // 60 * self.hmult[parent])
        mults = list(mc.horizon_mults)
        k = mults.index(int(self.hmult[parent]))
        if rng.random() < 0.1:
            k = int(np.clip(k + rng.choice((-1, 1)), 0, len(mults) - 1))
        self.hmult[child] = mults[k]
        self.explore[child] = float(np.clip(self.explore[parent] * np.exp(0.2 * rng.normal()),
                                            0.01, 1.0))
        self.kappa[child] = float(np.clip(self.kappa[parent] * np.exp(0.1 * rng.normal()), 1.0, 10.0))
        self.memory[child] = float(np.clip(self.memory[parent] * np.exp(0.2 * rng.normal()),
                                           100.0, 20000.0))
        if not mc.inherit_mind:
            self.att[child] = self.att[parent]
            self._reset_mind(child)
            return
        for arr in (self.att, self.age, self.w, self.P, self.s2, self.nupd, self.recent,
                    self.recent_n, self.recent_pos, self.salience, self.frame, self.grip, self.vcal,
                    self.grip_drift, self.grip_vol,
                    self.loss_fast, self.loss_slow, self.lp):
            arr[child] = arr[parent]
        self.missing[child] = 0
        self.q_len[child] = 0
        self.decisions[child] = 0
        new_h = float(w.tf[child] // 60 * self.hmult[child])
        if new_h != old_h:                 # carry the model over to the new horizon
            self.w[child, 0] *= new_h / old_h
            self.w[child, 1, -1] += 0.5 * np.log(new_h / old_h)
            self.s2[child, 0] *= new_h / old_h
        if rng.random() < 0.5:
            cand = self._candidates(child)
            if len(cand):
                self._replace(child, rng.integers(self.B), rng.choice(cand))

    def clear(self, i):
        self.q_len[i] = 0

    def genome_bytes(self, i):
        genes = [self.hmult[i], self.explore[i], self.kappa[i], self.memory[i]]
        return np.concatenate([genes, self.att[i]]).astype(np.float32).tobytes()

    # -------------------------------------------------------------- reporting
    def snapshot(self):
        w = self.world
        wild = np.nonzero(w.alive & ~w.control)[0]
        if len(wild) == 0:
            return {}
        weight = np.zeros(N_ASPECTS)
        np.add.at(weight, self.att[wild].ravel(), self.salience[wild].ravel())
        weight /= max(weight.sum(), 1e-12)
        streams = {}
        for s, v in zip(STREAM_OF, weight):
            streams[s] = streams.get(s, 0.0) + float(v)
        top = np.argsort(-weight)[:8]
        learned = wild[self.nupd[wild] > 50]
        return {
            "grip": float(np.median(self.grip[learned])) if len(learned) else None,
            "grip_drift": float(np.median(self.grip_drift[learned])) if len(learned) else None,
            "grip_vol": float(np.median(self.grip_vol[learned])) if len(learned) else None,
            "learning_progress": float(np.median(self.lp[learned])) if len(learned) else None,
            "salience_by_stream": dict(sorted(streams.items(), key=lambda kv: -kv[1])),
            "top_aspects": [(NAMES[k], float(weight[k])) for k in top if weight[k] > 0],
        }

    def describe(self, i):
        order = np.argsort(-self.salience[i])[:3]
        return {
            "horizon_min": int(self._horizon(np.array([i]))[0]),
            "kappa": float(self.kappa[i]), "grip": float(self.grip[i]),
            "attends": [(NAMES[self.att[i, k]], float(self.salience[i, k])) for k in order],
        }
