"""A primordial soup on the BTC planet: life that is not designed (docs/soup.md).

Nothing here describes an organism. There is matter (bytes), space (sites on the
planet), time (ticks of the market), energy (one exact BTCUSDT account per site)
and a handful of physical laws. Whatever replicates, dies, mates, preys,
cooperates or trades does so because the laws allow it, not because a rule says
it should. The laws themselves are scalar code in physics.py, compiled for the
CPU and for CUDA (gpu.py), so that a world runs the same on either.

Matter     every site holds a tape of L bytes. Bytes are rewritten, never
           created or destroyed.
Chemistry  when two neighboring sites interact, their tapes are joined and run
           as one program (an extension of BFF, Agüera y Arcas et al. 2024; which
           byte value means which operation is a table, see chemistry()):
             < >   move head 0        { }   move head 1
             + -   change the byte under head 0
             . ,   copy between the heads (head 0 -> head 1, head 1 -> head 0)
             [ ]   loop while the byte under head 0 is not zero
           Every other byte is inert. Copying exists; replication does not.
Market     Bytes the market reads or writes are signed, and 0 means nothing: the
           zero that ends every loop is stillness, not a position.
           S   replace the byte under head 0 with a reading of the market
               stream it names, as this site's latitude and longitude perceive it
               (how far above or below its own recent median it is; 0: ordinary)
           E   replace it with the site's own energy (0: its starting stake) or
               position (0: flat): interoception
           A   set the site's exposure to BTC from it (positive long, negative
               short); the order fills at the open after its latitude's next tick
           T   move a bite of energy between the two sites: a positive byte takes
               up to one quantum from the other site's free balance, a negative
               one gives up to one quantum of its own
           An instruction acts for the site whose half of the joined tape it
           sits in, and never takes itself as its own argument (an A or T with
           head 0 on itself reads 0, nothing).
Physics    speed: an interaction runs at most STEPS instructions.
           metabolism: each tick a site wakes with a chance proportional to its
           energy to the power 3/4 (Kleiber's law): energy buys time, with
           diminishing returns for the very large; a world that loses energy
           slows down. A woken site claims a random neighbor; a site takes part
           in at most one interaction per tick, so crowding decides who meets.
           heat: every irreversible byte write (+ - . , S E) costs the writing
           site a little energy (Landauer's principle); a site that cannot pay
           cannot write, so matter without energy is inert. Thinking, copying and
           growing all cost; what is too big for the information it harvests
           runs too hot and fades.
           digestion: only part of the energy one site takes from (or gives to)
           another arrives; the rest is lost, as in every food web.
           noise: bytes flip at random, rarely.
"""

import lzma

import numpy as np
from numba import njit

from .physics import (ACT_NONE, COPIES, E_CLOSE, E_MARK, E_OPEN, E_T, ENTRY, FEES, FUNDING,
                      I_SEED, I_STEPS, I_XS, I_Y, INSTRUCTIONS, INTERACTIONS, L, LEV, LIQUIDATIONS,
                      MARGIN, N_ACCT, N_COUNT, N_FP, N_IP, ORDERS, P_DIGESTION, P_FEE,
                      P_HALF_SPREAD, P_HEAT, P_KLEIBER, P_MAX_EXPOSURE, P_MAX_LEV, P_MIN_NOTIONAL,
                      P_QUANTUM, P_RATE, P_STAKE, P_STEP, PENDING, Q, STEPS, TRADES, WALLET, cpu,
                      run_tape)
from .weather import NS, Weather

MNEMONIC = " <>{}-+.,[]SEAT"                       # MNEMONIC[op] names operation op
# Version of the laws. 1: initiators drawn in proportion to energy ** 0.75, a fixed number a
# second, run one after another (until October 2026). 2: physics.py, the same on CPU and GPU:
# each site wakes with a chance set by its own energy, and a site meets at most one neighbor
# a tick.
PHYSICS = 2


def chemistry(layout="symmetric", market=True):
    """A table of 256 operations, one per byte value.

    "bff": the ASCII bytes of Agüera y Arcas et al. 2024 (and S E A T for the market).
    "symmetric": mirror pairs at +/-(64 - k) as signed bytes (64 - k and 192 + k), so
    that code read as data is as often positive as negative (no built-in lean to long
    or short, to take or give), and readings of the market, which cluster near 0,
    rarely become instructions.
    """
    table = np.zeros(256, np.uint8)
    ops = MNEMONIC[1:] if market else MNEMONIC[1:11]
    if layout == "bff":
        for ch in ops:
            table[ord(ch)] = MNEMONIC.index(ch)
    else:
        pairs = ["-+", "<>", "{}", ",.", "[]", "SE", "AT"]
        for k, (low, high) in enumerate(pairs, start=1):
            for ch, b in ((low, 64 - k), (high, 192 + k)):
                if ch in ops:
                    table[b] = MNEMONIC.index(ch)
    return table


BFF = chemistry("bff")
SYMMETRIC = chemistry("symmetric")


def assemble(text, table=SYMMETRIC, filler=128):
    """Bytes for a program written in mnemonics; any other character is inert filler."""
    code = {MNEMONIC[op]: b for b, op in enumerate(table) if op}
    return np.array([code.get(ch, filler) if isinstance(ch, str) else ch for ch in text], np.uint8)


def disassemble(tape, table=SYMMETRIC):
    return "".join(MNEMONIC[table[b]] if table[b] else "·" for b in np.asarray(tape, np.uint8))


def translate(matter, source=BFF, target=SYMMETRIC):
    """The same programs in another chemistry: each instruction becomes the byte that means
    it there, zero stays zero, and any other inert byte that would mean something there
    moves to the nearest byte that does not."""
    m = np.asarray(matter, np.uint8)
    code = {op: b for b, op in enumerate(target) if op}
    inert = np.nonzero(target == 0)[0]
    inert = inert[inert > 0]
    lut = np.arange(256, dtype=np.uint8)
    for b in range(256):
        if source[b]:
            lut[b] = code.get(int(source[b]), inert[np.argmin(np.abs(inert - b))])
        elif b and target[b]:
            lut[b] = inert[np.argmin(np.abs(inert - b))]
    return lut[m]


_run = njit(cache=True)(run_tape)


def run(tape, steps, sense_a, sense_b, self_a, self_b, wallet, a, b, act, flow, heat=0.0,
        digestion=1.0, quantum=0.01, table=SYMMETRIC):
    """Run one joined tape in place (physics.run_tape). Returns (instructions executed, bytes copied)."""
    return _run(tape, steps, sense_a, sense_b, self_a, self_b, wallet, a, b, act, flow, float(heat),
                float(digestion), float(quantum), table)


def neighbors(X, Y, radius=2):
    """Offsets of each site's neighborhood on an X (wrapping) by Y (bounded) lattice."""
    offs = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if (dx, dy) != (0, 0) and abs(dy) < Y and abs(dx) < X]
    return np.array(offs, np.int64).reshape(-1, 2)


def high_order_entropy(soup):
    """Shannon entropy of the bytes minus their compressed size, in bits per byte.

    Near zero for random matter; it rises sharply when copies of a few programs
    fill the soup (the measure of Agüera y Arcas et al. 2024).
    """
    flat = soup.tobytes()
    counts = np.bincount(np.frombuffer(flat, np.uint8), minlength=256)
    p = counts[counts > 0] / len(flat)
    h0 = float(-(p * np.log2(p)).sum())
    k = 8.0 * len(lzma.compress(flat, preset=6)) / len(flat)
    return h0 - k


def replicates(tape, steps=STEPS, partner=None, table=SYMMETRIC):
    """Does this tape copy itself into an empty (or given) neighbor? Share of bytes copied."""
    n = len(tape)
    joined = np.concatenate([np.asarray(tape, np.uint8),
                             np.zeros(n, np.uint8) if partner is None else np.asarray(partner, np.uint8)])
    s = np.zeros(1, np.uint8)
    run(joined, steps, s, s, np.zeros(2, np.uint8), np.zeros(2, np.uint8), np.zeros(2), 0, 1,
        np.full(2, -1, np.int16), np.zeros((2, 3)), 0.0, 1.0, 0.0, table)
    return float((joined[n:] == np.asarray(tape, np.uint8)).mean())


def census_of_matter(soup, top=8, table=SYMMETRIC, sample=16384):
    """The most common tapes, how much of the soup they fill, and whether they replicate.

    High-order entropy and the share of instructions are measured on at most `sample`
    tapes, evenly spaced over the planet (compressing a soup of millions of tapes would
    take longer than living).
    """
    soup = np.ascontiguousarray(soup, np.uint8)
    _, first, counts = np.unique(tape_hashes(soup), return_index=True, return_counts=True)
    order = np.argsort(-counts)[:top]
    some = soup if len(soup) <= sample else soup[np.linspace(0, len(soup) - 1, sample).astype(np.int64)]
    return {
        "distinct": int(len(counts)),
        "instructions": float((table[some] > 0).mean()),
        "top": [(disassemble(soup[first[j]], table), int(counts[j]), replicates(soup[first[j]], table=table))
                for j in order],
        "entropy": high_order_entropy(some),
    }


def tape_hashes(soup):
    """A 64-bit fingerprint of each tape (equal tapes, equal fingerprints; others collide
    with odds of about one in 10^19 a pair)."""
    words = np.ascontiguousarray(soup, np.uint8).view(np.uint64)
    h = np.zeros(len(soup), np.uint64)
    with np.errstate(over="ignore"):
        for j in range(words.shape[1]):
            h = (h ^ words[:, j]) * np.uint64(0x9E3779B97F4A7C15)
            h ^= h >> np.uint64(29)
    return h


class Primordial:
    """Matter alone, with no market: does life start? (experiments/soup_emergence.py)

    Each epoch every site wakes with chance `wake`, claims a random neighbor, and
    the pairs whose claims won run their joined tapes; bytes flip at random,
    rarely. Nothing rewards anything.
    """

    def __init__(self, X, Y, seed=0, table=BFF, steps=STEPS, wake=1.0, matter=None):
        self.X, self.Y, self.S = X, Y, X * Y
        self.seed, self.wake, self.steps = int(seed), float(wake), int(steps)
        self.table = np.asarray(table, np.uint8)
        self.rng = np.random.default_rng(seed)
        self.soup = (self.rng.integers(0, 256, (self.S, L), dtype=np.uint8) if matter is None
                     else np.array(matter, np.uint8).reshape(self.S, L).copy())
        self.offsets = neighbors(X, Y)
        self.partner = np.full(self.S, -1, np.int64)
        self.claimv = np.zeros(self.S, np.uint64)
        self.claims = np.zeros((2, self.S), np.uint64)
        self.stats = np.zeros((self.S, 3), np.int64)       # interactions, instructions, copies
        self.epoch = 0
        self._engine = None

    def to_gpu(self, sync="grid"):
        from .gpu import PrimordialEngine
        self._engine = PrimordialEngine(self, sync=sync)
        return self

    def sync(self):
        if self._engine is not None:
            self._engine.download()
        return self

    def __getstate__(self):
        self.sync()
        state = self.__dict__.copy()
        state["_engine"] = None
        return state

    def advance(self, epochs, mutation=0.0):
        """`epochs` epochs; then bytes flip, each with chance `mutation` per epoch."""
        e0, e1 = self.epoch, self.epoch + int(epochs)
        if self._engine is not None:
            self._engine.epochs(e0, e1)
        else:
            cpu.cpu_epochs(e0, e1, self.seed, self.wake, self.soup, self.offsets, self.X, self.Y,
                       self.partner, self.claimv, self.claims, self.steps, self.table, self.stats,
                       np.empty(2 * L, np.uint8), np.zeros(1, np.uint8), np.zeros(2, np.uint8),
                       np.zeros(2), np.full(2, -1, np.int16), np.zeros((2, 3)))
        self.epoch = e1
        k = self.rng.poisson(mutation * self.S * L * epochs)
        if k:
            self.sync()
            flat = self.soup.reshape(-1)
            flat[self.rng.integers(flat.size, size=k)] = self.rng.integers(0, 256, size=k, dtype=np.uint8)
            if self._engine is not None:
                self._engine.upload_soup()


# --------------------------------------------------------------------------- the world
def signed(byte):
    """A byte as a signed number: 0..127 as they are, 128..255 as -128..-1."""
    b = np.asarray(byte, np.int64) & 255
    return np.where(b >= 128, b - 256, b)


def unsigned(value):
    return (np.round(np.asarray(value, float)).astype(np.int64) % 256).astype(np.uint8)


# Exposure as a signed byte: 0 is flat, 127 the most long, -127 the most short, and
# logarithmic in between (at 125x: 1 is about 4% of equity, 32 about 2.4x, 64 about 10x;
# at 1x: 64 is about 0.42).
def exposure_of(byte, max_exposure=125.0):
    d = np.clip(signed(byte) / 127.0, -1.0, 1.0)
    return np.sign(d) * (np.power(1.0 + max_exposure, np.abs(d)) - 1.0)


def byte_of_exposure(x, max_exposure=125.0):
    x = np.asarray(x, float)
    d = np.sign(x) * np.log1p(np.minimum(np.abs(x), max_exposure)) / np.log1p(max_exposure)
    return unsigned(127 * d)


def byte_of_energy(ratio):
    """Energy relative to a starting stake, signed: 0 at one stake, +32 per doubling."""
    with np.errstate(divide="ignore"):
        b = 32 * np.log2(np.maximum(np.asarray(ratio, float), 1e-12))
    return unsigned(np.clip(b, -127, 127))


def brackets_matrix(cfg):
    """The exchange's leverage brackets as rows: caps, max leverage, maintenance rate, amount."""
    from .exchange import Brackets
    b = Brackets(cfg.brackets)
    return np.ascontiguousarray(np.stack([b.caps, b.levs, b.rates, b.cum]), dtype=np.float64)


def noise_thresholds(lam, most=16):
    """Cumulative Poisson probabilities: the number of thresholds a uniform exceeds is Poisson(lam)."""
    cdf, p, k = [], np.exp(-lam), 0
    total = p
    while True:
        cdf.append(min(total, 1.0))
        if total >= 1.0 - 1e-15 or k >= most - 1:
            break
        k += 1
        p *= lam / k
        total += p
    cdf[-1] = 1.0
    return np.array(cdf)


class Soup:
    """Matter, energy and time on the planet: the world in which nothing is designed.

    Sites lie on a lattice of planet.X * per_place longitudes by the planet's
    latitudes; a site senses what its place on the planet senses. Each holds L
    bytes of matter and one exact BTCUSDT perpetual account. Every tick (one row
    of market data), physics.py's two phases run: the market (orders placed by
    matter fill at the open, liquidations and funding follow, senses update, a few
    bytes flip, sites wake and claim neighbors) and the interactions. About
    `interactions` sites wake per tick when every site holds its starting stake;
    more when the world is richer, fewer when it is poorer. Energy enters and
    leaves only through the market; transfers only move it.

    The world runs on the CPU, or on a GPU after to_gpu() (gpu.py): the same laws,
    the same random numbers, the same world.
    """

    def __init__(self, cfg, planet, per_place=16, seed=0, interactions=1024, noise=0.00024,
                 steps=STEPS, matter=None, kleiber=0.75, max_exposure=1.0, heat=0.0,
                 digestion=1.0, quantum=0.01, layout="symmetric"):
        self.cfg, self.planet = cfg, planet
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.per_place = per_place
        self.Xs, self.Y = planet.X * per_place, planet.Y
        self.S = S = self.Xs * self.Y
        self.E, self.steps, self.noise, self.kleiber = interactions, int(steps), float(noise), float(kleiber)
        self.max_exposure = min(float(max_exposure), float(cfg.max_leverage))
        self.heat = float(heat)                            # energy per irreversible byte write
        self.digestion = float(digestion)                  # share of a transfer that arrives
        self.quantum = float(quantum)                      # the most energy one bite moves
        self.layout = layout
        self.table = chemistry(layout)                     # which byte means which operation
        self.stake = float(cfg.initial_capital)
        self.soup = (rng.integers(0, 256, (S, L), dtype=np.uint8) if matter is None
                     else np.array(matter, np.uint8).reshape(S, L).copy())
        self.acct = np.zeros((N_ACCT, S))
        self.acct[WALLET] = self.stake
        self.acct[LEV] = 1.0
        self.acct[PENDING] = np.nan
        self.count = np.zeros((N_COUNT, S), np.int64)
        self.flow = np.zeros((S, 3))                       # taken in, given out, dissipated
        self.act = np.full(S, ACT_NONE, np.int16)
        self.injected = self.stake * S
        site = np.arange(S)
        self.site_y = site // self.Xs
        self.site_place = (site % self.Xs) // per_place
        here = planet.avail[self.site_place, self.site_y, :NS // 2]          # (S, streams)
        self.mask = np.concatenate([here, here], 1).astype(np.uint8)
        self.senses = np.zeros((S, NS), np.uint8)          # 0: nothing out of the ordinary
        self.due = np.zeros(S, np.uint8)                   # its band ticked: its order fills next
        self.partner = np.full(S, -1, np.int64)
        self.claimv = np.zeros(S, np.uint64)
        self.claims = np.zeros((2, S), np.uint64)
        self.band_sites = [np.nonzero(self.site_y == y)[0] for y in range(self.Y)]
        self.offsets = neighbors(self.Xs, self.Y)
        self.brackets = brackets_matrix(cfg)
        self.expo = exposure_of(np.arange(256), self.max_exposure)    # what an A byte means
        self.noise_cdf = noise_thresholds(self.noise * self.E * L / S)
        self.fp = np.zeros(N_FP)
        self.fp[[P_STAKE, P_RATE, P_KLEIBER, P_HEAT, P_DIGESTION, P_QUANTUM, P_MAX_EXPOSURE,
                 P_MAX_LEV, P_STEP, P_FEE, P_HALF_SPREAD, P_MIN_NOTIONAL]] = (
            self.stake, self.E / S, self.kleiber, self.heat, self.digestion, self.quantum,
            self.max_exposure, cfg.max_leverage, cfg.qty_step, cfg.taker_fee, cfg.half_spread,
            cfg.min_notional)
        self.ip = np.zeros(N_IP, np.int64)
        self.ip[[I_SEED, I_XS, I_Y, I_STEPS]] = (self.seed, self.Xs, self.Y, self.steps)
        self.weather = Weather(planet.taus, planet.step_s)
        self.tick = 0
        self.t = 0.0
        self.price = self.mark = self.price0 = np.nan
        self._engine = None
        self._stale = False                                # the GPU holds newer state than the host

    # ------------------------------------------------------------------ devices
    def to_gpu(self, sync="grid", block=128):
        """Run on a CUDA GPU from now on (gpu.py). sync: "grid" (one launch per chunk) or "launch"."""
        from .gpu import Engine
        self._engine = Engine(self, sync=sync, block=block)
        return self

    def to_cpu(self):
        self.sync()
        self._engine = None
        return self

    def changed(self):
        """Tell the GPU that the host's arrays were edited (tests, surgery)."""
        if self._engine is not None:
            self._engine.upload()

    @property
    def device(self):
        return "cpu" if self._engine is None else "gpu"

    def sync(self):
        """Bring the state back from the GPU, if it runs there and has moved on."""
        if self._engine is not None and self._stale:
            self._engine.download()
            self._stale = False
        return self

    def __getstate__(self):
        self.sync()
        state = self.__dict__.copy()
        state["_engine"] = None
        state["_stale"] = False
        return state

    # ------------------------------------------------------------------ time
    def advance(self, rows):
        """Live through these rows of market data, one tick each."""
        rows = np.asarray(rows, np.float64)
        if rows.ndim == 1:
            rows = rows[None]
        if len(rows) == 0:
            return
        env, done, band = self.weather.feed(rows)
        if self.price0 != self.price0:
            self.price0 = env[0, E_OPEN]
        if self._engine is not None:
            self._engine.ticks(env, done, band, self.tick)
            self._stale = True
        else:
            cpu.cpu_ticks(0, len(rows), self.tick, env, done, band, self.acct, self.count, self.flow,
                      self.soup, self.senses, self.mask, self.site_y, self.due, self.partner,
                      self.claimv, self.claims, self.act, self.table, self.expo, self.brackets,
                      self.offsets, self.noise_cdf, self.fp, self.ip, np.empty(2 * L, np.uint8),
                      np.empty(2, np.uint8), np.empty(2, np.uint8))
        self.tick += len(rows)
        self.t, self.price, self.mark = env[-1, E_T], env[-1, E_CLOSE], env[-1, E_MARK]

    def step(self, row, done=None):
        """One row (`done` is accepted for the planet's run loop and not needed)."""
        self.advance(row)

    # ------------------------------------------------------------------ reading
    @property
    def wallet(self):
        return self.acct[WALLET]

    @property
    def q(self):
        return self.acct[Q]

    @property
    def pending(self):
        return self.acct[PENDING]

    @property
    def lev(self):
        return self.acct[LEV]

    @property
    def counts(self):
        self.sync()
        c = self.count.sum(1)
        return {"interactions": int(c[INTERACTIONS]), "copies": int(c[COPIES]),
                "instructions": int(c[INSTRUCTIONS]), "orders": int(c[ORDERS]),
                "trades": int(c[TRADES]), "liquidations": int(c[LIQUIDATIONS]), "rain": 0}

    def equity(self, idx=slice(None)):
        self.sync()
        a = self.acct
        return a[WALLET, idx] + a[MARGIN, idx] + a[Q, idx] * (self.mark - a[ENTRY, idx])

    def exposure(self, idx=slice(None)):
        eq = self.equity(idx)                              # (synced)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(eq > 0, self.acct[Q, idx] * self.mark / np.maximum(eq, 1e-9), 0.0)

    # -------------------------------------------------------------- watching
    def census(self, top=8):
        self.sync()
        pl = self.planet
        eq = self.equity()
        x = self.exposure()
        alive = eq >= 0.1 * self.stake
        place = self.site_place * self.Y + self.site_y
        energy = np.bincount(place, weights=eq, minlength=pl.X * pl.Y).reshape(pl.X, pl.Y)
        living = np.bincount(place, weights=alive, minlength=pl.X * pl.Y).reshape(pl.X, pl.Y)
        held = np.bincount(place, weights=np.abs(x) * alive, minlength=pl.X * pl.Y).reshape(pl.X, pl.Y)
        matter = census_of_matter(self.soup, top=top, table=self.table)
        counts = self.counts
        n = max(counts["interactions"], 1)
        by_band = lambda v: [round(float(v[self.band_sites[y]].sum()), 2) for y in range(self.Y)]
        return {
            "t": self.t, "price": self.price, "price0": self.price0, "sites": self.S,
            "alive": int(alive.sum()), "alive_by_band": by_band(alive.astype(float)),
            "energy": float(eq.sum()), "energy_by_band": by_band(eq),
            "net": float(eq.sum()) - self.injected, "injected": self.injected,
            "fees": float(self.acct[FEES].sum()), "funding": float(self.acct[FUNDING].sum()),
            "liquidations": counts["liquidations"],
            "taken": float(self.flow[:, 0].sum()), "heat": float(self.flow[:, 2].sum()),
            "long": float((x > 0.01).mean()), "short": float((x < -0.01).mean()),
            "mean_abs_exposure": float(np.abs(x[alive]).mean()) if alive.any() else 0.0,
            "energy_map": energy.round(1).tolist(), "alive_map": living.astype(int).tolist(),
            "exposure_map": (held / np.maximum(living, 1)).round(3).tolist(),
            "matter": matter, "copies_per_interaction": counts["copies"] / n,
            "instructions_per_interaction": counts["instructions"] / n,
            "interactions_per_tick": counts["interactions"] / max(self.tick, 1),
            "counts": counts,
        }


__all__ = ["L", "STEPS", "PHYSICS", "BFF", "SYMMETRIC", "MNEMONIC", "chemistry", "assemble", "disassemble",
           "translate",
           "run", "neighbors", "high_order_entropy", "replicates", "census_of_matter", "Primordial",
           "Soup", "signed", "unsigned", "exposure_of", "byte_of_exposure", "byte_of_energy"]
