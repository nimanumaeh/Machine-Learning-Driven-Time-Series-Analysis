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

from .physics import (BIRTHS, BORN, CARCASS, COPIES, DEATHS, E_CLOSE, E_MARK, E_OPEN, E_T, ENTRY, FEES,
                      FOUNDER, FUNDING, GEAR, GEN, I_MEET, I_SEED, I_THINK, I_X, I_Y, ID, L, LEV,
                      LIQUIDATIONS, LOST, MARGIN, MEETINGS, METABOLISM, N_ACCT, N_COUNT, N_FLOW, N_FP,
                      N_IDS, N_IP, N_LIFE, N_SELF, ORDERS, P_BIRTH_MIN, P_DIGESTION, P_FEE, P_FLOOR,
                      P_HALF_SPREAD, P_HEAT, P_KLEIBER, P_MAX_LEV, P_METABOLISM, P_MIN_NOTIONAL,
                      P_MUTATION, P_QUANTUM, P_RATE, P_STAKE, P_STEP, P_UPKEEP, P_DIVIDE_AT, P_CYCLE, PENDING, Q,
                      STEPS, TAKEN,
                      THOUGHTS, TO_CHILDREN, TRADES, TRADES0, WALLET, cpu, run_tape)
from .planet import REGIONS
from .selfmade import N_REC, RECEPTORS
from .weather import NS, Weather

MNEMONIC = " <>{}-+.,[]SEATL"                      # MNEMONIC[op] names operation op
# Version of the laws. 1: initiators drawn in proportion to energy ** 0.75, a fixed number a
# second, run one after another, on latitudes of fixed timescales (until October 2026).
# 2: the same on CPU and GPU, each site waking with a chance set by its own energy and
# meeting at most one neighbor a tick. 3: organisms that think on their own at the finest
# tick, pay to live, die, divide, and choose their leverage (physics.py, docs/soup.md).
PHYSICS = 3


def chemistry(layout="symmetric", market=True):
    """A table of 256 operations, one per byte value.

    "bff": the ASCII bytes of Agüera y Arcas et al. 2024 (and S E A T L for the market).
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
        pairs = ["-+", "<>", "{}", ",.", "[]", "SE", "AT", "LL"]
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


def gear_of(byte):
    """Leverage as a signed byte: 0 is 1x, and its size on a log scale up to 125x (sign ignored)."""
    v = np.abs(signed(byte)) / 127.0
    return np.clip(np.round(np.power(126.0, v)), 1.0, 125.0)


class Geography:
    """A torus of X by Y sites; which market streams can be sensed where (continents).

    Price, volume, trades and the clocks exist everywhere. Each regional group of
    streams (order flow, premium, spot, open interest, positioning, funding, mark)
    covers about 55% of the surface, in smooth random continents fixed at birth.
    """

    def __init__(self, X, Y, step_s=60, seed=0):
        self.X, self.Y, self.step_s = int(X), int(Y), int(step_s)
        self.taus = np.array([self.step_s], np.int64)        # one timescale: the finest there is
        rng = np.random.default_rng(seed)
        x, y = np.meshgrid(np.arange(self.X) / self.X, np.arange(self.Y) / self.Y, indexing="ij")
        self.avail = np.ones((self.X, self.Y, N_REC + 1), bool)
        self.regions = {}
        for name, receptors in REGIONS.items():
            field = np.zeros((self.X, self.Y))
            for k in range(1, 4):
                for _ in range(2):
                    a, b = rng.integers(-k, k + 1, size=2)
                    field += rng.normal() * np.sin(2 * np.pi * (a * x + b * y) + rng.uniform(0, 2 * np.pi)) / k
            here = field > np.quantile(field, 0.45)
            self.regions[name] = here
            for r in receptors:
                self.avail[:, :, RECEPTORS.index(r)] = here


class Soup:
    """Living matter on the BTC market: the world in which only trading keeps you alive.

    A torus of X by Y sites. A living site is an organism: L bytes of matter that run
    on their own (physics.think), registers saying where they stopped, and one exact
    BTCUSDT perpetual account whose equity is its energy. Every tick (one row of the
    market at the finest resolution there is) physics.py's three phases run: live,
    birth, meet. Energy enters only through trading (and leaves through fees, funding,
    heat, digestion, the cost of living, and death). Nothing else is given: whatever
    lives, divides, preys, cooperates or trades well does so because it pays.

    Laws and their dials:
      metabolism_days  living costs energy in proportion to energy ** 0.75 (Kleiber), set
                       so that an organism holding one stake and earning nothing loses half
                       of it in this many days (the poorer, the slower it burns, so a loss
                       is not the end)
      upkeep           and a body costs this much (USDT a day) whatever it holds: many tiny
                       bodies cost more to keep than one large one
      floor            an organism whose equity falls to this (USDT) dies
      birth_min        a child needs at least this much (USDT) to be born
      divide_at        a body holding at least this much energy (USDT) may divide in half (as
                       cells do), into an empty site next to it or over a neighbor holding less
                       than the child would: success becomes offspring; 0 means two stakes
      cycle_days       how often a body at one stake tries to divide (more often the more it
                       holds, Kleiber); 0: every tick
      mutation         chance each byte is miscopied when a parent divides
      noise            chance a byte flips, per byte per day
      think            instructions an organism's matter runs each tick
      meet, meetings   instructions a meeting runs; meetings an organism holding one stake
                       starts a day (more the more it holds: Kleiber)
      heat             energy (USDT) each byte written costs (Landauer)
      digestion, quantum  share of a bite that arrives; the most one bite moves (USDT): 0, T
                       moves nothing, and energy passes between organisms only when a child
                       takes over a weaker neighbor's site
    The world runs on the CPU, or on a GPU after to_gpu() (gpu.py): the same laws, the
    same random numbers, the same world.
    """

    def __init__(self, cfg, width=64, height=64, step_s=60, seed=0, occupancy=0.5, think=32, meet=128,
                 meetings=90.0, metabolism_days=365.0, upkeep=1.0, floor=1.0, birth_min=100.0,
                 divide_at=0.0, cycle_days=1.0, mutation=1 / 64,
                 noise=1e-3, heat=2e-5, digestion=0.8, quantum=0.0, kleiber=0.75, matter=None,
                 layout="symmetric", geography_seed=None):
        self.cfg = cfg
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.planet = Geography(width, height, step_s, seed if geography_seed is None else geography_seed)
        self.X, self.Y = self.planet.X, self.planet.Y
        self.S = S = self.X * self.Y
        self.think_steps, self.meet_steps = int(think), int(meet)
        self.meetings, self.kleiber = float(meetings), float(kleiber)
        self.metabolism_days, self.floor, self.birth_min = float(metabolism_days), float(floor), float(birth_min)
        self.upkeep = float(upkeep)
        self.divide_at = float(divide_at) if divide_at else 2.0 * float(cfg.initial_capital)
        self.cycle_days = float(cycle_days)
        self.mutation, self.noise, self.heat = float(mutation), float(noise), float(heat)
        self.digestion, self.quantum = float(digestion), float(quantum)
        self.layout = layout
        self.table = chemistry(layout)
        self.stake = float(cfg.initial_capital)
        # matter, and who is alive
        self.alive = np.zeros(S, np.uint8)
        if matter is None:
            self.soup = np.zeros((S, L), np.uint8)
            born = rng.random(S) < occupancy
            self.soup[born] = rng.integers(0, 256, (int(born.sum()), L), dtype=np.uint8)
        else:
            m = np.atleast_2d(np.asarray(matter, np.uint8))
            self.soup = m[np.arange(S) % len(m)].copy()
            born = rng.random(S) < occupancy
            self.soup[~born] = 0
        self.alive[born] = 1
        self.regs = np.zeros((S, 3), np.int64)
        # energy: one exact account each
        self.acct = np.zeros((N_ACCT, S))
        self.acct[WALLET, born] = self.stake
        self.acct[LEV] = 1.0
        self.acct[GEAR] = 1.0
        self.acct[PENDING] = np.nan
        self.injected = self.stake * float(born.sum())
        self.count = np.zeros((N_COUNT, S), np.int64)
        self.flow = np.zeros((S, N_FLOW))
        self.life = np.zeros((N_LIFE, S), np.int64)
        self.ids = np.zeros((N_IDS, S), np.uint64)
        first = np.nonzero(born)[0].astype(np.uint64)
        self.ids[ID, born] = first + np.uint64(1)              # founders, named by their site
        self.ids[FOUNDER, born] = first + np.uint64(1)
        # senses: what exists where
        site = np.arange(S)
        self.site_x, self.site_y = site % self.X, site // self.X
        here = self.planet.avail[self.site_x, self.site_y, :NS // 2]
        self.mask = np.concatenate([here, here], 1).astype(np.uint8)
        # the bookkeeping of one tick
        self.doomed = np.zeros(S, np.uint8)
        self.partner = np.full(S, -1, np.int64)
        self.claimv = np.zeros(S, np.uint64)
        self.claims = np.zeros((2, S), np.uint64)
        self.target = np.full(S, -1, np.int64)
        self.birthv = np.zeros(S, np.uint64)
        self.births = np.zeros((2, S), np.uint64)
        self.divide = np.zeros(S)
        self.act = np.full(S, -1, np.int16)
        # the laws' constants
        self.offsets = neighbors(self.X, self.Y)
        self.brackets = brackets_matrix(cfg)
        self.expo = exposure_of(np.arange(256), 1.0)        # A: a share of equity, long or short
        self.gears = gear_of(np.arange(256))                  # L: leverage, 1x to 125x
        day = step_s / 86400.0
        self.noise_cdf = noise_thresholds(self.noise * L * day)
        k_day = 0.0 if not np.isfinite(self.metabolism_days) else \
            4 * (1 - 2 ** -0.25) * self.stake ** 0.25 / self.metabolism_days
        self.fp = np.zeros(N_FP)
        self.fp[[P_STAKE, P_RATE, P_KLEIBER, P_HEAT, P_DIGESTION, P_QUANTUM, P_MAX_LEV, P_STEP, P_FEE,
                 P_HALF_SPREAD, P_MIN_NOTIONAL, P_METABOLISM, P_FLOOR, P_BIRTH_MIN, P_MUTATION, P_UPKEEP,
                 P_DIVIDE_AT, P_CYCLE]] = (
            self.stake, self.meetings * day, self.kleiber, self.heat, self.digestion, self.quantum,
            cfg.max_leverage, cfg.qty_step, cfg.taker_fee, cfg.half_spread, cfg.min_notional,
            k_day * day, self.floor, self.birth_min, self.mutation, self.upkeep * day, self.divide_at,
            day / self.cycle_days if self.cycle_days > 0 else np.inf)
        self.ip = np.zeros(N_IP, np.int64)
        self.ip[[I_SEED, I_X, I_Y, I_THINK, I_MEET]] = (self.seed, self.X, self.Y, self.think_steps,
                                                        self.meet_steps)
        self.weather = Weather(self.planet.taus, step_s)
        self.sensed = np.zeros(NS, np.uint8)                  # the market's bytes at the last tick
        self.tick = 0
        self.t = 0.0
        self.price = self.mark = self.price0 = np.nan
        self._engine = None
        self._stale = False

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
            self._engine.ticks(env, band, self.tick)
            self._stale = True
        else:
            cpu.cpu_life(0, len(rows), self.tick, env, band, self.acct, self.count, self.flow, self.soup,
                         self.regs, self.mask, self.alive, self.doomed, self.partner,
                         self.claimv, self.claims, self.target, self.birthv, self.births, self.divide,
                         self.act, self.ids, self.life, self.offsets, self.noise_cdf, self.expo, self.gears,
                         self.table, self.brackets, self.fp, self.ip, np.empty(2 * L, np.uint8),
                         np.empty(N_SELF, np.uint8), np.empty(N_SELF, np.uint8), np.empty(N_SELF, np.uint8))
        self.tick += len(rows)
        self.t, self.price, self.mark = env[-1, E_T], env[-1, E_CLOSE], env[-1, E_MARK]
        self.sensed = band[-1, 0].copy()

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
    def gear(self):
        return self.acct[GEAR]

    @property
    def counts(self):
        self.sync()
        c = self.count.sum(1)
        return {"meetings": int(c[MEETINGS]), "copies": int(c[COPIES]), "thoughts": int(c[THOUGHTS]),
                "orders": int(c[ORDERS]), "trades": int(c[TRADES]), "liquidations": int(c[LIQUIDATIONS]),
                "births": int(c[BIRTHS]), "deaths": int(c[DEATHS])}

    def equity(self, idx=slice(None)):
        self.sync()
        a = self.acct
        return a[WALLET, idx] + a[MARGIN, idx] + a[Q, idx] * (self.mark - a[ENTRY, idx])

    def exposure(self, idx=slice(None)):
        eq = self.equity(idx)                              # (synced)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(eq > 0, self.acct[Q, idx] * self.mark / np.maximum(eq, 1e-9), 0.0)

    def holding_s(self):
        """Each living organism's timescale: its age over the positions it has taken (seconds)."""
        age = (self.tick - self.life[BORN]) * self.planet.step_s
        own = self.count[TRADES] - self.life[TRADES0]
        return age / np.maximum(own, 1) * np.where(own > 0, 1.0, np.inf)

    # -------------------------------------------------------------- watching
    TIMESCALES = (("never", np.inf), ("< 5m", 300), ("< 1h", 3600), ("< 6h", 21600), ("< 1d", 86400),
                  ("< 1w", 604800), ("longer", np.inf))
    GEARS = (("1x", 1.5), ("2-4x", 4.5), ("5-19x", 19.5), ("20x+", np.inf))

    def census(self, top=8, lines=8):
        self.sync()
        pl = self.planet
        eq = self.equity()
        x = self.exposure()
        alive = self.alive.astype(bool)
        n = max(int(alive.sum()), 1)
        flow = self.flow.sum(0)
        counts = self.counts
        # lines of descent: the founders whose descendants are alive
        founders, members = np.unique(self.ids[FOUNDER, alive], return_counts=True)
        order = np.argsort(-members)[:lines]
        gen = self.life[GEN, alive]
        line_rows = []
        for j in order:
            m = alive & (self.ids[FOUNDER] == founders[j])
            line_rows.append({"founder": int(founders[j]), "alive": int(members[j]),
                              "energy": round(float(eq[m].sum()), 2),
                              "generation": int(self.life[GEN, m].max()),
                              "long": float((x[m] > 0.01).mean()), "short": float((x[m] < -0.01).mean())})
        # timescales: how long each organism holds a position, on average, over its life
        hold = self.holding_s()[alive]
        own = (self.count[TRADES] - self.life[TRADES0])[alive]
        bins = [int((own == 0).sum())]
        energy_bins = [round(float(eq[alive][own == 0].sum()), 2)]
        edges = [0, 300, 3600, 21600, 86400, 604800, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (own > 0) & (hold >= lo) & (hold < hi)
            bins.append(int(m.sum()))
            energy_bins.append(round(float(eq[alive][m].sum()), 2))
        g = self.acct[GEAR, alive]
        gear_bins = [int((g < 1.5).sum()), int(((g >= 1.5) & (g < 4.5)).sum()),
                     int(((g >= 4.5) & (g < 19.5)).sum()), int((g >= 19.5).sum())]
        # maps, at most 32 x 32 cells
        fx, fy = max(1, pl.X // 32), max(1, pl.Y // 32)
        cell = (self.site_x // fx) * ((pl.Y + fy - 1) // fy) + self.site_y // fy
        cx, cy = (pl.X + fx - 1) // fx, (pl.Y + fy - 1) // fy
        energy_map = np.bincount(cell, weights=np.where(alive, eq, 0.0), minlength=cx * cy).reshape(cx, cy)
        alive_map = np.bincount(cell, weights=alive, minlength=cx * cy).reshape(cx, cy)
        held = np.bincount(cell, weights=np.abs(x) * alive, minlength=cx * cy).reshape(cx, cy)
        matter = census_of_matter(self.soup[alive] if alive.any() else self.soup[:1], top=top, table=self.table)
        return {
            "t": self.t, "price": self.price, "price0": self.price0, "sites": self.S,
            "alive": int(alive.sum()), "energy": float(eq[alive].sum()),
            "net": float(eq[alive].sum()) - self.injected, "injected": self.injected,
            "fees": float(self.acct[FEES].sum()), "funding": float(self.acct[FUNDING].sum()),
            "liquidations": counts["liquidations"],
            "taken": float(flow[TAKEN]), "lost": float(flow[LOST]), "heat": float(flow[LOST]),
            "metabolism": float(flow[METABOLISM]), "to_children": float(flow[TO_CHILDREN]),
            "carcass": float(flow[CARCASS]),
            "births": counts["births"], "deaths": counts["deaths"],
            "generation_max": int(gen.max()) if len(gen) else 0,
            "generation_mean": float(gen.mean()) if len(gen) else 0.0,
            "lines": int(len(founders)), "top_lines": line_rows,
            "long": float((x[alive] > 0.01).sum() / n), "short": float((x[alive] < -0.01).sum() / n),
            "mean_abs_exposure": float(np.abs(x[alive]).mean()) if alive.any() else 0.0,
            "timescales": bins, "timescale_energy": energy_bins, "gears": gear_bins,
            "energy_map": energy_map.round(1).tolist(), "alive_map": alive_map.astype(int).tolist(),
            "exposure_map": (held / np.maximum(alive_map, 1)).round(3).tolist(),
            "matter": matter, "counts": counts,
            "thoughts_per_tick": counts["thoughts"] / max(self.tick, 1),
        }


__all__ = ["L", "STEPS", "PHYSICS", "BFF", "SYMMETRIC", "MNEMONIC", "chemistry", "assemble", "disassemble",
           "translate",
           "run", "neighbors", "high_order_entropy", "replicates", "census_of_matter", "Primordial",
           "Soup", "signed", "unsigned", "exposure_of", "byte_of_exposure", "byte_of_energy"]
