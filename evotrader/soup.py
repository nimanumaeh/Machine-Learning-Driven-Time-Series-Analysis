"""A primordial soup on the BTC planet: life that is not designed (docs/soup.md).

Nothing here describes an organism. There is matter (bytes), space (sites on the
planet), time (interactions between neighbors), energy (one exact BTCUSDT
account per site) and a handful of physical laws. Whatever replicates, dies,
mates, preys, cooperates or trades does so because the laws allow it, not
because a rule says it should.

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
           metabolism: a site initiates interactions at a rate proportional to
           its energy to the power 3/4 (Kleiber's law): energy buys time, with
           diminishing returns for the very large.
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

try:
    from numba import njit
except ImportError:                                    # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

L = 64                                     # bytes of matter per site
STEPS = 1 << 13                            # most instructions an interaction runs
# Operations, and the chemistry: which byte means which operation (0: inert).
OP_LEFT, OP_RIGHT, OP_LEFT1, OP_RIGHT1, OP_DEC, OP_INC, OP_COPY01, OP_COPY10, \
    OP_OPEN, OP_CLOSE, OP_SENSE, OP_SELF, OP_ACT, OP_TRANSFER = range(1, 15)
MNEMONIC = " <>{}-+.,[]SEAT"                       # MNEMONIC[op] names operation op


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


@njit(cache=True)
def run(tape, steps, sense_a, sense_b, self_a, self_b, wallet, a, b, act, flow, heat=0.0,
        digestion=1.0, quantum=0.01, table=SYMMETRIC):
    """Run one joined tape in place. Returns (instructions executed, bytes copied).

    sense_*: (n_senses,) uint8 readings; self_*: (2,) uint8 energy and position;
    wallet: free balances of all sites (moved by T, and spent as heat); act: (sites,)
    int16 exposure bytes set by A (-1: none); flow: (sites, 3) energy taken in, given
    out, and dissipated. heat: the energy an irreversible byte write costs the site
    whose code makes it (Landauer); a site that cannot pay cannot write.
    digestion: the share of energy moved by T that arrives; the rest is dissipated
    (no transfer between living things is lossless). quantum: the most energy one T
    moves (a bite); draining a neighbor takes a loop, and time.
    """
    n = tape.shape[0]
    half = n // 2
    ip = 0
    h0 = 0
    h1 = 0
    copies = 0
    executed = 0
    for _ in range(steps):
        if ip < 0 or ip >= n:
            break
        c = table[tape[ip]]
        executed += 1
        if heat > 0.0 and (c == OP_DEC or c == OP_INC or c == OP_COPY01 or c == OP_COPY10
                           or c == OP_SENSE or c == OP_SELF):
            me = a if ip < half else b
            if wallet[me] < heat:                          # no energy, no writing
                ip += 1
                continue
            wallet[me] -= heat
            flow[me, 2] += heat
        if c == OP_LEFT:
            h0 = (h0 - 1) % n
        elif c == OP_RIGHT:
            h0 = (h0 + 1) % n
        elif c == OP_LEFT1:
            h1 = (h1 - 1) % n
        elif c == OP_RIGHT1:
            h1 = (h1 + 1) % n
        elif c == OP_DEC:
            tape[h0] = (tape[h0] + 255) & 255
        elif c == OP_INC:
            tape[h0] = (tape[h0] + 1) & 255
        elif c == OP_COPY01:
            tape[h1] = tape[h0]
            copies += 1
        elif c == OP_COPY10:
            tape[h0] = tape[h1]
            copies += 1
        elif c == OP_OPEN:
            if tape[h0] == 0:
                depth = 1
                j = ip + 1
                while j < n and depth > 0:
                    if table[tape[j]] == OP_OPEN:
                        depth += 1
                    elif table[tape[j]] == OP_CLOSE:
                        depth -= 1
                    j += 1
                if depth > 0:
                    break
                ip = j - 1
        elif c == OP_CLOSE:
            if tape[h0] != 0:
                depth = 1
                j = ip - 1
                while j >= 0 and depth > 0:
                    if table[tape[j]] == OP_CLOSE:
                        depth += 1
                    elif table[tape[j]] == OP_OPEN:
                        depth -= 1
                    j -= 1
                if depth > 0:
                    break
                ip = j + 1
        elif c == OP_SENSE:
            s = sense_a if ip < half else sense_b
            tape[h0] = s[tape[h0] % s.shape[0]]
        elif c == OP_SELF:
            s = self_a if ip < half else self_b
            tape[h0] = s[tape[h0] & 1]
        elif c == OP_ACT:
            act[a if ip < half else b] = tape[h0] if h0 != ip else 0   # never its own argument
        elif c == OP_TRANSFER:
            me = a if ip < half else b
            other = b if ip < half else a
            v = tape[h0] if h0 != ip else 0                # never its own argument
            if v == 0:
                pass
            elif v < 128:                                  # positive: take
                amount = min(max(wallet[other], 0.0), quantum * v / 127.0)
                wallet[other] -= amount
                wallet[me] += amount * digestion
                flow[me, 0] += amount * digestion
                flow[other, 1] += amount
                flow[me, 2] += amount * (1.0 - digestion)
            else:                                          # negative: give
                amount = min(max(wallet[me], 0.0), quantum * (256 - v) / 128.0)
                wallet[me] -= amount
                wallet[other] += amount * digestion
                flow[me, 1] += amount
                flow[other, 0] += amount * digestion
                flow[me, 2] += amount * (1.0 - digestion)
        ip += 1
    return executed, copies


@njit(cache=True)
def interact(soup, pairs, steps, senses, selfs, wallet, act, flow, stats, heat=0.0, digestion=1.0,
             quantum=0.01, table=SYMMETRIC):
    """Run every pair (initiator, neighbor) in order, rewriting the soup in place.

    stats: (len(pairs), 2) instructions executed and bytes copied per interaction.
    """
    n = soup.shape[1]
    tape = np.empty(2 * n, np.uint8)
    for k in range(pairs.shape[0]):
        a = pairs[k, 0]
        b = pairs[k, 1]
        tape[:n] = soup[a]
        tape[n:] = soup[b]
        e, c = run(tape, steps, senses[a], senses[b], selfs[a], selfs[b], wallet, a, b, act, flow,
                   heat, digestion, quantum, table)
        stats[k, 0] = e
        stats[k, 1] = c
        soup[a] = tape[:n]
        soup[b] = tape[n:]


def neighbors(X, Y, radius=2):
    """Offsets of each site's neighborhood on an X (wrapping) by Y (bounded) lattice."""
    offs = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if (dx, dy) != (0, 0) and abs(dy) < Y and abs(dx) < X]
    return np.array(offs, np.int64)


def pair_up(rng, X, Y, initiators, offsets):
    """A random neighbor for each initiator site (sites are numbered y * X + x)."""
    x, y = initiators % X, initiators // X
    o = offsets[rng.integers(len(offsets), size=len(initiators))]
    nx = (x + o[:, 0]) % X
    ny = y + o[:, 1]
    flip = (ny < 0) | (ny >= Y)                            # the poles reflect
    ny = np.clip(np.where(flip, y - o[:, 1], ny), 0, Y - 1)
    return np.stack([initiators, ny * X + nx], 1)


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
    act = np.full(2, -1, np.int16)
    flow = np.zeros((2, 3))
    run(joined, steps, s, s, s[:1].repeat(2), s[:1].repeat(2), np.zeros(2), 0, 1, act, flow,
        0.0, 1.0, 0.01, table)
    return float((joined[n:] == np.asarray(tape, np.uint8)).mean())


def census_of_matter(soup, top=8, table=SYMMETRIC):
    """The most common tapes, how much of the soup they fill, and whether they replicate."""
    uniq, counts = np.unique(soup, axis=0, return_counts=True)
    order = np.argsort(-counts)[:top]
    return {
        "distinct": int(len(uniq)),
        "instructions": float((table[soup] > 0).mean()),
        "top": [(disassemble(uniq[j], table), int(counts[j]), replicates(uniq[j], table=table))
                for j in order],
        "entropy": high_order_entropy(soup),
    }


__all__ = ["L", "STEPS", "BFF", "SYMMETRIC", "chemistry", "assemble", "disassemble", "run",
           "interact", "neighbors", "pair_up", "high_order_entropy", "replicates", "census_of_matter"]


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


class Soup:
    """Matter, energy and time on the planet: the world in which nothing is designed.

    Sites lie on a lattice of planet.X * per_place longitudes by the planet's
    latitudes; a site senses what its place on the planet senses. Each holds L
    bytes of matter and one exact BTCUSDT perpetual account. Every second:
    orders placed by matter fill at the open, liquidations and funding follow,
    then `interactions` pairs of neighboring sites run their joined tapes, the
    initiators drawn in proportion to energy ** 0.75, and a few bytes flip.
    Energy enters and leaves only through the market; transfers only move it.
    Optionally (`floor` > 0), when the planet's total energy falls below that share
    of what it started with, fresh stakes fall on dead sites, counted as money put
    in. Off by default: rain would pay random matter to gamble.
    """

    def __init__(self, cfg, planet, per_place=16, seed=0, interactions=1024, noise=0.00024,
                 floor=0.0, steps=STEPS, matter=None, kleiber=0.75, max_exposure=1.0, heat=0.0,
                 digestion=1.0, quantum=0.01, layout="symmetric"):
        from .exchange import Accounts
        self.cfg, self.planet = cfg, planet
        self.rng = np.random.default_rng(seed)
        self.per_place = per_place
        self.Xs, self.Y = planet.X * per_place, planet.Y
        self.S = S = self.Xs * self.Y
        self.E, self.steps, self.noise, self.kleiber = interactions, steps, noise, kleiber
        self.max_exposure = min(float(max_exposure), float(cfg.max_leverage))
        self.heat = float(heat)                            # energy per irreversible byte write
        self.digestion = float(digestion)                  # share of a transfer that arrives
        self.quantum = float(quantum)                      # the most energy one bite moves
        self.table = chemistry(layout)                     # which byte means which operation
        self.stake = cfg.initial_capital
        self.soup = (self.rng.integers(0, 256, (S, L), dtype=np.uint8) if matter is None
                     else np.array(matter, np.uint8).reshape(S, L).copy())
        self.acct = Accounts(S, cfg)
        for s in range(S):
            self.acct.open(s, self.stake)
        self.injected = self.stake * S
        self.floor_total = floor * self.injected
        self.lev = np.ones(S)
        self.pending = np.full(S, np.nan)
        self.act = np.full(S, -1, np.int16)
        self.flow = np.zeros((S, 3))                       # taken in, given out, dissipated
        self.stats = np.zeros((interactions, 2), np.int64)
        site = np.arange(S)
        self.site_y = site // self.Xs
        self.site_place = (site % self.Xs) // per_place
        ns = planet.sense_bytes.shape[1]
        here = planet.avail[self.site_place, self.site_y, :ns // 2]          # (S, streams)
        self.mask = np.concatenate([here, here], 1)
        self.senses = np.zeros((S, ns), np.uint8)              # 0: nothing out of the ordinary
        self.selfs = np.zeros((S, 2), np.uint8)
        self.band_sites = [np.nonzero(self.site_y == y)[0] for y in range(self.Y)]
        self.offsets = neighbors(self.Xs, self.Y)
        self.counts = {"interactions": 0, "copies": 0, "instructions": 0, "orders": 0, "rain": 0}
        self.due = np.zeros(S, bool)                       # sites whose band ticked: their orders fill next
        self.t = 0.0
        self.price = self.mark = self.price0 = np.nan

    # --------------------------------------------------------------- one second
    def step(self, row, done):
        from .data import C
        self.t = self.planet.t
        if self.price0 != self.price0:
            self.price0 = row[C["open"]]
            self.mark = row[C["open"]]
        self._execute(row[C["open"]], self.due)
        hi, lo = row[C["mark_high"]], row[C["mark_low"]]
        if not (hi > 0 and lo > 0):
            hi, lo = row[C["high"]], row[C["low"]]
        self.acct.liquidate(hi, lo)
        f = row[C["funding_rate"]]
        self.price = row[C["close"]]
        self.mark = row[C["mark_close"]] if row[C["mark_close"]] > 0 else self.price
        if f and f == f:
            self.acct.settle_funding(f, self.mark)
        for y in done:
            sites = self.band_sites[int(y)]
            self.senses[sites] = np.where(self.mask[sites], self.planet.sense_bytes[int(y)], 0)
        self._live()
        self._noise()
        self.due = np.isin(self.site_y, np.asarray(done))  # a latitude decides at its own tick
        if int(round(self.t)) % 60 == 0:
            self._rain()

    def equity(self, idx=slice(None)):
        return self.acct.equity(self.mark, idx)

    def exposure(self, idx=slice(None)):
        eq = self.equity(idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(eq > 0, self.acct.q[idx] * self.mark / np.maximum(eq, 1e-9), 0.0)

    def _execute(self, price, due):
        """The latest intention of each site whose band just ticked becomes an order now."""
        idx = np.nonzero(due & ~np.isnan(self.pending))[0]
        if len(idx) == 0:
            return
        x = self.pending[idx]
        self.pending[idx] = np.nan
        self.lev[idx] = np.clip(np.ceil(np.abs(x)), 1, self.cfg.max_leverage)
        eq = np.maximum(self.acct.equity(price, idx), 0.0)
        cap = self.acct.brackets.max_notional(self.lev[idx])
        notional = np.clip(x * eq, -cap, cap)
        self.acct.execute(idx, notional / price, price, self.lev[idx])
        self.counts["orders"] += len(idx)

    def _live(self):
        eq = np.maximum(self.equity(), 0.0)
        w = eq ** self.kleiber
        total = w.sum()
        if total <= 0:
            return
        starters = self.rng.choice(self.S, size=self.E, p=w / total)
        pairs = pair_up(self.rng, self.Xs, self.Y, starters, self.offsets)
        self.selfs[:, 0] = byte_of_energy(eq / self.stake)
        self.selfs[:, 1] = byte_of_exposure(self.exposure(), self.max_exposure)
        interact(self.soup, pairs, self.steps, self.senses, self.selfs, self.acct.wallet,
                 self.act, self.flow, self.stats, self.heat, self.digestion, self.quantum,
                 self.table)
        acted = np.nonzero(self.act >= 0)[0]
        if len(acted):
            self.pending[acted] = exposure_of(self.act[acted], self.max_exposure)
            self.act[acted] = -1
        self.counts["interactions"] += self.E
        self.counts["instructions"] += int(self.stats[:, 0].sum())
        self.counts["copies"] += int(self.stats[:, 1].sum())

    def _noise(self):
        k = self.rng.poisson(self.noise * self.E * L)          # noise per byte, per S interactions
        if k:
            flat = self.soup.reshape(-1)
            flat[self.rng.integers(flat.size, size=k)] = self.rng.integers(0, 256, size=k, dtype=np.uint8)

    def _rain(self):
        """When the planet runs down, fresh stakes fall on dead sites (money put in)."""
        eq = self.equity()
        deficit = self.floor_total - eq.sum()
        if deficit <= 0:
            return
        dead = np.nonzero(eq < 0.1 * self.stake)[0]
        n = min(len(dead), int(np.ceil(deficit / self.stake)))
        for s in self.rng.permutation(dead)[:n]:
            self.acct.wallet[s] += self.stake
            self.injected += self.stake
            self.counts["rain"] += 1

    # -------------------------------------------------------------- watching
    def census(self, top=8):
        pl = self.planet
        eq = self.equity()
        x = self.exposure()
        alive = eq >= 0.1 * self.stake
        place = self.site_place * self.Y + self.site_y
        energy = np.bincount(place, weights=eq, minlength=pl.X * pl.Y).reshape(pl.X, pl.Y)
        living = np.bincount(place, weights=alive, minlength=pl.X * pl.Y).reshape(pl.X, pl.Y)
        held = np.bincount(place, weights=np.abs(x) * alive, minlength=pl.X * pl.Y).reshape(pl.X, pl.Y)
        matter = census_of_matter(self.soup, top=top, table=self.table)
        fees = float(self.acct.fees.sum())
        funding = float(self.acct.funding.sum())
        n = max(self.counts["interactions"], 1)
        by_band = lambda v: [round(float(v[self.band_sites[y]].sum()), 2) for y in range(self.Y)]
        return {
            "t": self.t, "price": self.price, "price0": self.price0, "sites": self.S,
            "alive": int(alive.sum()), "alive_by_band": by_band(alive.astype(float)),
            "energy": float(eq.sum()), "energy_by_band": by_band(eq),
            "net": float(eq.sum()) - self.injected, "injected": self.injected,
            "fees": fees, "funding": funding, "liquidations": int(self.acct.liquidations.sum()),
            "taken": float(self.flow[:, 0].sum()), "heat": float(self.flow[:, 2].sum()),
            "long": float((x > 0.01).mean()), "short": float((x < -0.01).mean()),
            "mean_abs_exposure": float(np.abs(x[alive]).mean()) if alive.any() else 0.0,
            "energy_map": energy.round(1).tolist(), "alive_map": living.astype(int).tolist(),
            "exposure_map": (held / np.maximum(living, 1)).round(3).tolist(),
            "matter": matter, "copies_per_interaction": self.counts["copies"] / n,
            "instructions_per_interaction": self.counts["instructions"] / n,
            "counts": dict(self.counts),
        }
