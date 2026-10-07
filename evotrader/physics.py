"""The soup's laws as scalar code, compiled for the CPU (numba) and for CUDA (docs/soup.md).

Every function here acts on one site, or on one pair of sites, and compiles
unchanged for both targets (numba.extending.register_jitable). A tick of the
world has two phases. Within a phase no two sites write the same memory, so the
CPU loops over the sites one by one and a GPU runs them all at once, with the
same result:

market       each site alone: the order its matter placed at its latitude's last
             tick fills at the open, liquidation and funding follow, its senses
             update if its latitude ticked, a few of its bytes may flip, and it
             may wake (more often the more energy it has: Kleiber's law) and claim
             a random neighbor.
interaction  every woken site whose claims on itself and on its neighbor both won
             (the highest random priority among all claims on either site) runs
             their joined tape. A site takes part in at most one interaction per
             tick: crowding, not a rule, decides who meets whom.

Randomness is a pure function of (seed, tick, stream, site), so a world does not
depend on the order in which its sites are visited, nor on the device.

The laws are written once, as plain functions, and compiled for each target by
target(): inlined into the CPU loops below, device functions for gpu.py.
"""

import math
from types import FunctionType, ModuleType

import numpy as np
from numba import njit
from numba.extending import register_jitable

L = 64                                     # bytes of matter per site
TAPE = 2 * L                               # a joined tape
STEPS = 1 << 13                            # most instructions an interaction runs
OP_LEFT, OP_RIGHT, OP_LEFT1, OP_RIGHT1, OP_DEC, OP_INC, OP_COPY01, OP_COPY10, \
    OP_OPEN, OP_CLOSE, OP_SENSE, OP_SELF, OP_ACT, OP_TRANSFER = range(1, 15)

# rows of the account matrix: one exact BTCUSDT perpetual account per site
WALLET, Q, ENTRY, MARGIN, FEES, FUNDING, LEV, PENDING = range(8)
N_ACCT = 8
# rows of the counter matrix
LIQUIDATIONS, TRADES, ORDERS, INTERACTIONS, INSTRUCTIONS, COPIES = range(6)
N_COUNT = 6
# columns of the market, one row per tick
E_T, E_OPEN, E_CLOSE, E_HIGH, E_LOW, E_MARK, E_FUNDING = range(7)
N_ENV = 7
# the world's constants
P_STAKE, P_RATE, P_KLEIBER, P_HEAT, P_DIGESTION, P_QUANTUM, P_MAX_EXPOSURE, P_MAX_LEV, \
    P_STEP, P_FEE, P_HALF_SPREAD, P_MIN_NOTIONAL = range(12)
N_FP = 12
I_SEED, I_XS, I_Y, I_STEPS = range(4)
N_IP = 4
# independent streams of randomness
WAKE, PARTNER, KEY, NOISE, NOISE_AT = range(1, 6)
ACT_NONE = -1                              # no intention this tick

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_EPS = 1e-9


# --------------------------------------------------------------------- randomness
def mix(z):
    """SplitMix64's finalizer: 64 well-mixed bits from 64 bits."""
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def bits(seed, tick, stream, i):
    """64 random bits, a pure function of (seed, tick, stream, i)."""
    z = mix(np.uint64(seed) + _GOLD * np.uint64(tick))
    return mix(z ^ (np.uint64(stream) << np.uint64(56)) ^ np.uint64(i))


def uniform(seed, tick, stream, i):
    """A uniform number in [0, 1)."""
    return np.float64(bits(seed, tick, stream, i) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


# --------------------------------------------------------------------- chemistry
def run_tape(tape, steps, sense_a, sense_b, self_a, self_b, wallet, a, b, act, flow, heat,
             digestion, quantum, table):
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
            h0 = (h0 - 1 + n) % n
        elif c == OP_RIGHT:
            h0 = (h0 + 1) % n
        elif c == OP_LEFT1:
            h1 = (h1 - 1 + n) % n
        elif c == OP_RIGHT1:
            h1 = (h1 + 1) % n
        elif c == OP_DEC:
            tape[h0] = (np.int64(tape[h0]) + 255) & 255
        elif c == OP_INC:
            tape[h0] = (np.int64(tape[h0]) + 1) & 255
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
            if ip < half:
                tape[h0] = sense_a[tape[h0] % sense_a.shape[0]]
            else:
                tape[h0] = sense_b[tape[h0] % sense_b.shape[0]]
        elif c == OP_SELF:
            if ip < half:
                tape[h0] = self_a[tape[h0] & 1]
            else:
                tape[h0] = self_b[tape[h0] & 1]
        elif c == OP_ACT:
            v = tape[h0] if h0 != ip else 0                # never its own argument
            if ip < half:
                act[a] = v
            else:
                act[b] = v
        elif c == OP_TRANSFER:
            me = a if ip < half else b
            other = b if ip < half else a
            v = np.int64(tape[h0]) if h0 != ip else 0      # never its own argument
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


# --------------------------------------------------------------------- the exchange
# Exactly what exchange.Accounts does for one account, in the same order of operations.
def sign(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def tier(notional, br):
    """Maintenance rate and amount of the bracket holding this notional (br: caps, levs, rates, cum)."""
    nb = br.shape[1]
    k = 0
    while k < nb and br[0, k] < notional:
        k += 1
    if k > nb - 1:
        k = nb - 1
    return br[2, k], br[3, k]


def max_notional(lev, br):
    best = 0.0
    for k in range(br.shape[1]):
        if br[1, k] >= lev and br[0, k] > best:
            best = br[0, k]
    return best


def equity(s, mark, acct):
    return acct[WALLET, s] + acct[MARGIN, s] + acct[Q, s] * (mark - acct[ENTRY, s])


def fill(s, target_q, price, leverage, acct, counts, step, fee, hs, min_notional):
    """A market order taking site s toward target_q BTC at price. Returns the fee paid."""
    q0 = acct[Q, s]
    E = acct[ENTRY, s]
    M = acct[MARGIN, s]
    W = acct[WALLET, s]
    s0 = sign(q0)
    st = sign(target_q)
    a0 = abs(q0)
    at = abs(target_q)
    # 1) reduce, close, or close before flipping
    c = 0.0
    if s0 != 0.0 and st != s0:
        c = a0
    elif s0 != 0.0 and st == s0 and at < a0:
        c = math.floor((a0 - at) / step + _EPS) * step
    c = min(c, a0)
    pc = price - s0 * hs
    share = c / a0 if a0 > 0.0 else 0.0
    release = M * share
    fee_c = c * pc * fee
    W += release + c * (pc - E) * s0 - fee_c
    M -= release
    q = q0 - s0 * c
    if abs(q) < _EPS:
        W += M
        q = 0.0
        E = 0.0
        M = 0.0
    # 2) open or add, limited by what the wallet can margin
    a = abs(q)
    o = 0.0
    if st != 0.0 and (sign(q) == st or a == 0.0):
        o = math.floor(max(at - a, 0.0) / step + _EPS) * step
    po = price + st * hs
    per_btc = po * (1.0 / leverage + fee)
    o = min(o, math.floor(max(W, 0.0) / per_btc / step + _EPS) * step)
    if not (o * po >= min_notional):
        o = 0.0
    new_a = a + o
    if o > 0.0:
        E = (a * E + o * po) / max(new_a, _EPS)
    fee_o = o * po * fee
    M += o * po / leverage
    W -= o * po / leverage + fee_o
    if o > 0.0:
        q = st * new_a
    acct[Q, s] = q
    acct[ENTRY, s] = E
    acct[MARGIN, s] = M
    acct[WALLET, s] = W
    paid = fee_c + fee_o
    acct[FEES, s] += paid
    if c > 0.0 or o > 0.0:
        counts[TRADES, s] += 1
    return paid


def liquidate(s, high, low, acct, counts, br):
    """Isolated margin: a position the mark's range touched is lost, the rest of the wallet is not."""
    q = acct[Q, s]
    if q == 0.0:
        return
    e = acct[ENTRY, s]
    m = acct[MARGIN, s]
    mmr, cum = tier(abs(q) * e, br)
    if q > 0.0:
        hit = low <= (q * e - m - cum) / (q * (1 - mmr))
    else:
        hit = high >= (m - q * e + cum) / (-q * (1 + mmr))
    if hit:
        acct[WALLET, s] += min(m, 0.0)                    # funding debt, if any
        acct[Q, s] = 0.0
        acct[ENTRY, s] = 0.0
        acct[MARGIN, s] = 0.0
        counts[LIQUIDATIONS, s] += 1


def fund(s, rate, mark, acct):
    """Funding settlement: longs pay shorts when the rate is positive."""
    q = acct[Q, s]
    if q != 0.0:
        pay = q * mark * rate
        acct[MARGIN, s] -= pay
        acct[FUNDING, s] += pay


# --------------------------------------------------------------------- signed bytes
def rint(x):
    """Round half to even, as numpy does."""
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1.0
    if d < 0.5:
        return f
    return f if f - 2.0 * math.floor(f / 2.0) == 0.0 else f + 1.0


def to_byte(v):
    """A finite number as a signed byte held in 0..255 (soup.unsigned)."""
    return np.int64(rint(v)) & 255


def energy_byte(eq, stake):
    """Energy relative to the stake: 0 at one stake, +32 per doubling (soup.byte_of_energy)."""
    b = 32.0 * math.log2(max(eq / stake, 1e-12))
    return to_byte(min(max(b, -127.0), 127.0))


def exposure_byte(x, max_exposure):
    """An exposure as a logarithmic signed byte (soup.byte_of_exposure)."""
    d = sign(x) * math.log1p(min(abs(x), max_exposure)) / math.log1p(max_exposure)
    return to_byte(127.0 * d)


def read_self(s, mark, acct, fp, out):
    """Interoception: the site's energy and position, as two signed bytes."""
    eq = equity(s, mark, acct)
    held = 0.0
    x = 0.0
    if eq > 0.0:
        held = eq
        x = acct[Q, s] * mark / max(eq, 1e-9)
    out[0] = energy_byte(held, fp[P_STAKE])
    out[1] = exposure_byte(x, fp[P_MAX_EXPOSURE])


# --------------------------------------------------------------------- space
def neighbor(s, dx, dy, Xs, Y):
    """The site at offset (dx, dy) from s: longitude wraps, the poles reflect."""
    x = s % Xs
    y = s // Xs
    nx = (x + dx + Xs) % Xs
    ny = y + dy
    if ny < 0 or ny >= Y:
        ny = y - dy
    if ny < 0:
        ny = 0
    if ny > Y - 1:
        ny = Y - 1
    return ny * Xs + nx


# --------------------------------------------------------------------- one tick
def market_site(s, tick, r, env, done, band, acct, counts, soup, senses, mask, site_y, due,
                partner, claimv, br, offsets, noise_cdf, fp, ip_):
    """The market phase of tick `tick` (row r of env) for site s. Returns its claim (0: none)."""
    seed = ip_[I_SEED]
    # the order its matter placed at its latitude's last tick fills at the open
    x = acct[PENDING, s]
    if due[s] != 0 and x == x:
        acct[PENDING, s] = np.nan
        price = env[r, E_OPEN]
        lv = min(max(math.ceil(abs(x)), 1.0), fp[P_MAX_LEV])
        acct[LEV, s] = lv
        e = equity(s, price, acct)
        if not (e > 0.0):
            e = 0.0
        cap = max_notional(lv, br)
        fill(s, min(max(x * e, -cap), cap) / price, price, lv, acct, counts, fp[P_STEP],
             fp[P_FEE], fp[P_HALF_SPREAD], fp[P_MIN_NOTIONAL])
        counts[ORDERS, s] += 1
    liquidate(s, env[r, E_HIGH], env[r, E_LOW], acct, counts, br)
    f = env[r, E_FUNDING]
    if f != 0.0 and f == f:
        fund(s, f, env[r, E_MARK], acct)
    # senses, when its latitude ticks; streams that do not exist here read 0
    y = site_y[s]
    if done[r, y] != 0:
        for k in range(senses.shape[1]):
            senses[s, k] = band[r, y, k] if mask[s, k] != 0 else 0
    due[s] = done[r, y]                                   # its next order fills at the next open
    # noise: a few bytes flip
    u = uniform(seed, tick, NOISE, s)
    flips = 0
    while flips < noise_cdf.shape[0] and u > noise_cdf[flips]:
        flips += 1
    for j in range(flips):
        z = bits(seed, tick, NOISE_AT, s * 64 + j)
        soup[s, np.int64(z % np.uint64(soup.shape[1]))] = np.uint8((z >> np.uint64(32)) & np.uint64(255))
    # waking, at a rate set by energy (Kleiber), and claiming a random neighbor
    partner[s] = -1
    claimv[s] = np.uint64(0)
    eq = equity(s, env[r, E_MARK], acct)
    if eq > 0.0 and offsets.shape[0] > 0:
        ratio = eq / fp[P_STAKE]
        kl = fp[P_KLEIBER]
        if kl == 0.75:
            w = math.sqrt(ratio) * math.sqrt(math.sqrt(ratio))
        elif kl == 1.0:
            w = ratio
        else:
            w = ratio ** kl
        if uniform(seed, tick, WAKE, s) < fp[P_RATE] * w:
            z = bits(seed, tick, PARTNER, s)
            o = np.int64(z % np.uint64(offsets.shape[0]))
            partner[s] = neighbor(s, offsets[o, 0], offsets[o, 1], ip_[I_XS], ip_[I_Y])
            claimv[s] = ((bits(seed, tick, KEY, s) >> np.uint64(32)) << np.uint64(32)) | np.uint64(s + 1)
    return claimv[s]


def interact_site(s, buf, r, env, acct, counts, flow, soup, senses, act, partner, claimv, claims,
                  table, expo, fp, ip_, tape, sa, sb):
    """The interaction phase for the pair site s started, if both its claims won."""
    claims[1 - buf, s] = np.uint64(0)                     # clean for the next tick
    j = partner[s]
    if j < 0:
        return
    v = claimv[s]
    if claims[buf, s] != v or claims[buf, j] != v:
        return                                            # crowded out this tick
    mark = env[r, E_MARK]
    read_self(s, mark, acct, fp, sa)
    read_self(j, mark, acct, fp, sb)
    n = soup.shape[1]
    for k in range(n):
        tape[k] = soup[s, k]
        tape[n + k] = soup[j, k]
    e, c = run_tape(tape, ip_[I_STEPS], senses[s], senses[j], sa, sb, acct[WALLET], s, j, act, flow,
                    fp[P_HEAT], fp[P_DIGESTION], fp[P_QUANTUM], table)
    for k in range(n):
        soup[s, k] = tape[k]
        soup[j, k] = tape[n + k]
    if act[s] >= 0:                                       # an intention, filled at the next tick
        acct[PENDING, s] = expo[act[s]]
        act[s] = -1
    if act[j] >= 0:
        acct[PENDING, j] = expo[act[j]]
        act[j] = -1
    counts[INTERACTIONS, s] += 1
    counts[INSTRUCTIONS, s] += e
    counts[COPIES, s] += c


def cpu_ticks(r0, r1, tick0, env, done, band, acct, counts, flow, soup, senses, mask, site_y, due,
              partner, claimv, claims, act, table, expo, br, offsets, noise_cdf, fp, ip_, tape, sa, sb):
    """Rows r0..r1 of env, one tick each, on the CPU (tick number tick0 + r).

    tape (2L,), sa and sb (2,) uint8: work space for a joined tape and two selves.
    """
    S = soup.shape[0]
    for r in range(r0, r1):
        tick = tick0 + r
        buf = tick & 1
        for s in range(S):
            v = market_site(s, tick, r, env, done, band, acct, counts, soup, senses, mask, site_y, due,
                            partner, claimv, br, offsets, noise_cdf, fp, ip_)
            if v != np.uint64(0):
                j = partner[s]
                if v > claims[buf, s]:
                    claims[buf, s] = v
                if v > claims[buf, j]:
                    claims[buf, j] = v
        for s in range(S):
            interact_site(s, buf, r, env, acct, counts, flow, soup, senses, act, partner, claimv, claims,
                          table, expo, fp, ip_, tape, sa, sb)


# --------------------------------------------------------------------- matter alone
def bff_claim(s, epoch, seed, wake, offsets, Xs, Y, partner, claimv):
    """Abiogenesis without a market: site s may wake and claim a random neighbor."""
    partner[s] = -1
    claimv[s] = np.uint64(0)
    if offsets.shape[0] > 0 and uniform(seed, epoch, WAKE, s) < wake:
        z = bits(seed, epoch, PARTNER, s)
        o = np.int64(z % np.uint64(offsets.shape[0]))
        partner[s] = neighbor(s, offsets[o, 0], offsets[o, 1], Xs, Y)
        claimv[s] = ((bits(seed, epoch, KEY, s) >> np.uint64(32)) << np.uint64(32)) | np.uint64(s + 1)
    return claimv[s]


def bff_interact(s, buf, soup, partner, claimv, claims, steps, table, sense, selfs, wallet, act,
                 flow, stats, tape):
    claims[1 - buf, s] = np.uint64(0)
    j = partner[s]
    if j < 0:
        return
    v = claimv[s]
    if claims[buf, s] != v or claims[buf, j] != v:
        return
    n = soup.shape[1]
    for k in range(n):
        tape[k] = soup[s, k]
        tape[n + k] = soup[j, k]
    e, c = run_tape(tape, steps, sense, sense, selfs, selfs, wallet, 0, 1, act, flow, 0.0, 1.0, 0.0,
                    table)
    for k in range(n):
        soup[s, k] = tape[k]
        soup[j, k] = tape[n + k]
    stats[s, 0] += 1
    stats[s, 1] += e
    stats[s, 2] += c


def cpu_epochs(e0, e1, seed, wake, soup, offsets, Xs, Y, partner, claimv, claims, steps, table, stats,
               tape, sense, selfs, wallet, act, flow):
    """Epochs e0..e1 of matter alone (no market): wake, claim, run the pairs that won.

    tape (2L,) and the rest: work space (the market instructions act on nothing here).
    """
    S = soup.shape[0]
    for epoch in range(e0, e1):
        buf = epoch & 1
        for s in range(S):
            v = bff_claim(s, epoch, seed, wake, offsets, Xs, Y, partner, claimv)
            if v != np.uint64(0):
                j = partner[s]
                if v > claims[buf, s]:
                    claims[buf, s] = v
                if v > claims[buf, j]:
                    claims[buf, j] = v
        for s in range(S):
            bff_interact(s, buf, soup, partner, claimv, claims, steps, table, sense, selfs, wallet, act,
                         flow, stats, tape)


# --------------------------------------------------------------------- compiling the laws
LAWS = ("mix", "bits", "uniform", "run_tape", "sign", "tier", "max_notional", "equity", "fill",
        "liquidate", "fund", "rint", "to_byte", "energy_byte", "exposure_byte", "read_self", "neighbor",
        "market_site", "interact_site", "bff_claim", "bff_interact")


def target(name, decorate, loops=(), compile_loop=None, special=None):
    """A module holding every law cloned, with the laws' calls bound to each other's clones.

    decorate(fn) makes a clone callable from compiled code for the target (special
    maps a law's name to another decorator); compile_loop compiles the drivers in
    `loops`. The original plain functions stay as they are (and run as Python).
    """
    special = special or {}
    space = dict(globals())
    mod = ModuleType(name)
    for law in LAWS + tuple(loops):
        fn = globals()[law]
        clone = FunctionType(fn.__code__, space, law, fn.__defaults__, fn.__closure__)
        clone.__qualname__, clone.__module__ = fn.__qualname__, __name__
        clone = compile_loop(clone) if law in loops else special.get(law, decorate)(clone)
        space[law] = clone
        setattr(mod, law, clone)
    return mod


# The CPU: the laws inlined into the loops (a call per site per tick would cost more than
# most laws), except those called only once an interaction starts, which run long enough
# not to mind; and no reference counting anywhere (passing arrays around would otherwise
# cost tens of nanoseconds per site per tick in atomic increments).
cpu = target("evotrader.physics.cpu", lambda f: register_jitable(inline="always")(f),
             loops=("cpu_ticks", "cpu_epochs"),
             compile_loop=lambda f: njit(cache=True, _nrt=False)(f),
             special={law: (lambda f: register_jitable(_nrt=False)(f)) for law in ("run_tape", "read_self")})
