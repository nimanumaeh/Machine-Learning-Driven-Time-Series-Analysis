"""The soup's laws as scalar code, compiled for the CPU (numba) and for CUDA (docs/soup.md).

Every function here acts on one site, or on one pair of sites, and compiles
unchanged for both targets. A tick of the world (one row of market data, at the
finest resolution there is) has three phases. Within a phase no two sites write
the same memory, so the CPU loops over the sites one by one and a GPU runs them
all at once, with the same result:

live    each living organism alone: its order fills at the open, liquidation,
        funding and the cost of living (a body's upkeep, and Kleiber's law) follow;
        once its trading since its last meal has paid for its living since (at most
        the cost of one hunger: a famine is no debt), that is a meal. If its energy
        is gone it is doomed.
        Otherwise it senses what is happening now, and its own tape runs on from
        where it stopped (a few instructions a tick: matter that thinks). If it is
        large enough and not hungry it may try to divide (about once a cell
        cycle), and it may wake to meet a neighbor; both more often the more
        energy it has (Kleiber).
birth   the doomed dissolve: their matter returns to nothing and their site to
        space. A site claimed by a dividing parent becomes its child, as a cell
        divides: a copy of the parent's tape (with copying errors) and
        registers, and half of everything it holds, cash, position and margin
        alike. A child takes an empty site, or the site of a weaker organism, or
        of one that is hungry (it has gone too long without a meal): the
        displaced dies, and the child takes over what it held.
meet    two neighbors whose claims won run their joined tape: copying and
        mating, and with bites on, feeding on each other, happen here.

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
STEPS = 1 << 13                            # most instructions an interaction of matter alone runs
OP_LEFT, OP_RIGHT, OP_LEFT1, OP_RIGHT1, OP_DEC, OP_INC, OP_COPY01, OP_COPY10, \
    OP_OPEN, OP_CLOSE, OP_SENSE, OP_SELF, OP_ACT, OP_TRANSFER, OP_GEAR = range(1, 16)
N_SELF = 3                                 # what E reads: energy, position, leverage

# rows of the account matrix: one exact BTCUSDT perpetual account per organism, and what its
# trading has made since its last meal (realized, after fees, funding and liquidations), less
# what living has cost it since (the cost of living and heat), owing at most a hunger's worth
WALLET, Q, ENTRY, MARGIN, FEES, FUNDING, LEV, PENDING, GEAR, MEAL = range(10)
N_ACCT = 10
# rows of the counter matrix: totals at each site, over every organism that lived there
LIQUIDATIONS, TRADES, ORDERS, MEETINGS, THOUGHTS, COPIES, BIRTHS, DEATHS, STARVED = range(9)
N_COUNT = 9
# the organism living at a site: when it was born, its generation, the site's trades and
# births when it was born (so that its own are the difference), and when it last fed (closed
# a trade at a profit)
BORN, GEN, TRADES0, BIRTHS0, FED = range(5)
N_LIFE = 5
# its name, its parent's, and its founder's (the first organism of its line)
ID, PARENT, FOUNDER = range(3)
N_IDS = 3
# where energy went, at each site: taken in by bites, given out by bites, lost (heat and
# digestion), the cost of living, given to children, left behind at death
TAKEN, GIVEN, LOST, METABOLISM, TO_CHILDREN, CARCASS = range(6)
N_FLOW = 6
# columns of the market, one row per tick
E_T, E_OPEN, E_CLOSE, E_HIGH, E_LOW, E_MARK, E_FUNDING = range(7)
N_ENV = 7
# the world's constants
P_STAKE, P_RATE, P_KLEIBER, P_HEAT, P_DIGESTION, P_QUANTUM, P_MAX_LEV, P_STEP, P_FEE, \
    P_HALF_SPREAD, P_MIN_NOTIONAL, P_METABOLISM, P_FLOOR, P_BIRTH_MIN, P_MUTATION, P_UPKEEP, \
    P_DIVIDE_AT, P_CYCLE, P_TAKEOVER, P_STARVE = range(20)
N_FP = 20
I_SEED, I_X, I_Y, I_THINK, I_MEET = range(5)
N_IP = 5
# independent streams of randomness
WAKE, PARTNER, KEY, NOISE, NOISE_AT, BIRTH, BIRTH_KEY, MUTATE, MUTATE_TO, DIVIDE = range(1, 11)

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
    """Interoception: the organism's energy, position and leverage, as three signed bytes."""
    eq = equity(s, mark, acct)
    held = 0.0
    x = 0.0
    if eq > 0.0:
        held = eq
        x = acct[Q, s] * mark / max(eq, 1e-9)
    out[0] = energy_byte(held, fp[P_STAKE])
    out[1] = exposure_byte(x, fp[P_MAX_LEV])
    out[2] = to_byte(127.0 * math.log(max(acct[GEAR, s], 1.0)) / math.log(126.0))


# --------------------------------------------------------------------- space
def neighbor(s, dx, dy, Xs, Y):
    """The site at offset (dx, dy) from s on a band of latitudes: longitude wraps, the poles reflect."""
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


def torus(s, dx, dy, X, Y):
    """The site at offset (dx, dy) from s on a torus of X by Y sites."""
    x = s % X
    y = s // X
    return ((y + dy + Y) % Y) * X + (x + dx + X) % X


# --------------------------------------------------------------------- thought
def think(s, steps, soup, regs, sense, mask, selfs, acct, flow, heat, table, expo, gears):
    """Site s's own matter runs alone for `steps` instructions, from where it stopped.

    The tape is a loop: the instruction pointer and both heads wrap around, so matter
    runs for as long as it lives, a few instructions a tick. S reads what this site
    senses now (sense: the market's bytes this tick; mask: which streams exist here,
    the others read 0), E its own energy, position or leverage. A sets the position it wants
    (a share of its equity, filled at the next open), L its leverage (1x unless it
    chooses more).
    None takes itself as its own argument. Every write costs heat; a site that cannot
    pay cannot write. Returns (instructions executed, bytes copied).
    """
    n = soup.shape[1]
    ip = regs[s, 0]
    h0 = regs[s, 1]
    h1 = regs[s, 2]
    executed = 0
    copies = 0
    for _ in range(steps):
        c = table[soup[s, ip]]
        executed += 1
        if heat > 0.0 and (c == OP_DEC or c == OP_INC or c == OP_COPY01 or c == OP_COPY10
                           or c == OP_SENSE or c == OP_SELF):
            if acct[WALLET, s] < heat:                     # no energy, no writing
                ip += 1
                if ip == n:
                    ip = 0
                continue
            acct[WALLET, s] -= heat
            acct[MEAL, s] -= heat
            flow[s, LOST] += heat
        if c == OP_LEFT:                                   # (wrapping without a division)
            h0 = h0 - 1 if h0 > 0 else n - 1
        elif c == OP_RIGHT:
            h0 = h0 + 1 if h0 < n - 1 else 0
        elif c == OP_LEFT1:
            h1 = h1 - 1 if h1 > 0 else n - 1
        elif c == OP_RIGHT1:
            h1 = h1 + 1 if h1 < n - 1 else 0
        elif c == OP_DEC:
            soup[s, h0] = (np.int64(soup[s, h0]) + 255) & 255
        elif c == OP_INC:
            soup[s, h0] = (np.int64(soup[s, h0]) + 1) & 255
        elif c == OP_COPY01:
            soup[s, h1] = soup[s, h0]
            copies += 1
        elif c == OP_COPY10:
            soup[s, h0] = soup[s, h1]
            copies += 1
        elif c == OP_OPEN:
            if soup[s, h0] == 0:                           # skip to after the matching ]
                depth = 1
                j = ip
                k = 0
                while k < n - 1 and depth > 0:
                    j = j + 1 if j < n - 1 else 0
                    k += 1
                    cj = table[soup[s, j]]
                    if cj == OP_OPEN:
                        depth += 1
                    elif cj == OP_CLOSE:
                        depth -= 1
                if depth == 0:
                    ip = j
        elif c == OP_CLOSE:
            if soup[s, h0] != 0:                           # back to after the matching [
                depth = 1
                j = ip
                k = 0
                while k < n - 1 and depth > 0:
                    j = j - 1 if j > 0 else n - 1
                    k += 1
                    cj = table[soup[s, j]]
                    if cj == OP_CLOSE:
                        depth += 1
                    elif cj == OP_OPEN:
                        depth -= 1
                if depth == 0:
                    ip = j
        elif c == OP_SENSE:
            k = soup[s, h0] % sense.shape[0]
            soup[s, h0] = sense[k] if mask[s, k] != 0 else 0
        elif c == OP_SELF:
            soup[s, h0] = selfs[soup[s, h0] % N_SELF]
        elif c == OP_ACT:
            v = np.int64(soup[s, h0]) if h0 != ip else 0
            acct[PENDING, s] = expo[v]
        elif c == OP_GEAR:
            v = np.int64(soup[s, h0]) if h0 != ip else 0
            acct[GEAR, s] = gears[v]
        ip += 1
        if ip == n:
            ip = 0
    regs[s, 0] = ip
    regs[s, 1] = h0
    regs[s, 2] = h1
    return executed, copies


def meet_tape(tape, steps, sense, mask, self_a, self_b, acct, a, b, act, flow, heat, digestion,
              quantum, table, gears):
    """Two organisms meet: their joined tape (a's first) runs once, from its start.

    As in BFF, the program stops at either end of the tape, at an unmatched bracket,
    or after `steps` instructions. Instructions act for the organism in whose half
    they sit: A records the position it wants in act (as a byte), L sets its
    leverage, T moves a bite of energy between the two (positive: takes from the
    other, negative: gives; only `digestion` of it arrives), and copying between
    the heads moves matter from one organism to the other.
    Returns (instructions executed, bytes copied).
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
            if acct[WALLET, me] < heat:
                ip += 1
                continue
            acct[WALLET, me] -= heat
            acct[MEAL, me] -= heat
            flow[me, LOST] += heat
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
            me = a if ip < half else b
            k = tape[h0] % sense.shape[0]
            tape[h0] = sense[k] if mask[me, k] != 0 else 0
        elif c == OP_SELF:
            if ip < half:
                tape[h0] = self_a[tape[h0] % N_SELF]
            else:
                tape[h0] = self_b[tape[h0] % N_SELF]
        elif c == OP_ACT:
            v = np.int64(tape[h0]) if h0 != ip else 0      # never its own argument
            if ip < half:
                act[a] = v
            else:
                act[b] = v
        elif c == OP_GEAR:
            v = np.int64(tape[h0]) if h0 != ip else 0
            if ip < half:
                acct[GEAR, a] = gears[v]
            else:
                acct[GEAR, b] = gears[v]
        elif c == OP_TRANSFER:
            me = a if ip < half else b
            other = b if ip < half else a
            v = np.int64(tape[h0]) if h0 != ip else 0
            if v == 0:
                pass
            elif v < 128:                                  # positive: take
                amount = min(max(acct[WALLET, other], 0.0), quantum * v / 127.0)
                acct[WALLET, other] -= amount
                acct[WALLET, me] += amount * digestion
                flow[me, TAKEN] += amount * digestion
                flow[other, GIVEN] += amount
                flow[me, LOST] += amount * (1.0 - digestion)
            else:                                          # negative: give
                amount = min(max(acct[WALLET, me], 0.0), quantum * (256.0 - v) / 128.0)
                acct[WALLET, me] -= amount
                acct[WALLET, other] += amount * digestion
                flow[me, GIVEN] += amount
                flow[other, TAKEN] += amount * digestion
                flow[me, LOST] += amount * (1.0 - digestion)
        ip += 1
    return executed, copies


# --------------------------------------------------------------------- one tick
def hungry(s, tick, life, fp):
    """Has the organism at s gone longer than the world allows without closing a trade at a profit?"""
    return fp[P_STARVE] > 0.0 and tick - life[FED, s] > fp[P_STARVE]


def live_site(s, tick, r, env, band, acct, counts, flow, soup, regs, mask, alive, doomed, life,
              partner, claimv, target, birthv, divide, offsets, noise_cdf, expo, gears, table, br, fp,
              ip_, selfs):
    """Phase 1 of a tick for site s. It writes only s's own state, and reads whether its
    neighbors are alive (which no one changes in this phase)."""
    partner[s] = -1
    claimv[s] = np.uint64(0)
    target[s] = -1
    birthv[s] = np.uint64(0)
    doomed[s] = 0
    if alive[s] == 0:
        return
    seed = ip_[I_SEED]
    # the order it placed fills at this tick's open, at the leverage it chose
    x = acct[PENDING, s]
    if x == x:
        acct[PENDING, s] = np.nan
        price = env[r, E_OPEN]
        lv = acct[GEAR, s]
        e = equity(s, price, acct)
        if not (e > 0.0):
            e = 0.0
        cap = max_notional(lv, br)
        q0 = acct[Q, s]
        e0 = acct[ENTRY, s]
        paid = fill(s, min(max(x * lv * e, -cap), cap) / price, price, lv, acct, counts, fp[P_STEP],
                    fp[P_FEE], fp[P_HALF_SPREAD], fp[P_MIN_NOTIONAL])
        acct[LEV, s] = lv
        counts[ORDERS, s] += 1
        # what closing all or part of a position realized, after fees
        q1 = acct[Q, s]
        closed = 0.0
        if q0 != 0.0:
            if q1 == 0.0 or sign(q1) != sign(q0):
                closed = abs(q0)
            elif abs(q1) < abs(q0):
                closed = abs(q0) - abs(q1)
        acct[MEAL, s] += closed * (price - sign(q0) * fp[P_HALF_SPREAD] - e0) * sign(q0) - paid
    m0 = acct[MARGIN, s]
    if acct[Q, s] != 0.0:
        liquidate(s, env[r, E_HIGH], env[r, E_LOW], acct, counts, br)
        if acct[Q, s] == 0.0:
            acct[MEAL, s] -= max(m0, 0.0)                 # the margin is lost
    mark = env[r, E_MARK]
    f = env[r, E_FUNDING]
    if f != 0.0 and f == f:
        acct[MEAL, s] -= acct[Q, s] * mark * f
        fund(s, f, mark, acct)
    # living costs energy: the upkeep of a body, and Kleiber's three quarters of what it holds
    e = equity(s, mark, acct)
    if e > 0.0:
        cost = min(fp[P_UPKEEP] + fp[P_METABOLISM] * math.sqrt(e) * math.sqrt(math.sqrt(e)), e)
        acct[WALLET, s] -= cost
        acct[MEAL, s] -= cost
        flow[s, METABOLISM] += cost
        e = e - cost
        # what it must make to eat is never more than living costs for one hunger
        if fp[P_STARVE] > 0.0:
            acct[MEAL, s] = max(acct[MEAL, s], -cost * fp[P_STARVE])
    # once its trading since its last meal has paid for its living since, it has eaten
    if acct[MEAL, s] > 0.0:
        life[FED, s] = tick
        acct[MEAL, s] = 0.0
    if not (e > fp[P_FLOOR]):
        doomed[s] = 1                                      # nothing left to live on
        return
    # a few of its bytes may flip
    u = uniform(seed, tick, NOISE, s)
    flips = 0
    while flips < noise_cdf.shape[0] and u > noise_cdf[flips]:
        flips += 1
    for j in range(flips):
        z = bits(seed, tick, NOISE_AT, s * 64 + j)
        soup[s, np.int64(z % np.uint64(soup.shape[1]))] = np.uint8((z >> np.uint64(32)) & np.uint64(255))
    # its matter thinks
    read_self(s, mark, acct, fp, selfs)
    ex, cp = think(s, ip_[I_THINK], soup, regs, band[r, 0], mask, selfs, acct, flow, fp[P_HEAT], table, expo,
                   gears)
    counts[THOUGHTS, s] += ex
    counts[COPIES, s] += cp
    n_off = offsets.shape[0]
    if n_off == 0:
        return
    X = ip_[I_X]
    Y = ip_[I_Y]
    # the more energy it has, the more often it acts on its neighbors (Kleiber)
    ratio = e / fp[P_STAKE]
    kl = fp[P_KLEIBER]
    if kl == 0.75:
        w = math.sqrt(ratio) * math.sqrt(math.sqrt(ratio))
    elif kl == 1.0:
        w = ratio
    else:
        w = ratio ** kl
    # a body large enough to divide, and not hungry, tries to about once a cell cycle: in
    # half, into a site next to it (empty, or held by someone weaker or hungry: phase 2)
    divide[s] = 0.0
    if e >= fp[P_DIVIDE_AT] and not hungry(s, tick, life, fp) and \
            uniform(seed, tick, DIVIDE, s) < fp[P_CYCLE] * w:
        divide[s] = 0.5
        z = bits(seed, tick, BIRTH, s)
        o = np.int64(z % np.uint64(n_off))
        target[s] = torus(s, offsets[o, 0], offsets[o, 1], X, Y)
        birthv[s] = ((bits(seed, tick, BIRTH_KEY, s) >> np.uint64(32)) << np.uint64(32)) | np.uint64(s + 1)
    # and it may wake to meet a living neighbor
    if uniform(seed, tick, WAKE, s) < fp[P_RATE] * w:
        z = bits(seed, tick, PARTNER, s)
        o = np.int64(z % np.uint64(n_off))
        j = torus(s, offsets[o, 0], offsets[o, 1], X, Y)
        if alive[j] != 0:
            partner[s] = j
            claimv[s] = ((bits(seed, tick, KEY, s) >> np.uint64(32)) << np.uint64(32)) | np.uint64(s + 1)


def die(s, mark, acct, counts, flow, soup, regs, alive, doomed, ids):
    """Death: the organism's matter returns to nothing, its site to space; what it held is lost."""
    flow[s, CARCASS] += equity(s, mark, acct)              # (negative: a debt forgiven)
    acct[WALLET, s] = 0.0
    acct[Q, s] = 0.0
    acct[ENTRY, s] = 0.0
    acct[MARGIN, s] = 0.0
    acct[PENDING, s] = np.nan
    acct[GEAR, s] = 1.0
    acct[LEV, s] = 1.0
    acct[MEAL, s] = 0.0
    for k in range(soup.shape[1]):
        soup[s, k] = 0
    regs[s, 0] = 0
    regs[s, 1] = 0
    regs[s, 2] = 0
    ids[ID, s] = np.uint64(0)
    ids[PARENT, s] = np.uint64(0)
    ids[FOUNDER, s] = np.uint64(0)
    alive[s] = 0
    doomed[s] = 0
    counts[DEATHS, s] += 1


def birth_site(s, tick, buf, r, env, acct, counts, flow, soup, regs, alive, doomed, births, birthv,
               divide, ids, life, fp, ip_):
    """Phase 2 for site s: the doomed die; a site claimed by a parent is born into.

    A child takes an empty site, or the site of an organism that is hungry or holds
    less than a share (`takeover`) of what its parent holds: the displaced dies, and
    the child takes over what it held (nothing is lost: space changes hands, energy
    does not vanish). An organism dividing this tick holds its ground. A parent claims
    one site a tick, and only that site's thread reads or writes the parent here."""
    births[1 - buf, s] = np.uint64(0)                     # clean for the next tick
    mark = env[r, E_MARK]
    if alive[s] != 0 and doomed[s] != 0:
        die(s, mark, acct, counts, flow, soup, regs, alive, doomed, ids)
    v = births[buf, s]
    if v == np.uint64(0):
        return
    p = np.int64(v & np.uint64(0xFFFFFFFF)) - 1
    if birthv[p] != v:
        return
    share = divide[p]
    pe = equity(p, mark, acct)
    give = share * pe
    if not (give >= fp[P_BIRTH_MIN]):
        return                                             # too little to live on: no child
    eaten = 0.0
    if alive[s] != 0:
        occupant = equity(s, mark, acct)
        weak = hungry(s, tick, life, fp)
        if birthv[s] != np.uint64(0) or not (weak or occupant < fp[P_TAKEOVER] * pe):
            return                                         # it holds its ground
        if weak:
            counts[STARVED, s] += 1                        # it died hungry
        if occupant > 0.0:                                 # the weaker is displaced; the child takes over
            eaten = occupant                               # what it held (closed at the mark)
            flow[s, CARCASS] -= occupant                   # (die() counts it as left behind)
        die(s, mark, acct, counts, flow, soup, regs, alive, doomed, ids)
    # the child takes its share of everything the parent holds: its position in whole lots
    # (with their margin), and cash for the rest
    qp = acct[Q, p]
    q = sign(qp) * math.floor(abs(qp) * share / fp[P_STEP] + _EPS) * fp[P_STEP]
    m = acct[MARGIN, p] * (q / qp) if qp != 0.0 else 0.0
    w = give - (m + q * (mark - acct[ENTRY, p]))
    meal = share * acct[MEAL, p]
    acct[WALLET, p] -= w
    acct[Q, p] -= q
    acct[MARGIN, p] -= m
    acct[MEAL, p] -= meal
    flow[p, TO_CHILDREN] += give
    seed = ip_[I_SEED]
    mu = fp[P_MUTATION]
    n = soup.shape[1]
    for k in range(n):                                     # a copy, with copying errors
        b = soup[p, k]
        if mu > 0.0 and uniform(seed, tick, MUTATE, s * 256 + k) < mu:
            b = np.uint8(bits(seed, tick, MUTATE_TO, s * 256 + k) >> np.uint64(56))
        soup[s, k] = b
    regs[s, 0] = regs[p, 0]                                # and goes on thinking where the parent was
    regs[s, 1] = regs[p, 1]
    regs[s, 2] = regs[p, 2]
    acct[WALLET, s] = w + eaten
    flow[s, TAKEN] += eaten
    acct[Q, s] = q
    acct[ENTRY, s] = acct[ENTRY, p] if q != 0.0 else 0.0
    acct[MARGIN, s] = m
    acct[PENDING, s] = acct[PENDING, p]
    acct[GEAR, s] = acct[GEAR, p]
    acct[LEV, s] = acct[LEV, p]
    acct[MEAL, s] = meal
    ids[ID, s] = (np.uint64(tick) << np.uint64(24)) | np.uint64(s)
    ids[PARENT, s] = ids[ID, p]
    ids[FOUNDER, s] = ids[FOUNDER, p]
    life[BORN, s] = tick
    life[GEN, s] = life[GEN, p] + 1
    life[TRADES0, s] = counts[TRADES, s]
    life[BIRTHS0, s] = counts[BIRTHS, s]
    life[FED, s] = life[FED, p]                            # as hungry as its parent
    alive[s] = 1
    counts[BIRTHS, p] += 1


def meet_site(s, tick, buf, r, env, band, acct, counts, flow, soup, mask, act, partner, claimv, claims, alive,
              life, table, expo, gears, fp, ip_, tape, sa, sb):
    """Phase 3: the meeting site s started, if both its claims won and both are still alive."""
    claims[1 - buf, s] = np.uint64(0)                     # clean for the next tick
    j = partner[s]
    if j < 0:
        return
    v = claimv[s]
    if claims[buf, s] != v or claims[buf, j] != v:
        return                                             # crowded out this tick
    if alive[s] == 0 or alive[j] == 0:
        return
    if life[BORN, j] == tick:
        return                                             # its partner was displaced by a newborn
    mark = env[r, E_MARK]
    read_self(s, mark, acct, fp, sa)
    read_self(j, mark, acct, fp, sb)
    n = soup.shape[1]
    for k in range(n):
        tape[k] = soup[s, k]
        tape[n + k] = soup[j, k]
    act[s] = -1
    act[j] = -1
    e, c = meet_tape(tape, ip_[I_MEET], band[r, 0], mask, sa, sb, acct, s, j, act, flow, fp[P_HEAT],
                     fp[P_DIGESTION], fp[P_QUANTUM], table, gears)
    for k in range(n):
        soup[s, k] = tape[k]
        soup[j, k] = tape[n + k]
    if act[s] >= 0:
        acct[PENDING, s] = expo[act[s]]
        act[s] = -1
    if act[j] >= 0:
        acct[PENDING, j] = expo[act[j]]
        act[j] = -1
    counts[MEETINGS, s] += 1
    counts[THOUGHTS, s] += e
    counts[COPIES, s] += c


def cpu_life(r0, r1, tick0, env, band, acct, counts, flow, soup, regs, mask, alive, doomed,
             partner, claimv, claims, target, birthv, births, divide, act, ids, life, offsets, noise_cdf,
             expo, gears, table, br, fp, ip_, tape, sa, sb, selfs):
    """Rows r0..r1 of env, one tick each, on the CPU (tick number tick0 + r).

    tape (2L,), sa, sb and selfs (N_SELF,) uint8: work space.
    """
    S = soup.shape[0]
    for r in range(r0, r1):
        tick = tick0 + r
        buf = tick & 1
        for s in range(S):
            live_site(s, tick, r, env, band, acct, counts, flow, soup, regs, mask, alive, doomed, life,
                      partner, claimv, target, birthv, divide, offsets, noise_cdf, expo, gears, table, br,
                      fp, ip_, selfs)
            v = birthv[s]
            if v != np.uint64(0):
                j = target[s]
                if v > births[buf, j]:
                    births[buf, j] = v
            v = claimv[s]
            if v != np.uint64(0):
                j = partner[s]
                if v > claims[buf, s]:
                    claims[buf, s] = v
                if v > claims[buf, j]:
                    claims[buf, j] = v
        for s in range(S):
            birth_site(s, tick, buf, r, env, acct, counts, flow, soup, regs, alive, doomed, births, birthv,
                       divide, ids, life, fp, ip_)
        for s in range(S):
            meet_site(s, tick, buf, r, env, band, acct, counts, flow, soup, mask, act, partner, claimv, claims,
                      alive, life, table, expo, gears, fp, ip_, tape, sa, sb)


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
LAWS = ("mix", "bits", "uniform", "run_tape", "sign", "tier", "max_notional", "equity", "fill", "hungry",
        "liquidate", "fund", "rint", "to_byte", "energy_byte", "exposure_byte", "read_self", "neighbor",
        "torus", "think", "meet_tape", "live_site", "die", "birth_site", "meet_site", "bff_claim",
        "bff_interact")


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


# The CPU: the small laws inlined into the loops (a call per site per tick would cost more
# than they do); the interpreters and whole phases called, since they run long enough not
# to mind; and no reference counting anywhere (passing arrays around would otherwise cost
# tens of nanoseconds per site per tick in atomic increments).
_CALLED = ("run_tape", "read_self", "think", "meet_tape")
cpu = target("evotrader.physics.cpu", lambda f: register_jitable(inline="always")(f),
             loops=("cpu_life", "cpu_epochs"),
             compile_loop=lambda f: njit(cache=True, _nrt=False)(f),
             special={law: (lambda f: register_jitable(_nrt=False)(f)) for law in _CALLED})
