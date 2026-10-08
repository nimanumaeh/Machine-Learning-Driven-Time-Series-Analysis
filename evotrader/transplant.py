"""Reverse engineering by transplant: an organism is code, so the organism is the strategy.

Lift an organism out of a soup (its tape, where its thought had got to, the leverage
it had chosen), plant it alone in a sandbox world with the same physics and the
senses its birthplace offered, switch evolution off (no noise, no mutation, no
meetings, no division) and, unless asked, the cost of living too, and let it live
through market data it has never seen. Its positions over time are its strategy;
its P&L there is the test; how its positions move with what it senses says what
drives it. This works however strange the matter has become, because nothing
about it has to be understood to be run.
"""

import numpy as np

from .data import C
from .physics import BORN, FEES, FOUNDER, FUNDING, GEAR, GEN, LOST, METABOLISM, TRADES, TRADES0
from .selfmade import RECEPTORS
from .soup import Soup


def organisms(world, k=10, min_energy=None):
    """The k richest living organisms: (site, energy, x, y, tape)."""
    eq = np.where(world.alive.astype(bool), world.equity(), -np.inf)
    out = []
    for s in np.argsort(-eq, kind="stable")[:k]:
        if not np.isfinite(eq[s]) or (min_energy is not None and eq[s] < min_energy):
            break
        out.append((int(s), float(eq[s]), int(world.site_x[s]), int(world.site_y[s]), world.soup[s].copy()))
    return out


def sandbox(world, s, matter=None, sites=1, seed=0, living=False):
    """`sites` copies of an organism alone, with the senses of site s and the same physics.

    By default the organism is the one living at site s, as it is: its tape, its
    registers (where its thought had got to) and the leverage it had chosen. Given
    `matter`, that matter is planted instead, starting to think at its first byte.
    Evolution is off, and so (unless `living`) is the cost of living: what it earns
    or loses is its trading.
    """
    own = matter is None
    tape = world.soup[s] if own else matter
    sb = Soup(world.cfg, width=sites, height=1, step_s=world.planet.step_s, seed=seed, occupancy=1.0,
              think=world.think_steps, meet=world.meet_steps, meetings=0.0,
              metabolism_days=world.metabolism_days if living else float("inf"),
              upkeep=world.upkeep if living else 0.0, floor=world.floor, birth_min=world.birth_min,
              divide_at=float("inf"), mutation=0.0, noise=0.0, heat=world.heat, digestion=world.digestion,
              quantum=world.quantum, kleiber=world.kleiber, matter=_fill(tape, sites), layout=world.layout)
    sb.mask[:] = world.mask[s]
    if own:
        sb.regs[:] = world.regs[s]
        sb.acct[GEAR] = world.acct[GEAR, s]
    return sb


def _fill(matter, sites):
    m = np.atleast_2d(np.asarray(matter, np.uint8))
    return m[np.arange(sites) % len(m)]


def live(soup, rows, every=60):
    """Run the sandbox; sample its energy, mean position and senses every `every` rows."""
    rows = np.asarray(rows, np.float64)
    t, energy, position, senses, price = [], [], [], [], []

    def sample(row):
        alive = soup.alive.astype(bool)
        t.append(soup.t)
        energy.append(float(soup.equity()[alive].sum()))
        position.append(float(soup.exposure()[alive].mean()) if alive.any() else 0.0)
        senses.append(np.where(soup.mask[0] != 0, soup.sensed, 0).astype(np.uint8))   # what it perceives
        price.append(float(row[C["close"]]))

    start = 0
    for k in range(0, len(rows), every):
        soup.advance(rows[start:k + 1])
        start = k + 1
        sample(rows[k])
    if start < len(rows):
        soup.advance(rows[start:])
        sample(rows[-1])
    n = soup.counts
    return {"t": np.array(t), "energy": np.array(energy), "position": np.array(position),
            "senses": np.array(senses), "price": np.array(price),
            "fees": float(soup.acct[FEES].sum()), "funding": float(soup.acct[FUNDING].sum()),
            "heat": float(soup.flow[:, LOST].sum()), "living": float(soup.flow[:, METABOLISM].sum()),
            "orders": n["orders"], "trades": n["trades"], "liquidations": n["liquidations"],
            "died": n["deaths"] > 0, "start": soup.injected}


def drivers(result, top=5):
    """Which senses its position moves with (correlation of position with each sense byte)."""
    pos, sen = result["position"], result["senses"].astype(np.int64)
    sen = np.where(sen >= 128, sen - 256, sen).astype(float)
    if pos.std() == 0:
        return []
    names = [f"{r} level" for r in RECEPTORS] + [f"{r} change" for r in RECEPTORS]
    out = []
    for j in range(sen.shape[1]):
        if sen[:, j].std() > 0:
            out.append((names[j], float(np.corrcoef(pos, sen[:, j])[0, 1])))
    return sorted(out, key=lambda kv: -abs(kv[1]))[:top]


def summary(result):
    """Its return on unseen data, holding BTC over the same span, and where the money went."""
    e, p, pos = result["energy"], result["price"], result["position"]
    start = result["start"]
    ret = e[-1] / start - 1
    hold = p[-1] / p[0] - 1
    trades = np.abs(pos).mean() > 0.01 and result["orders"] > 0
    sharpe = None
    if trades:
        r = np.diff(np.log(np.maximum(e, 1e-9)))
        years = max((result["t"][-1] - result["t"][0]) / 31_557_600, 1e-9)
        sharpe = float(r.mean() / r.std() * np.sqrt(len(r) / years)) if r.std() > 0 else 0.0
    costs = result["fees"] + result["funding"] + result["heat"] + result["living"]
    return {"return": float(ret), "hold_btc": float(hold), "sharpe": sharpe, "trades": bool(trades),
            "mean_abs_position": float(np.abs(pos).mean()), "orders": result["orders"],
            "liquidations": result["liquidations"], "died": result["died"],
            "fees": result["fees"] / start, "funding": result["funding"] / start,
            "heat": result["heat"] / start, "living": result["living"] / start,
            "market": float(ret + costs / start)}


def evaluate(world, rows, k=10, baseline=5, seed=0, warmup=256, living=False, log=print):
    """Transplant the k richest organisms and `baseline` random tapes; compare on `rows`.

    The first `warmup` rows only let the sandbox's senses learn what is ordinary
    (as they had in the world); the test is the rest.
    """
    rows = rows if isinstance(rows, np.ndarray) else np.array(list(rows), np.float64)
    report = []
    rng = np.random.default_rng(seed)
    picks = [("organism", s, e, None) for s, e, *_ in organisms(world, k)]
    alive = np.nonzero(world.alive)[0]
    for _ in range(baseline):
        s = int(rng.choice(alive)) if len(alive) else int(rng.integers(world.S))
        picks.append(("random", s, 0.0, rng.integers(0, 256, world.soup.shape[1], dtype=np.uint8)))
    hold_s = world.holding_s()
    for kind, s, e, tape in picks:
        sb = sandbox(world, s, tape, seed=seed, living=living)
        sb.advance(rows[:warmup])
        sb.injected = float(sb.equity()[sb.alive.astype(bool)].sum())     # the test starts here
        res = live(sb, rows[warmup:])
        own = kind == "organism"
        row = {"kind": kind, "site": s, "energy_in_world": e, "x": int(world.site_x[s]),
               "y": int(world.site_y[s]),
               "generation": int(world.life[GEN, s]) if own else None,
               "founder": int(world.ids[FOUNDER, s]) if own else None,
               "age_days": float((world.tick - world.life[BORN, s]) * world.planet.step_s / 86400) if own else None,
               "trades_in_world": int(world.count[TRADES, s] - world.life[TRADES0, s]) if own else None,
               "holding_s_in_world": float(hold_s[s]) if own else None,
               "gear": float(world.acct[GEAR, s]) if own else 1.0,
               **summary(res), "drivers": drivers(res)}
        report.append(row)
        sharpe = f"{row['sharpe']:+5.2f}" if row["sharpe"] is not None else "  -  "
        who = (f"gen {row['generation']:3d} {row['age_days']:6.0f}d {row['gear']:4.0f}x" if own
               else " " * 21)
        log(f"  {kind:8s} site {s:5d} {who}  return {row['return']:+7.1%} "
            f"(market {row['market']:+6.1%} fees {-row['fees']:+5.1%} heat {-row['heat']:+5.1%})  "
            f"hold {row['hold_btc']:+6.1%}  sharpe {sharpe}  |pos| {row['mean_abs_position']:.2f}  "
            f"orders {row['orders']:6d}{'  died' if row['died'] else ''}  "
            + ", ".join(f"{n} {c:+.2f}" for n, c in row["drivers"][:2]))
    return report
