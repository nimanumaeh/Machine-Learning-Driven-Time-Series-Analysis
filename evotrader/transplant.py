"""Reverse engineering by transplant: an organism is code, so the organism is the strategy.

Lift any matter out of a soup (one site's tape, a lineage, a whole region), plant a
colony of it in a sandbox world with the same physics, the same latitude and the
same senses its birthplace offered, switch evolution off, and let it live through
market data it has never seen. Its positions over time are its strategy; its P&L
there is the test; how its positions move with what it senses says what drives it.
This works however strange the matter has become, because nothing about it has
to be understood to be run.
"""

import numpy as np

from .data import C
from .planet import Planet
from .selfmade import RECEPTORS
from .soup import Soup


def organisms(world, k=10, min_energy=None):
    """The k richest sites: (site, energy, place x, latitude y, tape)."""
    eq = world.equity()
    order = np.argsort(-eq)
    out = []
    for s in order[:k]:
        if min_energy is not None and eq[s] < min_energy:
            break
        out.append((int(s), float(eq[s]), int(world.site_place[s]), int(world.site_y[s]),
                    world.soup[s].copy()))
    return out


def sandbox(world, x, y, matter, sites=64, seed=0):
    """A world of one place at latitude y with the senses of place (x, y) and the same physics."""
    pl0 = world.planet
    pl = Planet(width=1, seed=seed, taus=[int(pl0.taus[y])], step_s=pl0.step_s)
    pl.avail[0, 0] = pl0.avail[x, y]
    rate = world.E / world.S                                   # interactions per site per row
    soup = Soup(world.cfg, pl, per_place=sites, seed=seed, interactions=max(1, int(round(rate * sites))),
                noise=0.0, floor=0.0, steps=world.steps, matter=_fill(matter, sites),
                kleiber=world.kleiber, max_exposure=world.max_exposure, heat=world.heat,
                digestion=world.digestion, quantum=world.quantum)
    soup.table = world.table
    return pl, soup


def _fill(matter, sites):
    m = np.atleast_2d(np.asarray(matter, np.uint8))
    return m[np.arange(sites) % len(m)]


def live(pl, soup, rows, every=60):
    """Run the sandbox; sample its total energy, mean position and senses every `every` rows."""
    t, energy, position, senses, price = [], [], [], [], []
    for k, row in enumerate(rows):
        soup.step(row, pl.step(row))
        if k % every == 0:
            t.append(pl.t)
            energy.append(float(soup.equity().sum()))
            position.append(float(soup.exposure().mean()))
            senses.append(pl.sense_bytes[0].copy())
            price.append(float(row[C["close"]]))
    return {"t": np.array(t), "energy": np.array(energy), "position": np.array(position),
            "senses": np.array(senses), "price": np.array(price),
            "fees": float(soup.acct.fees.sum()), "funding": float(soup.acct.funding.sum()),
            "heat": float(soup.flow[:, 2].sum()), "orders": soup.counts["orders"],
            "start": soup.stake * soup.S}


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
    ret = e[-1] / result["start"] - 1
    hold = p[-1] / p[0] - 1
    trades = np.abs(pos).mean() > 0.01 and result["orders"] > 0
    sharpe = None
    if trades:
        r = np.diff(np.log(np.maximum(e, 1e-9)))
        years = max((result["t"][-1] - result["t"][0]) / 31_557_600, 1e-9)
        sharpe = float(r.mean() / r.std() * np.sqrt(len(r) / years)) if r.std() > 0 else 0.0
    start = result["start"]
    return {"return": float(ret), "hold_btc": float(hold), "sharpe": sharpe, "trades": bool(trades),
            "mean_abs_position": float(np.abs(pos).mean()), "orders": result["orders"],
            "fees": result["fees"] / start, "funding": result["funding"] / start,
            "heat": result["heat"] / start,
            "market": float(ret + (result["fees"] + result["funding"] + result["heat"]) / start)}


def evaluate(world, rows, k=10, sites=64, baseline=5, seed=0, log=print):
    """Transplant the k richest organisms and `baseline` random tapes; compare on `rows`."""
    rows = list(rows)
    report = []
    rng = np.random.default_rng(seed)
    picks = [("organism", s, e, x, y, tape) for s, e, x, y, tape in organisms(world, k)]
    for j in range(baseline):
        s = int(rng.integers(world.S))
        picks.append(("random", s, 0.0, int(world.site_place[s]), int(world.site_y[s]),
                      rng.integers(0, 256, world.soup.shape[1], dtype=np.uint8)))
    for kind, s, e, x, y, tape in picks:
        pl, soup = sandbox(world, x, y, tape, sites=sites, seed=seed)
        res = live(pl, soup, rows)
        row = {"kind": kind, "site": s, "energy_in_world": e, "place": x, "latitude": y,
               "band_s": int(world.planet.taus[y]), **summary(res), "drivers": drivers(res)}
        report.append(row)
        sharpe = f"{row['sharpe']:+5.2f}" if row["sharpe"] is not None else "  -  "
        log(f"  {kind:8s} site {s:5d} lat {row['band_s']:>7d}s  return {row['return']:+7.1%} "
            f"(market {row['market']:+6.1%} fees {-row['fees']:+5.1%} heat {-row['heat']:+5.1%})  "
            f"hold {row['hold_btc']:+6.1%}  sharpe {sharpe}  |pos| {row['mean_abs_position']:.2f}  "
            f"orders {row['orders']:6d}  " + ", ".join(f"{n} {c:+.2f}" for n, c in row["drivers"][:2]))
    return report
