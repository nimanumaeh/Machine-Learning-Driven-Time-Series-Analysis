"""What happened on a planet: populations, money, and what its organisms found relevant.

    python experiments/planet_check.py runs/planet-planted runs/planet-null

For each run: the population story (births against newcomers), the money
(planet net, before costs, against holding BTC), conscious episodes early and
late, the oldest observatories and largest families. Then the relevance test:
how much of the organisms' salience lands on each kind of stream, against how
much of their perception they devote to it. A ratio above 1 means a stream
matters to them more than its share of their organs; it should rise for the
crowding streams in a planted world and stay near 1 in a null one.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evotrader.life import K                                     # noqa: E402
from evotrader.planet import GROWTH, TAU                         # noqa: E402
from evotrader.planet_run import load, read_census               # noqa: E402
from evotrader.selfmade import RECEPTORS, ZERO                   # noqa: E402

KINDS = {
    "price": ("close", "high", "low", "mark"),
    "volume": ("volume", "quote_volume", "trades"),
    "order flow": ("taker_buy",),
    "crowding": ("premium", "funding", "open_interest", "open_interest_value", "top_account_ls",
                 "top_position_ls", "account_ls", "taker_ls", "spot"),
    "spot flow": ("spot_volume", "spot_taker_buy"),
    "clocks": ("day_sin", "day_cos", "week_sin", "week_cos", "cycle_sin", "cycle_cos"),
}
KIND_OF = {RECEPTORS.index(r): k for k, rs in KINDS.items() for r in rs}
BANDS = ["1s", "5s", "30s", "2m", "10m", "1h", "6h", "1d"]


def kind(channel):
    if channel < ZERO:
        return KIND_OF.get(int(channel), "other")
    if channel == ZERO:
        return None
    return "markers" if channel >= GROWTH else "observatories"


def attention(life, live):
    """Salience and plain organ share by kind of stream, per band: {band: {kind: (sal, base)}}."""
    out = {}
    for y in range(len(TAU)):
        m = live[life.y[live] == y]
        sal, base = {}, {}
        for i in m:
            for k in range(K):
                a, b = life.prog[i, k, 0], life.prog[i, k, 1]
                parts = [c for c in (kind(a), kind(b)) if c is not None]
                for c in parts:
                    sal[c] = sal.get(c, 0.0) + life.salience[i, k] / len(parts)
                    base[c] = base.get(c, 0.0) + (1.0 / K) / len(parts)
        n = max(len(m), 1)
        out[y] = {c: (sal.get(c, 0.0) / n, base[c] / n) for c in base}
    return out


def learned(life, live):
    """Organisms whose values carry any evidence at all (salience is not flat)."""
    return live[np.abs(life.salience[live] - 1.0 / K).max(1) > 1e-9]


def report(run):
    frames = read_census(run)
    planet, life = load(os.path.join(run, "planet.pkl"))
    live = np.nonzero(life.alive)[0]
    last = frames[-1]
    c = last["counts"]
    days = (last["t"] - frames[0]["t"]) / 86400 + 0.5 / 24
    print(f"\n=== {run}: {days:.1f} days, BTC {last['price'] / last['price0'] - 1:+.1%}")
    print(f"population {last['population']}  by band {dict(zip(BANDS, last['by_band']))}")
    print(f"born {c['born']}  died {c['died']}  displaced {c['displaced']}  newcomers {c['seeded']}  "
          f"max generation {last['max_generation']}  migrations {c['migrated']}  built {c['built']}")
    gross = last["net"] + last["fees"] + last["funding"]
    print(f"money: net {last['net']:+,.0f}  before fees and funding {gross:+,.0f}  fees {last['fees']:,.0f}  "
          f"funding {last['funding']:+,.0f}  liquidations {last['liquidations']}  put in {last['injected']:,.0f}")
    gens = life.gen[live]
    eq = life.equity(live) / life.ref[live]
    for name, m in (("first generation", gens == 0), ("born on the planet", gens > 0)):
        if m.any():
            print(f"  {name:>18}: {m.sum():4d} alive, median wealth since last division {np.median(eq[m]) - 1:+.1%}")
    q = max(1, len(frames) // 4)
    early = np.mean([f["conscious"] for f in frames[:q]], 0)
    late = np.mean([f["conscious"] for f in frames[-q:]], 0)
    print("conscious share, first quarter -> last quarter, by band:")
    print("  " + "  ".join(f"{b}: {100 * e:.0f}%->{100 * l:.0f}%" for b, e, l in zip(BANDS, early, late)))
    theta = life.gate_theta[live]
    print(f"gate threshold: median {np.median(theta):.2f} (born at 1.5 +/- 0.5)")
    print("largest families:", ", ".join(f"#{l['root']}: {l['members']} (gen {l['max_generation']})"
                                         for l in last["lineages"][:5]))
    print("oldest observatories:")
    for o in last["oldest_observatories"][:6]:
        print(f"  {o['program']:<44} at {o['x']:>2}, {BANDS[o['y']]:<3}  {o['age_s'] / 3600:6.1f} h  "
              f"upkeep {o['integrity']:.0%}")
    who = learned(life, live)
    print(f"organisms whose values carry evidence: {len(who)} of {len(live)}")
    families(life, live)
    return attention(life, who), life, live


def families(life, live, stake=1000.0):
    """Each founder's family: its living members' wealth against the founder's stake."""
    roots, inv = np.unique(life.root[live], return_inverse=True)
    wealth = np.bincount(inv, weights=life.equity(live))
    size = np.bincount(inv)
    grown = size > 1
    multiple = wealth / stake
    print(f"families: {len(roots)} alive, {grown.sum()} with more than one living member; "
          f"{(multiple > 1).sum()} hold more than their founder's stake "
          f"(median multiple {np.median(multiple):.3f})")
    for j in np.argsort(-multiple)[:5]:
        m = live[inv == j]
        bands = np.bincount(life.y[m], minlength=len(TAU))
        where = ", ".join(f"{BANDS[y]}: {n}" for y, n in enumerate(bands) if n)
        print(f"  #{roots[j]}: {size[j]} alive, generation {life.gen[m].max()}, wealth {wealth[j]:,.0f} "
              f"= {multiple[j]:.2f}x the stake, lives at {where}, leverage {np.median(life.lev[m]):.0f}x")


def main(runs):
    results = {run: report(run)[0] for run in runs}
    kinds = list(KINDS) + ["observatories", "markers"]
    print("\nsalience share / organ share, by kind of stream (organisms with evidence only)")
    for run, att in results.items():
        print(f"\n{run}")
        print("  band  " + "".join(f"{k:>14}" for k in kinds))
        for y, row in att.items():
            cells = []
            for k in kinds:
                s, b = row.get(k, (0.0, 0.0))
                cells.append(f"{s / b:14.2f}" if b > 0 else f"{'-':>14}")
            print(f"  {BANDS[y]:>4}  " + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/planet-planted", "runs/planet-null"])
