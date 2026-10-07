"""Freeze a soup world: what transplants and forward tests need, readable by any later code.

    git worktree add /tmp/soup-v1 379e150          # the code that made a physics-1 world
    python -I experiments/soup_freeze.py /tmp/soup-v1 runs/soup-landauer [more runs]

Writes runs/<world>/frozen.npz (matter, energy, places, physics) and final.json (the
last census). -I keeps the current code out of the way; the first argument is put
first on the path, so the world is read by the code that made it.
"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])                     # the soup-v1 worktree
from evotrader import planet_run
for run in sys.argv[2:]:
    pl, w = planet_run.load(f"{run}/planet.pkl")
    c = w.census(top=8)
    eq = w.equity()
    np.savez_compressed(f"{run}/frozen.npz", soup=w.soup, energy=eq, exposure=w.exposure(),
                        site_place=w.site_place, site_y=w.site_y, avail=pl.avail, taus=pl.taus,
                        step_s=pl.step_s, X=pl.X, E=w.E, S=w.S, steps=w.steps, kleiber=w.kleiber,
                        max_exposure=w.max_exposure, heat=w.heat, digestion=w.digestion,
                        quantum=w.quantum, table=w.table, stake=w.stake, fees=w.acct.fees,
                        funding=w.acct.funding, flow=w.flow, t=w.t, price=w.price, price0=w.price0)
    with open(f"{run}/final.json", "w") as f:
        json.dump(c, f, default=lambda v: v.tolist() if hasattr(v, "tolist") else v.item())
    print(run, "t", w.t, "alive", c["alive"], "energy", round(c["energy"]), "net", round(c["net"]))
