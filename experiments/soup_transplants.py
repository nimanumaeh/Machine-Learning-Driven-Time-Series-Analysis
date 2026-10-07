"""Transplant the richest organisms of frozen soup worlds onto unseen minutes (2025-01 to 2026-10).

    python experiments/soup_transplants.py runs/soup-landauer [more runs]     # after soup_freeze.py

Each organism is run as a colony of 64 copies in a sandbox at its own latitude and place,
with the same heat, digestion, bites and leverage, evolution off, under physics 2 (the
pairing law differs from the one it evolved under; for a colony of copies that matters
little). Random tapes from the same places are the baseline. Then the whole world, all
4,096 sites with evolution off, runs forward as a fund.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from evotrader import data
from evotrader.config import Config
from evotrader.planet import Planet
from evotrader.soup import Soup
from evotrader.transplant import evaluate, summary, live

class Frozen:
    """A physics-1 world read back from frozen.npz, enough to transplant from."""
    def __init__(self, path):
        z = np.load(path)
        self.soup, self.energy = z["soup"], z["energy"]
        self.site_place, self.site_y = z["site_place"], z["site_y"]
        self.planet = Planet(width=int(z["X"]), seed=0, taus=z["taus"], step_s=int(z["step_s"]))
        self.planet.avail[:] = z["avail"]
        self.E, self.S, self.steps = int(z["E"]), int(z["S"]), int(z["steps"])
        self.kleiber, self.max_exposure = float(z["kleiber"]), float(z["max_exposure"])
        self.heat, self.digestion, self.quantum = float(z["heat"]), float(z["digestion"]), float(z["quantum"])
        self.layout, self.cfg = "symmetric", Config()
    def equity(self):
        return self.energy

ms = lambda y, m: int(np.datetime64(f"{y}-{m:02d}-01", "ms").astype(np.int64))
rows = np.concatenate(list(data.iter_blocks("data/market", "BTCUSDT", ms(2025, 1), ms(2026, 10))))
print(f"unseen rows: {len(rows)} minutes, BTC {rows[0, 1]:.0f} -> {rows[-1, 4]:.0f} ({rows[-1, 4] / rows[0, 1] - 1:+.1%})", flush=True)
for run in sys.argv[1:]:
    w = Frozen(f"{run}/frozen.npz")
    print(f"\n{run}: transplants (64 copies each, evolution off)", flush=True)
    tic = time.time()
    rep = evaluate(w, rows, k=10, sites=64, baseline=5, seed=0)
    print(f"  ({time.time() - tic:.0f}s)", flush=True)
    with open(f"{run}/transplants.json", "w") as f:
        json.dump(rep, f, default=lambda v: v.item() if hasattr(v, "item") else str(v))
