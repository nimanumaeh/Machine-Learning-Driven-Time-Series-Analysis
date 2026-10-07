"""Run a whole frozen soup world forward on unseen minutes, evolution off, as a fund; against
the same world (same places, same capital per site) made of random matter.

    python experiments/soup_forward.py runs/soup-landauer evolved      # after soup_freeze.py
    python experiments/soup_forward.py runs/soup-landauer random
"""
import json, sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from evotrader import data
from evotrader.config import Config
from evotrader.planet import Planet
from evotrader.physics import WALLET, FEES, FUNDING
from evotrader.soup import Soup

run, kind = sys.argv[1], sys.argv[2]
z = np.load(f"{run}/frozen.npz")
pl = Planet(width=int(z["X"]), seed=0, taus=z["taus"], step_s=int(z["step_s"]))
pl.avail[:] = z["avail"]
matter = z["soup"] if kind == "evolved" else np.random.default_rng(7).integers(0, 256, z["soup"].shape, dtype=np.uint8)
w = Soup(Config(), pl, per_place=int(z["S"]) // (int(z["X"]) * len(z["taus"])), seed=7, interactions=int(z["E"]),
         noise=0.0, steps=int(z["steps"]), matter=matter, kleiber=float(z["kleiber"]),
         max_exposure=float(z["max_exposure"]), heat=float(z["heat"]), digestion=float(z["digestion"]),
         quantum=float(z["quantum"]))
capital = np.maximum(z["energy"], 0.0)
w.acct[WALLET] = capital                                  # the world's capital, each site flat
w.injected = float(capital.sum())
ms = lambda y, m: int(np.datetime64(f"{y}-{m:02d}-01", "ms").astype(np.int64))
rows = np.concatenate(list(data.iter_blocks("data/market", "BTCUSDT", ms(2025, 1), ms(2026, 10))))
tic = time.time()
day = 1440
curve = []
for k in range(0, len(rows), day):
    w.advance(rows[k:k + day])
    curve.append((float(w.t), float(w.equity().sum()), float(rows[min(k + day, len(rows)) - 1, 4])))
eq = w.equity()
out = {"run": run, "kind": kind, "start": w.injected, "end": float(eq.sum()),
       "return": float(eq.sum() / w.injected - 1), "hold_btc": float(rows[-1, 4] / rows[0, 1] - 1),
       "fees": float(w.acct[FEES].sum()), "funding": float(w.acct[FUNDING].sum()),
       "heat": float(w.flow[:, 2].sum()), "counts": w.counts, "curve": curve,
       "seconds": time.time() - tic}
with open(f"{run}/fund-{kind}.json", "w") as f:
    json.dump(out, f)
print(f"{run} {kind}: {out['start']:.0f} -> {out['end']:.0f} USDT ({out['return']:+.1%}), holding BTC {out['hold_btc']:+.1%}; "
      f"fees {out['fees']:.0f}, heat {out['heat']:.0f}, funding {out['funding']:.0f}; {out['counts']['orders']} orders; {out['seconds']:.0f}s")
