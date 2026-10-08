"""Read a world of organisms (soup physics 3): its lines of descent, what they do, and how its
richest organisms trade on data they never saw.

    python experiments/soup_life.py runs/life                     # unseen: 2025-01-01 to 2026-10-01
    python experiments/soup_life.py runs/life --start 2025-01-01 --until 2025-07-01 --k 10 --baseline 10

Writes runs/life/lines.json and runs/life/transplants.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evotrader import data, data_seconds, planet_run  # noqa: E402
from evotrader.physics import BORN, FOUNDER, GEAR, GEN, TRADES, TRADES0  # noqa: E402
from evotrader.soup import disassemble  # noqa: E402
from evotrader.transplant import evaluate  # noqa: E402


def lines_of(w, top=8):
    """The largest lines of descent alive: who they are and how they trade."""
    alive = w.alive.astype(bool)
    eq, x, hold = w.equity(), w.exposure(), w.holding_s()
    founders, members = np.unique(w.ids[FOUNDER, alive], return_counts=True)
    out = []
    for j in np.argsort(-members)[:top]:
        m = alive & (w.ids[FOUNDER] == founders[j])
        tapes, counts = np.unique(w.soup[m], axis=0, return_counts=True)
        own = (w.count[TRADES] - w.life[TRADES0])[m]
        out.append({
            "founder": int(founders[j]), "alive": int(m.sum()), "energy": float(eq[m].sum()),
            "generation": int(w.life[GEN, m].max()),
            "age_days": float(np.median((w.tick - w.life[BORN, m]) * w.planet.step_s / 86400)),
            "long": float((x[m] > 0.01).mean()), "short": float((x[m] < -0.01).mean()),
            "mean_abs_exposure": float(np.abs(x[m]).mean()),
            "gear_median": float(np.median(w.acct[GEAR, m])),
            "traded": float((own > 0).mean()),
            "holding_days_median": float(np.median(hold[m][own > 0]) / 86400) if (own > 0).any() else None,
            "tape": disassemble(tapes[np.argmax(counts)], w.table), "tape_share": float(counts.max() / m.sum()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--until", default="2026-10-01")
    ap.add_argument("--k", type=int, default=10, help="richest organisms to transplant")
    ap.add_argument("--baseline", type=int, default=10, help="random tapes to transplant")
    ap.add_argument("--store", default=None)
    args = ap.parse_args()

    pl, w = planet_run.load(os.path.join(args.run, "planet.pkl"))
    alive = w.alive.astype(bool)
    n = w.counts
    print(f"{args.run}: {time.strftime('%Y-%m-%d %H:%M', time.gmtime(w.t))} UTC, {int(alive.sum())} alive of "
          f"{w.S}, {n['births']} born, {n['deaths']} died, deepest line {int(w.life[GEN, alive].max(initial=0))} "
          f"generations, energy {w.equity()[alive].sum():,.0f} of {w.injected:,.0f} put in")
    lines = lines_of(w)
    for r in lines:
        hd = f"{r['holding_days_median']:.1f}d" if r["holding_days_median"] is not None else "  -  "
        print(f"  line #{r['founder']:<6d} {r['alive']:5d} alive  energy {r['energy']:10,.0f}  gen {r['generation']:4d}  "
              f"long {r['long']:4.0%} short {r['short']:4.0%} |x| {r['mean_abs_exposure']:.2f}  "
              f"{r['gear_median']:3.0f}x  traded {r['traded']:4.0%}  holds {hd}")
        print(f"      {r['tape']}  ({r['tape_share']:.0%} of the line)")
    with open(os.path.join(args.run, "lines.json"), "w") as f:
        json.dump(lines, f, indent=1)

    ms = lambda day: int(np.datetime64(day, "ms").astype(np.int64))
    if pl.step_s == 60:
        blocks = data.iter_blocks(args.store or "data/market", "BTCUSDT", ms(args.start), ms(args.until))
    else:
        blocks = data_seconds.iter_second_blocks(args.store or "data/seconds", "BTCUSDT", ms(args.start),
                                                 ms(args.until))
    rows = np.concatenate(list(blocks))
    print(f"\nunseen: {len(rows):,} rows from {args.start} to {args.until}, BTC {rows[0, 1]:,.0f} -> {rows[-1, 4]:,.0f}"
          f" ({rows[-1, 4] / rows[0, 1] - 1:+.1%}); the {args.k} richest organisms and {args.baseline} random tapes, "
          f"each alone, evolution and the cost of living off", flush=True)
    tic = time.time()
    report = evaluate(w, rows, k=args.k, baseline=args.baseline)
    print(f"  ({time.time() - tic:.0f}s)")
    for kind in ("organism", "random"):
        rs = [r["return"] for r in report if r["kind"] == kind]
        if rs:
            print(f"  {kind:8s} median return {np.median(rs):+.1%}, mean {np.mean(rs):+.1%}, "
                  f"traded {np.mean([r['trades'] for r in report if r['kind'] == kind]):.0%}")
    with open(os.path.join(args.run, "transplants.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))


if __name__ == "__main__":
    main()
