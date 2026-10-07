"""Does life emerge from random matter on the planet's lattice, with no market at all?

    python experiments/soup_emergence.py [--epochs 20000] [--seed 0] [--width 512]

Every site starts with random bytes. Each epoch every site interacts once with
a random neighbor; bytes flip at random, rarely. Nothing rewards anything. We
watch the soup's high-order entropy: near zero while matter is random, and
rising sharply when self-replicating programs take over (Agüera y Arcas et al.
2024 saw this transition in a large share of runs). When it happens, the most
common programs are printed with whether they copy themselves.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evotrader.soup import L, STEPS, census_of_matter, interact, neighbors, pair_up  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--mutation", type=float, default=0.00024, help="chance a byte flips, per epoch")
    ap.add_argument("--every", type=int, default=250)
    args = ap.parse_args()
    X, Y = args.width, args.rows
    S = X * Y
    rng = np.random.default_rng(args.seed)
    soup = rng.integers(0, 256, (S, L), dtype=np.uint8)
    senses = np.full((S, 1), 128, np.uint8)
    selfs = np.full((S, 2), 128, np.uint8)
    wallet = np.zeros(S)
    act = np.full(S, -1, np.int16)
    flow = np.zeros((S, 3))
    stats = np.zeros((S, 2), np.int64)
    offs = neighbors(X, Y)
    t0 = time.time()
    copies = []
    for epoch in range(1, args.epochs + 1):
        interact(soup, pair_up(rng, X, Y, rng.permutation(S), offs), STEPS, senses, selfs,
                 wallet, act, flow, stats)
        copies.append(stats[:, 1].mean())
        k = rng.poisson(args.mutation * S * L)
        if k:
            soup.reshape(-1)[rng.integers(S * L, size=k)] = rng.integers(0, 256, size=k, dtype=np.uint8)
        if epoch % args.every == 0:
            c = census_of_matter(soup, top=3)
            top = c["top"][0]
            print(f"epoch {epoch:6d}  {time.time() - t0:6.0f}s  entropy {c['entropy']:6.3f}  distinct {c['distinct']:5d}"
                  f"  copies/interaction {np.mean(copies[-args.every:]):7.1f}  top tape x{top[1]}"
                  f" (replicates {top[2]:.0%})", flush=True)
            if c["entropy"] > 1.0:
                print("\ntransition: the most common programs")
                for text, n, rep in c["top"]:
                    shown = "".join(ch if ch in "<>{}+-.,[]SEAT" else "·" for ch in text)
                    print(f"  x{n:5d}  copies itself {rep:4.0%}  {shown}")
                return


if __name__ == "__main__":
    main()
