"""Does life emerge from random matter on the planet's lattice, with no market at all?

    python experiments/soup_emergence.py [--epochs 20000] [--seed 0] [--width 512] [--device gpu]

Every site starts with random bytes. Each epoch every site may wake (with chance
--wake), claim a random neighbor, and the pairs whose claims won run their joined
tapes; bytes flip at random, rarely. Nothing rewards anything. We watch the
soup's high-order entropy: near zero while matter is random, and rising sharply
when self-replicating programs take over (Agüera y Arcas et al. 2024 saw this
transition in a large share of runs of 2^17 tapes). When it happens, the most
common programs are printed with whether they copy themselves, and the soup can
be saved to seed a market soup (python -m evotrader soup --matter ...).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evotrader.soup import Primordial, census_of_matter, chemistry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--wake", type=float, default=1.0, help="chance a site starts an interaction, per epoch")
    ap.add_argument("--mutation", type=float, default=0.00024, help="chance a byte flips, per epoch")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--chemistry", choices=("bff", "symmetric"), default="bff",
                    help="bff: the ASCII chemistry of the paper")
    ap.add_argument("--market", action="store_true",
                    help="keep the market instructions (inert here, but S and E still overwrite bytes)")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--save", help="save the soup here (.npy) when life appears")
    ap.add_argument("--threshold", type=float, default=1.0, help="high-order entropy that counts as life")
    args = ap.parse_args()
    table = chemistry(args.chemistry, market=args.market)
    world = Primordial(args.width, args.rows, seed=args.seed, table=table, wake=args.wake)
    if args.device == "gpu":
        from evotrader.gpu import device_name
        world.to_gpu()
        print(f"on the GPU: {device_name()}")
    print(f"{world.S} tapes of 64 bytes on a {args.width} x {args.rows} lattice, {args.chemistry} chemistry")
    t0 = time.time()
    last = np.zeros(3, np.int64)
    while world.epoch < args.epochs:
        world.advance(min(args.every, args.epochs - world.epoch), mutation=args.mutation)
        world.sync()
        c = census_of_matter(world.soup, top=3, table=table)
        totals = world.stats.sum(0)
        n, executed, copies = totals - last
        last = totals
        top = c["top"][0]
        print(f"epoch {world.epoch:6d}  {time.time() - t0:6.0f}s  entropy {c['entropy']:6.3f}  "
              f"distinct {c['distinct']:6d}  interactions/epoch {n / args.every:8.0f}  "
              f"copies/interaction {copies / max(n, 1):6.1f}  top tape x{top[1]} (replicates {top[2]:.0%})",
              flush=True)
        if c["entropy"] > args.threshold:
            print("\ntransition: the most common programs")
            for text, k, rep in c["top"]:
                print(f"  x{k:5d}  copies itself {rep:4.0%}  {text}")
            if args.save:
                np.save(args.save, world.soup)
                print(f"soup saved to {args.save}")
            return


if __name__ == "__main__":
    main()
