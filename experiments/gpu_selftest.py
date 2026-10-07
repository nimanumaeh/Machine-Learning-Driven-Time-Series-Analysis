"""Does the GPU compute the same world as the CPU? And how fast is it?

    python experiments/gpu_selftest.py                 # on a machine with a CUDA GPU
    NUMBA_ENABLE_CUDASIM=1 python experiments/gpu_selftest.py --small   # no GPU: the simulator
    python experiments/gpu_selftest.py --bench         # also time bigger worlds

The same small world (and the same lifeless primordial soup) is run on the CPU and
on the GPU, in both ways of separating the phases, and compared: the matter must
be identical byte for byte and the money equal to the last bits a fused
multiply-add can change. Exits non-zero if they differ.
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore", message="overflow encountered")   # uint64 hashing, simulated


def make(width, per_place, seed, steps, interactions):
    from evotrader.config import Config
    from evotrader.planet import Planet
    from evotrader.soup import Soup
    pl = Planet(width=width, seed=0)
    return Soup(Config(), pl, per_place=per_place, seed=seed, interactions=interactions, steps=steps,
                max_exposure=125, heat=0.0001, digestion=0.8, noise=0.002)


def compare(a, b, label):
    a.sync()
    b.sync()
    ok = np.array_equal(a.soup, b.soup) and np.array_equal(a.count, b.count)
    ok &= np.array_equal(a.senses, b.senses) and np.array_equal(a.act, b.act)
    money = np.nan_to_num(a.acct, nan=-1.0), np.nan_to_num(b.acct, nan=-1.0)
    close = np.allclose(money[0], money[1], rtol=1e-9, atol=1e-9)
    worst = float(np.max(np.abs(money[0] - money[1])))
    print(f"  {label:28s} matter {'same' if ok else 'DIFFERENT'}   money {'same' if close else 'DIFFERENT'}"
          f" (largest difference {worst:.2e})   interactions {int(b.counts['interactions'])}")
    return ok and close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="a world small enough for the CUDA simulator")
    ap.add_argument("--bench", action="store_true", help="time bigger worlds on the GPU")
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()
    from evotrader import gpu
    from evotrader.soup import BFF, Primordial
    from evotrader.synthetic import synthetic_rows
    print(f"device: {gpu.device_name()}")
    if args.small:
        width, per_place, steps, E, n_rows, X, Y, epochs = 1, 4, 256, 8, args.rows or 12, 8, 4, 6
    else:
        width, per_place, steps, E, n_rows, X, Y, epochs = 8, 16, 8192, 256, args.rows or 2000, 128, 8, 200
    rows = np.array(synthetic_rows(n_rows, seed=5, step_s=1, start_ms=1_700_006_400_000))
    good = True

    print(f"market soup: {width * per_place * 8} sites, {n_rows} ticks")
    cpu = make(width, per_place, 3, steps, E)
    tic = time.time()
    cpu.advance(rows)
    print(f"  cpu                          {time.time() - tic:7.2f}s")
    for sync in ("grid", "launch"):
        g = make(width, per_place, 3, steps, E).to_gpu(sync=sync)
        tic = time.time()
        g.advance(rows)
        g.sync()
        print(f"  gpu ({sync})                   {time.time() - tic:7.2f}s")
        good &= compare(cpu, g, f"cpu vs gpu ({sync})")

    print(f"primordial soup: {X * Y} tapes, {epochs} epochs")
    a = Primordial(X, Y, seed=1, table=BFF, steps=steps)
    a.advance(epochs)
    for sync in ("grid", "launch"):
        b = Primordial(X, Y, seed=1, table=BFF, steps=steps).to_gpu(sync=sync)
        b.advance(epochs)
        b.sync()
        same = np.array_equal(a.soup, b.soup) and np.array_equal(a.stats, b.stats)
        print(f"  cpu vs gpu ({sync})          matter {'same' if same else 'DIFFERENT'}")
        good &= same

    if args.bench and not gpu.simulated():
        bench(rows)
    print("PASS: the GPU computes the same world" if good else "FAIL: the GPU world differs")
    sys.exit(0 if good else 1)


def bench(rows):
    """Seconds per simulated tick for growing worlds, CPU against GPU."""
    from evotrader.soup import BFF, Primordial
    print("\nbenchmark (ms per tick; minute or second rows cost the same)")
    for width, per_place, E in ((32, 16, 512), (64, 64, 4096), (128, 128, 16384)):
        sites = width * per_place * 8
        line = f"  {sites:7d} sites, {E:6d} interactions a tick:"
        for device in ("cpu", "gpu"):
            if device == "cpu" and sites > 70_000:
                line += "   cpu    (skipped)"
                continue
            w = make(width, per_place, 3, 8192, E)
            if device == "gpu":
                w.to_gpu()
            n = 600 if device == "gpu" else 200
            w.advance(rows[:20])                                   # compile and warm up
            w.sync()
            tic = time.time()
            w.advance(rows[20:20 + n])
            w.sync()
            line += f"   {device} {1000 * (time.time() - tic) / n:8.3f}"
        print(line, flush=True)
    for X in (512, 2048):
        p = Primordial(X, 64, seed=0, table=BFF)
        p.to_gpu()
        p.advance(4)
        p.sync()
        tic = time.time()
        p.advance(64)
        p.sync()
        print(f"  primordial {X * 64:7d} tapes: {(time.time() - tic) / 64 * 1000:8.2f} ms per epoch on the gpu")


if __name__ == "__main__":
    main()
