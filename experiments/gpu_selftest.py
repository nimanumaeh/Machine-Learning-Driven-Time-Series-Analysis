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


def make(width, height, seed, think=32, meet=128, floor=1.0, divide_at=900.0):
    """A small, busy world: short lives (a one-day cost of living), bodies that divide young (at 900
    USDT, trying every tick), many births, deaths, takeovers, meetings and bites."""
    from evotrader.config import Config
    from evotrader.soup import Soup
    return Soup(Config(), width=width, height=height, step_s=1, seed=seed, think=think, meet=meet,
                meetings=21600.0, metabolism_days=1.0, upkeep=50.0, floor=floor, birth_min=20.0,
                divide_at=divide_at, cycle_days=0.0, mutation=0.05, noise=5.0, heat=1e-4, quantum=20.0)


def compare(a, b, label):
    a.sync()
    b.sync()
    ok = np.array_equal(a.soup, b.soup) and np.array_equal(a.count, b.count)
    ok &= np.array_equal(a.alive, b.alive) and np.array_equal(a.regs, b.regs)
    ok &= np.array_equal(a.ids, b.ids) and np.array_equal(a.life, b.life)
    money = np.nan_to_num(a.acct, nan=-1.0), np.nan_to_num(b.acct, nan=-1.0)
    close = np.allclose(money[0], money[1], rtol=1e-9, atol=1e-9) and np.allclose(a.flow, b.flow, rtol=1e-9,
                                                                                    atol=1e-9)
    worst = float(np.max(np.abs(money[0] - money[1])))
    c = b.counts
    print(f"  {label:28s} matter {'same' if ok else 'DIFFERENT'}   money {'same' if close else 'DIFFERENT'}"
          f" (largest difference {worst:.2e})   births {c['births']}  deaths {c['deaths']}  meetings {c['meetings']}")
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
    if args.small:                                  # the floor is high so that some die at once
        width, height, steps, n_rows, X, Y, epochs, floor = 6, 6, 256, args.rows or 12, 8, 4, 6, 300.0
    else:
        width, height, steps, n_rows, X, Y, epochs, floor = 32, 32, 8192, args.rows or 3000, 128, 8, 200, 1.0
    rows = np.array(synthetic_rows(n_rows, seed=5, step_s=1, start_ms=1_700_006_400_000))
    good = True

    print(f"living matter: {width * height} sites, {n_rows} ticks")
    cpu = make(width, height, 3, floor=floor)
    tic = time.time()
    cpu.advance(rows)
    print(f"  cpu                          {time.time() - tic:7.2f}s")
    for sync in ("grid", "launch"):
        g = make(width, height, 3, floor=floor).to_gpu(sync=sync)
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
    """Milliseconds per simulated tick for growing worlds, CPU against GPU."""
    from evotrader.soup import BFF, Primordial
    print("\nbenchmark (ms per tick; minute or second rows cost the same)")
    for side in (64, 128, 256, 512):
        line = f"  {side * side:7d} sites:"
        for device in ("cpu", "gpu"):
            if device == "cpu" and side > 128:
                line += "   cpu    (skipped)"
                continue
            w = make(side, side, 3)
            w.metabolism_days = 90.0
            if device == "gpu":
                w.to_gpu()
            n = 1000 if device == "gpu" else 100
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
