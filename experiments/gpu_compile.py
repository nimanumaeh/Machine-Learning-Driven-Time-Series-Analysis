"""Compile every soup kernel to PTX, with no GPU needed (only numba-cuda and its NVVM).

    python experiments/gpu_compile.py [--cc 8.0]

Catches what would stop a GPU run before it starts: code the CUDA target cannot
type or lower. It does not run anything; `modal run modal_app.py::selftest` (or
the simulator tests) check that the kernels compute the same world as the CPU.
"""

import argparse
import os
import sys
import time
import types as pytypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from numba import typeof  # noqa: E402
from numba import cuda  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default="8.0", help="compute capability to compile for")
    args = ap.parse_args()
    cc = tuple(int(x) for x in args.cc.split("."))
    import numba.cuda.dispatcher as disp
    if not cuda.is_available():                         # no device: compile for the one asked
        disp.get_current_device = lambda: pytypes.SimpleNamespace(compute_capability=cc)

    from evotrader import gpu
    from evotrader.config import Config
    from evotrader.planet import MINUTE_TAU, Planet
    from evotrader.soup import Primordial, Soup
    from evotrader.weather import NS

    pl = Planet(width=2, seed=0, taus=MINUTE_TAU, step_s=60)
    w = Soup(Config(), pl, per_place=4, seed=0, interactions=8)
    env, done, band = np.zeros((4, 7)), np.zeros((4, pl.Y), np.uint8), np.zeros((4, pl.Y, NS), np.uint8)
    i64 = np.int64(0)
    fields = {name: getattr(w, name) for name in gpu.Engine.MUTABLE + gpu.Engine.FIXED}
    p = Primordial(8, 2, seed=0)
    kernels = {
        "_ticks_grid": (i64, i64, i64, env, done, band, fields["acct"], fields["count"], fields["flow"],
                        fields["soup"], fields["senses"], fields["mask"], fields["site_y"], fields["due"],
                        fields["partner"], fields["claimv"], fields["claims"], fields["act"],
                        fields["table"], fields["expo"], fields["brackets"], fields["offsets"],
                        fields["noise_cdf"], fields["fp"], fields["ip"]),
        "_market": (i64, i64, env, done, band, fields["acct"], fields["count"], fields["soup"],
                    fields["senses"], fields["mask"], fields["site_y"], fields["due"], fields["partner"],
                    fields["claimv"], fields["claims"], fields["brackets"], fields["offsets"],
                    fields["noise_cdf"], fields["fp"], fields["ip"]),
        "_interact": (i64, i64, env, fields["acct"], fields["count"], fields["flow"], fields["soup"],
                      fields["senses"], fields["act"], fields["partner"], fields["claimv"],
                      fields["claims"], fields["table"], fields["expo"], fields["fp"], fields["ip"]),
        "_epochs_grid": (i64, i64, i64, 0.5, p.soup, p.offsets, i64, i64, p.partner, p.claimv, p.claims,
                         i64, p.table, p.stats),
        "_epoch_claim": (i64, i64, 0.5, p.soup, p.offsets, i64, i64, p.partner, p.claimv, p.claims),
        "_epoch_interact": (i64, p.soup, p.partner, p.claimv, p.claims, i64, p.table, p.stats),
    }
    for name, sample in kernels.items():
        kernel = getattr(gpu, name)
        sig = tuple(typeof(a) for a in sample)
        tic = time.time()
        ptx, _ = cuda.compile_ptx(kernel.py_func, sig, cc=cc)
        print(f"{name:16s} compiled for sm_{cc[0]}{cc[1]}: {len(ptx):7d} bytes of PTX in {time.time() - tic:5.1f}s")
    print("every kernel compiles")


if __name__ == "__main__":
    main()
