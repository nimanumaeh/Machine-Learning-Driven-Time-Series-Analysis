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
    from evotrader.soup import Primordial, Soup
    from evotrader.weather import NS

    w = Soup(Config(), width=4, height=4, step_s=60, seed=0)
    env, band = np.zeros((4, 7)), np.zeros((4, 1, NS), np.uint8)
    i64 = np.int64(0)
    f = {name: getattr(w, name) for name in gpu.Engine.MUTABLE + gpu.Engine.FIXED}
    p = Primordial(8, 2, seed=0)
    kernels = {
        "_life_grid": (i64, i64, i64, env, band, f["acct"], f["count"], f["flow"], f["soup"], f["regs"],
                       f["mask"], f["alive"], f["doomed"], f["partner"], f["claimv"], f["claims"], f["target"],
                       f["birthv"], f["births"], f["divide"], f["act"], f["ids"], f["life"], f["offsets"],
                       f["noise_cdf"], f["expo"], f["gears"], f["table"], f["brackets"], f["fp"], f["ip"]),
        "_life_live": (i64, i64, env, band, f["acct"], f["count"], f["flow"], f["soup"], f["regs"], f["mask"],
                       f["alive"], f["doomed"], f["life"], f["partner"], f["claimv"], f["claims"], f["target"], f["birthv"],
                       f["births"], f["divide"], f["offsets"], f["noise_cdf"], f["expo"], f["gears"],
                       f["table"], f["brackets"], f["fp"], f["ip"]),
        "_life_birth": (i64, i64, env, f["acct"], f["count"], f["flow"], f["soup"], f["regs"], f["alive"],
                        f["doomed"], f["births"], f["birthv"], f["divide"], f["ids"], f["life"], f["fp"], f["ip"]),
        "_life_meet": (i64, i64, env, band, f["acct"], f["count"], f["flow"], f["soup"], f["mask"], f["act"],
                       f["partner"], f["claimv"], f["claims"], f["alive"], f["life"], f["table"], f["expo"],
                       f["gears"], f["fp"], f["ip"]),
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
