"""The soup on a CUDA GPU: the same laws (physics.py), every site at once.

Each phase of a tick runs one GPU thread per site (or per stride of sites). The
whole world stays on the device; only the market rows go up, a chunk at a time,
and the state comes back when someone looks (a census, a checkpoint). Two ways
to separate the phases:

grid    one cooperative launch per chunk of rows, with a grid-wide barrier
        between phases (fast: no launch per tick)
launch  two kernel launches per tick (no cooperative launch needed)

Both give the same world as the CPU, up to the last bits of floating point (a
GPU may fuse a multiply and an add into one rounding): the random numbers are
pure functions of (seed, tick, stream, site), so the order in which threads run
does not matter, and no two threads of a phase write the same memory.

Needs numba with CUDA (pip install "numba-cuda[cu12]"); see docs/gpu.md.
"""

import math
import os

import numpy as np
from numba import cuda, uint8
from numba.extending import register_jitable

from . import physics

ph = physics.target("evotrader.physics.cuda", register_jitable)    # the laws, as device functions
_TAPE = physics.TAPE
BLOCK = 128                     # threads per block: the interpreter is heavy on registers
CHUNK = 4096                    # most ticks per launch


def simulated():
    return os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1"


# --------------------------------------------------------------------- kernels: the market soup
@cuda.jit
def _ticks_grid(r0, r1, tick0, env, done, band, acct, counts, flow, soup, senses, mask, site_y, due,
                partner, claimv, claims, act, table, expo, br, offsets, noise_cdf, fp, ip_):
    grid = cuda.cg.this_grid()
    tape = cuda.local.array(_TAPE, uint8)
    sa = cuda.local.array(2, uint8)
    sb = cuda.local.array(2, uint8)
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    S = soup.shape[0]
    for r in range(r0, r1):
        tick = tick0 + r
        buf = tick & 1
        s = start
        while s < S:
            v = ph.market_site(s, tick, r, env, done, band, acct, counts, soup, senses, mask, site_y,
                               due, partner, claimv, br, offsets, noise_cdf, fp, ip_)
            if v != np.uint64(0):
                cuda.atomic.max(claims, (buf, s), v)
                cuda.atomic.max(claims, (buf, partner[s]), v)
            s += stride
        grid.sync()
        s = start
        while s < S:
            ph.interact_site(s, buf, r, env, acct, counts, flow, soup, senses, act, partner, claimv,
                             claims, table, expo, fp, ip_, tape, sa, sb)
            s += stride
        grid.sync()


@cuda.jit
def _market(r, tick, env, done, band, acct, counts, soup, senses, mask, site_y, due, partner, claimv,
            claims, br, offsets, noise_cdf, fp, ip_):
    s = cuda.grid(1)
    if s < soup.shape[0]:
        v = ph.market_site(s, tick, r, env, done, band, acct, counts, soup, senses, mask, site_y, due,
                           partner, claimv, br, offsets, noise_cdf, fp, ip_)
        if v != np.uint64(0):
            buf = tick & 1
            cuda.atomic.max(claims, (buf, s), v)
            cuda.atomic.max(claims, (buf, partner[s]), v)


@cuda.jit
def _interact(r, tick, env, acct, counts, flow, soup, senses, act, partner, claimv, claims, table, expo,
              fp, ip_):
    tape = cuda.local.array(_TAPE, uint8)
    sa = cuda.local.array(2, uint8)
    sb = cuda.local.array(2, uint8)
    s = cuda.grid(1)
    if s < soup.shape[0]:
        ph.interact_site(s, tick & 1, r, env, acct, counts, flow, soup, senses, act, partner, claimv,
                         claims, table, expo, fp, ip_, tape, sa, sb)


# --------------------------------------------------------------------- kernels: matter alone
@cuda.jit
def _epochs_grid(e0, e1, seed, wake, soup, offsets, Xs, Y, partner, claimv, claims, steps, table,
                 stats):
    grid = cuda.cg.this_grid()
    tape = cuda.local.array(_TAPE, uint8)
    sense = cuda.local.array(1, uint8)
    selfs = cuda.local.array(2, uint8)
    wallet = cuda.local.array(2, np.float64)
    act = cuda.local.array(2, np.int16)
    flow = cuda.local.array((2, 3), np.float64)
    sense[0] = 0
    selfs[0] = 0
    selfs[1] = 0
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    S = soup.shape[0]
    for epoch in range(e0, e1):
        buf = epoch & 1
        s = start
        while s < S:
            v = ph.bff_claim(s, epoch, seed, wake, offsets, Xs, Y, partner, claimv)
            if v != np.uint64(0):
                cuda.atomic.max(claims, (buf, s), v)
                cuda.atomic.max(claims, (buf, partner[s]), v)
            s += stride
        grid.sync()
        s = start
        while s < S:
            wallet[0] = 0.0
            wallet[1] = 0.0
            ph.bff_interact(s, buf, soup, partner, claimv, claims, steps, table, sense, selfs, wallet,
                            act, flow, stats, tape)
            s += stride
        grid.sync()


@cuda.jit
def _epoch_claim(epoch, seed, wake, soup, offsets, Xs, Y, partner, claimv, claims):
    s = cuda.grid(1)
    if s < soup.shape[0]:
        v = ph.bff_claim(s, epoch, seed, wake, offsets, Xs, Y, partner, claimv)
        if v != np.uint64(0):
            buf = epoch & 1
            cuda.atomic.max(claims, (buf, s), v)
            cuda.atomic.max(claims, (buf, partner[s]), v)


@cuda.jit
def _epoch_interact(epoch, soup, partner, claimv, claims, steps, table, stats):
    tape = cuda.local.array(_TAPE, uint8)
    sense = cuda.local.array(1, uint8)
    selfs = cuda.local.array(2, uint8)
    wallet = cuda.local.array(2, np.float64)
    act = cuda.local.array(2, np.int16)
    flow = cuda.local.array((2, 3), np.float64)
    sense[0] = 0
    selfs[0] = 0
    selfs[1] = 0
    wallet[0] = 0.0
    wallet[1] = 0.0
    s = cuda.grid(1)
    if s < soup.shape[0]:
        ph.bff_interact(s, epoch & 1, soup, partner, claimv, claims, steps, table, sense, selfs, wallet,
                        act, flow, stats, tape)


# --------------------------------------------------------------------- engines
def _grid_for(kernel, args, n, block, cooperative):
    """Blocks to launch over n sites; a cooperative grid must fit on the device at once."""
    blocks = max(1, math.ceil(n / block))
    if cooperative and not simulated():
        sig = tuple(kernel.typeof_pyval(a) for a in args)
        blocks = min(blocks, kernel.compile(sig).max_cooperative_grid_blocks(block))
    return blocks


class Engine:
    """A Soup's state on the GPU (Soup.to_gpu)."""

    MUTABLE = ("acct", "count", "flow", "soup", "senses", "due", "partner", "claimv", "claims", "act")
    FIXED = ("mask", "site_y", "table", "expo", "brackets", "offsets", "noise_cdf", "fp", "ip")

    def __init__(self, soup, sync="grid", block=BLOCK, chunk=CHUNK):
        if sync not in ("grid", "launch"):
            raise ValueError("sync is 'grid' or 'launch'")
        if not (cuda.is_available() or simulated()):
            raise RuntimeError("no CUDA GPU here (see docs/gpu.md)")
        self.host, self.sync, self.block, self.chunk = soup, sync, block, chunk
        self.d = {}
        self.upload()
        self._grid = None

    def upload(self):
        for name in self.MUTABLE + self.FIXED:
            self.d[name] = cuda.to_device(np.ascontiguousarray(getattr(self.host, name)))

    def download(self):
        for name in self.MUTABLE:
            self.d[name].copy_to_host(getattr(self.host, name))

    def ticks(self, env, done, band, tick0):
        d = self.d
        T = len(env)
        S = self.host.S
        for c0 in range(0, T, self.chunk):
            c1 = min(T, c0 + self.chunk)
            e = cuda.to_device(np.ascontiguousarray(env[c0:c1]))
            dn = cuda.to_device(np.ascontiguousarray(done[c0:c1]))
            bd = cuda.to_device(np.ascontiguousarray(band[c0:c1]))
            if self.sync == "grid":
                args = (0, c1 - c0, tick0 + c0, e, dn, bd, d["acct"], d["count"], d["flow"], d["soup"],
                        d["senses"], d["mask"], d["site_y"], d["due"], d["partner"], d["claimv"],
                        d["claims"], d["act"], d["table"], d["expo"], d["brackets"], d["offsets"],
                        d["noise_cdf"], d["fp"], d["ip"])
                if self._grid is None:
                    self._grid = _grid_for(_ticks_grid, args, S, self.block, True)
                _ticks_grid[self._grid, self.block](*args)
            else:
                blocks = max(1, math.ceil(S / self.block))
                for r in range(c1 - c0):
                    tick = tick0 + c0 + r
                    _market[blocks, self.block](r, tick, e, dn, bd, d["acct"], d["count"], d["soup"],
                                                d["senses"], d["mask"], d["site_y"], d["due"],
                                                d["partner"], d["claimv"], d["claims"], d["brackets"],
                                                d["offsets"], d["noise_cdf"], d["fp"], d["ip"])
                    _interact[blocks, self.block](r, tick, e, d["acct"], d["count"], d["flow"],
                                                  d["soup"], d["senses"], d["act"], d["partner"],
                                                  d["claimv"], d["claims"], d["table"], d["expo"],
                                                  d["fp"], d["ip"])
        cuda.synchronize()


class PrimordialEngine:
    """A Primordial soup's matter on the GPU (Primordial.to_gpu)."""

    MUTABLE = ("soup", "partner", "claimv", "claims", "stats")

    def __init__(self, world, sync="grid", block=BLOCK, chunk=256):
        if not (cuda.is_available() or simulated()):
            raise RuntimeError("no CUDA GPU here (see docs/gpu.md)")
        self.host, self.sync, self.block, self.chunk = world, sync, block, chunk
        self.d = {name: cuda.to_device(np.ascontiguousarray(getattr(world, name)))
                  for name in self.MUTABLE + ("offsets", "table")}
        self._grid = None

    def upload_soup(self):
        self.d["soup"] = cuda.to_device(self.host.soup)

    def download(self):
        for name in self.MUTABLE:
            self.d[name].copy_to_host(getattr(self.host, name))

    def epochs(self, e0, e1):
        w, d = self.host, self.d
        for c0 in range(e0, e1, self.chunk):
            c1 = min(e1, c0 + self.chunk)
            if self.sync == "grid":
                args = (c0, c1, w.seed, w.wake, d["soup"], d["offsets"], w.X, w.Y, d["partner"],
                        d["claimv"], d["claims"], w.steps, d["table"], d["stats"])
                if self._grid is None:
                    self._grid = _grid_for(_epochs_grid, args, w.S, self.block, True)
                _epochs_grid[self._grid, self.block](*args)
            else:
                blocks = max(1, math.ceil(w.S / self.block))
                for epoch in range(c0, c1):
                    _epoch_claim[blocks, self.block](epoch, w.seed, w.wake, d["soup"], d["offsets"], w.X,
                                                     w.Y, d["partner"], d["claimv"], d["claims"])
                    _epoch_interact[blocks, self.block](epoch, d["soup"], d["partner"], d["claimv"],
                                                        d["claims"], w.steps, d["table"], d["stats"])
        cuda.synchronize()


def device_name():
    if simulated():
        return "CUDA simulator"
    dev = cuda.get_current_device()
    name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    return f"{name} (compute {dev.compute_capability[0]}.{dev.compute_capability[1]})"
