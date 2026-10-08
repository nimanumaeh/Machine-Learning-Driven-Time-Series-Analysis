# The soup on a GPU (and on Modal)

The soup's laws are written once (`evotrader/physics.py`) and compiled twice by numba:
- for the CPU, inlined into tight loops;
- for CUDA, as device functions (`evotrader/gpu.py`).

A world therefore runs the same on a laptop, a CPU server or a GPU. `modal_app.py` runs it on cloud GPUs through [Modal](https://modal.com), started from your own computer.

## 1. Why one world on any device

A tick (one row of market data) has three phases (`docs/soup.md` §1). In none of them do two sites write the same memory:

| Phase | Each site, alone |
|---|---|
| **live** | Its pending order fills at the open, at the leverage it chose. Liquidation, funding and the cost of living follow; an organism with nothing left is doomed. A few of its bytes may flip, and its matter thinks for a few instructions. If it is large enough it may try to divide (claiming a neighboring site), and it may wake to meet a neighbor (claiming it). |
| **birth** | The doomed die. A site claimed by a dividing parent becomes its child, if it is empty or its occupant holds less than the child would get. |
| **meet** | If its claims on itself and on its neighbor both won (the highest random claim on either site), the two run their joined tape. |

The CPU loops over the sites; the GPU gives each site a thread. Randomness is a pure function of (seed, tick, stream, site), so the result does not depend on the order in which sites run, nor on how the ticks are chunked.

How exact:
- In numba's CUDA simulator the GPU kernels give the CPU's world bit for bit (`tests/test_soup.py::test_the_gpu_computes_the_same_world`).
- On a real GPU the matter must still match byte for byte. Money may differ in the last bits, because a GPU may fuse a multiply and an add into one rounding.

`modal run modal_app.py::selftest` checks both before you spend money on a long run.

The weather (each band's bars read as percentiles) does not depend on the matter. It is computed on the host for a chunk of rows at a time (`evotrader/weather.py`, compiled) and sent to the GPU with the rows.

## 2. Speed

Measured on this project's 4-core CPU box (shared with other runs), 4,096 sites, minute data:

| Code | ms per tick | 4 years of minutes |
|---|---|---|
| Soup physics 1 (numpy per row, until Oct 2026) | 4.3 | 2.5 h |
| Soup physics 2 on the CPU (compiled loops, 430 meetings a tick) | 0.6 | 21 min |
| Soup physics 3 on the CPU (4,096 organisms each thinking 32 instructions a tick) | 1.2 to 1.5 | 45 to 55 min |
| GPU | not yet measured here (no GPU): run `modal run modal_app.py::selftest` | |

On a GPU every site works at once. A tick then costs about as long as its slowest site (a few dozen instructions of thought, at most one meeting of 128), whatever the world's size. The CPU instead pays for every organism one after another. So the GPU's advantage grows with the world:
- 4,096 sites: a few times faster.
- 65,000 to 260,000 sites: one to two orders of magnitude faster.

1-second data has 60 times more ticks per day than minute data; that is where the GPU matters.

The two ways of separating the phases on a GPU (`--sync`):
- `grid` (default): one cooperative launch per chunk of ticks, with a grid-wide barrier between phases. No launch per tick.
- `launch`: three kernel launches a tick. A fallback if a GPU refuses cooperative launches.

## 3. On Modal, step by step

Everything below runs from the repository's root on your own computer. Modal bills by the second; an L4 costs about $0.80 an hour, an H100 about $4.

**Set up, once:**
```bash
pip install modal
modal setup                     # opens a browser to log in; stores a token
```

**Check the GPU computes the same world, and see how fast it is (a few minutes):**
```bash
modal run modal_app.py::selftest                 # on an L4
modal run modal_app.py::selftest --gpu H100      # or any other
```
It prints `PASS: the GPU computes the same world` and the milliseconds per tick for growing worlds on the CPU and the GPU.

**Put market data into the cloud volume (once; an hour or two, on CPUs):**
```bash
modal run --detach modal_app.py::fetch                                        # minutes, 2017-08 to last month
modal run --detach modal_app.py::fetch --no-minutes --seconds-from 2026-09-01 --seconds-until 2026-10-01
```
The minute store is built from data.binance.vision archives, about 350 MB. The 1-second store is rebuilt from every trade, about 0.2 MB a day, and needs the minute store for its slow streams.

Binance's own API refuses some countries, including the US (HTTP 451); the archives do not. If the archives refuse your Modal region too, add `region="eu"` (or another region) to the `@app.function` decorators.

**Run a soup:**
```bash
modal run --detach modal_app.py::soup --name s1 --args "--minutes --from 2021-01-01 --until 2025-01-01"
modal run --detach modal_app.py::soup --name s2 --gpu H100 --args "--from 2026-09-01 --width 256 --height 256"
```
`--args` takes the flags of `python -m evotrader soup`; `--device gpu` and the run directory are added for you.

A run checkpoints to the volume every 10 minutes. After 23 hours (Modal allows 24 per call) it saves and starts its own next call, which resumes exactly where it stopped. With `--detach` it carries on after you close the terminal.
- `modal app logs evotrader` shows its census lines.
- `modal app stop evotrader` stops everything; the last checkpoint stays in the volume.

**Watch and read the results:**
```bash
modal run modal_app.py::page --names s1                  # writes s1.html here: open it in a browser
modal run modal_app.py::page --names s1,s2 --out both.html
modal volume ls evotrader runs/s1
modal volume get evotrader runs/s1/census.jsonl .
modal volume get evotrader runs/s1/planet.pkl .          # the whole world, to study or resume locally
```

**Reverse engineer: read the richest organisms back as strategies, on data they never saw:**
```bash
modal run modal_app.py::transplant --name s1 --start 2025-01-01 --until 2026-10-01
```

**Abiogenesis at scale (matter alone, no market):**
```bash
modal run --detach modal_app.py::abiogenesis --name life1 --args "--width 2048 --rows 64 --epochs 100000"
```
This is 131,072 tapes, the size of Agüera y Arcas et al. 2024. When self-replicators take over it saves the soup as `runs/life1.npy`. That matter can seed a market soup:
```bash
modal volume get evotrader runs/life1.npy .
modal volume put evotrader life1.npy matter/life1.npy
modal run --detach modal_app.py::soup --name seeded --args "--minutes --from 2021-01-01 --matter /vol/matter/life1.npy --layout bff"
```

## 4. If something goes wrong

- **`CUDA driver library cannot be found`, or the PTX version is too new for the driver.** The image pins numba 0.68.0, numba-cuda 0.30.4 and CUDA 12 from pip; numba-cuda links with nvJitLink, which works with any CUDA 12 driver. If Modal's driver is older, pin an older CUDA toolkit in `modal_app.py` (for example `cuda-toolkit[nvcc,nvjitlink]==12.4.*`).
- **NumPy 2.5 breaks numba-cuda 0.30** (it uses the removed `np.row_stack`). The image pins numpy 2.4.6.
- **`grid` refuses to launch** (cooperative launch not allowed): add `--sync launch` to `--args`.
- **First calls are slow.** numba compiles the CPU loops once (about 20 s) and caches them in the volume (`NUMBA_CACHE_DIR`); the GPU kernels compile in a few seconds on each new container.

Without a GPU, two checks run locally:
```bash
pip install "numpy<2.5" numba "numba-cuda[cu12]"
python experiments/gpu_compile.py --cc 8.9                          # every kernel compiles for this GPU
NUMBA_ENABLE_CUDASIM=1 python experiments/gpu_selftest.py --small    # and computes the CPU's world
```
