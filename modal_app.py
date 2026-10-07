"""Run the soup on cloud GPUs with Modal (https://modal.com). See docs/gpu.md.

Once, on your own computer, from this directory:
    pip install modal && modal setup

Then:
    modal run modal_app.py::selftest                       # the GPU computes the CPU's world? how fast?
    modal run --detach modal_app.py::fetch                 # market data into the cloud volume (hours)
    modal run --detach modal_app.py::soup --name s1 --args "--minutes --from 2021-01-01 --until 2025-01-01"
    modal run modal_app.py::page --names s1                # its census page, saved here as s1.html
    modal run --detach modal_app.py::abiogenesis --name life1 --args "--width 2048 --rows 64"
    modal run modal_app.py::transplant --name s1 --start 2025-01-01 --until 2026-10-01
    modal volume ls evotrader runs/s1                      # anything a run wrote
    modal volume get evotrader runs/s1/census.jsonl .

Choose the GPU with --gpu (T4, L4, A10G, L40S, A100, A100-80GB, H100, ...). A soup
run saves itself every 10 minutes and, if it is not done within a day (Modal's
limit for one call), starts its own next call and carries on where it stopped.
With --detach a run goes on after you close the terminal; `modal app logs evotrader`
shows what it prints.
"""

import os
import shlex
import subprocess
import sys
import threading
import time

import modal

VOL = "/vol"
GPU = "L4"                    # a good default for the soup: fast clocks, cheap; H100 for huge worlds
DAY = 24 * 3600

app = modal.App("evotrader")
volume = modal.Volume.from_name("evotrader", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.4.6", "numba==0.68.0", "numba-cuda[cu12]==0.30.4")
    .env({"NUMBA_CACHE_DIR": f"{VOL}/numba-cache", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("evotrader", remote_path="/root/evotrader", ignore=["**/__pycache__"])
    .add_local_dir("experiments", remote_path="/root/experiments", ignore=["**/__pycache__"])
)


def _keep_committing(every_s=600):
    """Commit the volume every few minutes, so a crash loses little."""
    stop = threading.Event()

    def loop():
        while not stop.wait(every_s):
            try:
                volume.commit()
            except Exception as e:                      # keep running; the final commit will retry
                print(f"(volume commit failed: {e})", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    return stop


def _evotrader(argv):
    sys.path.insert(0, "/root")
    from evotrader.__main__ import main
    return main(argv)


@app.function(image=image, gpu=GPU, volumes={VOL: volume}, timeout=3600)
def gpu_selftest(bench: bool = True):
    """The same worlds on the GPU and the CPU, compared; then timings for bigger worlds."""
    cmd = [sys.executable, "/root/experiments/gpu_selftest.py"] + (["--bench"] if bench else [])
    code = subprocess.run(cmd).returncode
    volume.commit()                                     # keep the compiled CPU code for next time
    if code:
        raise SystemExit(code)


@app.local_entrypoint()
def selftest(gpu: str = GPU, bench: bool = True):
    """modal run modal_app.py::selftest [--gpu H100] [--no-bench]"""
    gpu_selftest.with_options(gpu=gpu).remote(bench)


@app.function(image=image, cpu=2.0, memory=8192, volumes={VOL: volume}, timeout=DAY)
def fetch(minutes: bool = True, seconds_from: str = "", seconds_until: str = ""):
    """Build the minute store (and, if asked, the 1-second store for some days) in the volume.

    The minute store is every BTCUSDT minute since 2017-08 (about 350 MB); the
    1-second store is rebuilt from every trade, about 0.2 MB a day.
    """
    stop = _keep_committing()
    try:
        if minutes:
            _evotrader(["download", "--store", f"{VOL}/data/market", "--cache", "/tmp/raw"])
            volume.commit()
        if seconds_from:
            argv = ["seconds", "--store", f"{VOL}/data/seconds", "--minute-store", f"{VOL}/data/market",
                    "--from", seconds_from]
            if seconds_until:
                argv += ["--until", seconds_until]
            _evotrader(argv)
    finally:
        stop.set()
        volume.commit()


@app.function(image=image, gpu=GPU, volumes={VOL: volume}, timeout=DAY)
def gpu_soup(name: str, args: str = "", gpu: str = GPU, hours: float = 23.0):
    """A soup world in the volume's runs/<name>, on the GPU; args as for `python -m evotrader soup`."""
    volume.reload()
    run = f"{VOL}/runs/{name}"
    extra = shlex.split(args)
    argv = ["soup", "--device", "gpu", "--run", run, "--max-hours", str(hours)]
    if os.path.exists(f"{run}/planet.pkl"):
        argv.append("--resume")
    elif "--store" not in extra:
        argv += ["--store", f"{VOL}/data/market" if "--minutes" in extra else f"{VOL}/data/seconds"]
    stop = _keep_committing()
    try:
        finished = _evotrader(argv + extra)
    finally:
        stop.set()
        volume.commit()
    if finished is False:                               # a day was not enough: carry on in a new call
        print("carrying on in a new call", flush=True)
        gpu_soup.with_options(gpu=gpu).spawn(name, args, gpu, hours)


@app.local_entrypoint()
def soup(name: str, args: str = "", gpu: str = GPU, hours: float = 23.0):
    """modal run --detach modal_app.py::soup --name s1 [--gpu L4] --args "--minutes --from 2021-01-01" """
    gpu_soup.with_options(gpu=gpu).remote(name, args, gpu, hours)


@app.function(image=image, gpu=GPU, volumes={VOL: volume}, timeout=DAY)
def gpu_abiogenesis(name: str, args: str = ""):
    """Matter alone on the GPU: does life start? (experiments/soup_emergence.py)"""
    os.makedirs(f"{VOL}/runs", exist_ok=True)
    cmd = [sys.executable, "/root/experiments/soup_emergence.py", "--device", "gpu",
           "--save", f"{VOL}/runs/{name}.npy"] + shlex.split(args)
    stop = _keep_committing()
    try:
        code = subprocess.run(cmd).returncode
    finally:
        stop.set()
        volume.commit()
    if code:
        raise SystemExit(code)


@app.local_entrypoint()
def abiogenesis(name: str, args: str = "", gpu: str = GPU):
    """modal run --detach modal_app.py::abiogenesis --name life1 [--gpu H100] --args "--width 2048 --rows 64" """
    gpu_abiogenesis.with_options(gpu=gpu).remote(name, args)


@app.function(image=image, cpu=2.0, memory=8192, volumes={VOL: volume}, timeout=DAY)
def transplant(name: str, start: str, until: str, k: int = 10, baseline: int = 5):
    """Read a world's richest organisms back as strategies on data they never saw (transplant.py)."""
    sys.path.insert(0, "/root")
    from evotrader import data, data_seconds, planet_run
    from evotrader.transplant import evaluate
    volume.reload()
    pl, world = planet_run.load(f"{VOL}/runs/{name}/planet.pkl")
    ms = lambda day: int(time.mktime(time.strptime(day, "%Y-%m-%d")) - time.timezone) * 1000
    if pl.step_s == 60:
        blocks = data.iter_blocks(f"{VOL}/data/market", "BTCUSDT", ms(start), ms(until))
    else:
        blocks = data_seconds.iter_second_blocks(f"{VOL}/data/seconds", "BTCUSDT", ms(start), ms(until))
    import numpy as np
    rows = np.concatenate(list(blocks))
    print(f"{name}: {k} richest organisms and {baseline} random tapes, on {len(rows)} unseen rows")
    evaluate(world, rows, k=k, baseline=baseline)


@app.function(image=image, volumes={VOL: volume}, timeout=1800)
def render(names: str) -> str:
    """The census page of one or more soup runs (comma separated), as HTML."""
    sys.path.insert(0, "/root")
    from evotrader.viewer import write_soup_viewer
    volume.reload()
    write_soup_viewer([f"{VOL}/runs/{n}" for n in names.split(",")], "/tmp/soup.html")
    with open("/tmp/soup.html") as f:
        return f.read()


@app.local_entrypoint()
def page(names: str, out: str = ""):
    """Save the census page of soup runs here: modal run modal_app.py::page --names s1,s2"""
    out = out or names.replace(",", "+") + ".html"
    with open(out, "w") as f:
        f.write(render.remote(names))
    print(f"wrote {out}")
