"""Run a planet: step it and its life through 1-second rows, watch it, keep it.

A run directory holds:
  planet.pkl     the whole world (planet and life), to resume exactly
  census.jsonl   one census of life on the planet per line, every census_every_s
  meta.json      how the world was made
"""

import json
import os
import pickle
import signal
import time

import numpy as np


def save(path, planet, life):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump((planet, life), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def status_line(c):
    day = time.strftime("%Y-%m-%d %H:%M", time.gmtime(c["t"]))
    if "matter" in c:                                       # a soup world
        m, k = c["matter"], c["counts"]
        top = m["top"][0] if m["top"] else ("", 0, 0.0)
        move = c["price"] / c["price0"] - 1 if c["price0"] else 0.0
        return (f"{day}  alive {c['alive']:5d}/{c['sites']}  complexity {m['entropy']:5.2f}  "
                f"distinct {m['distinct']:5d}  top x{top[1]:<4d} copies {top[2]:4.0%}  "
                f"long {c['long']:4.0%} short {c['short']:4.0%}  taken {c['taken']:9.0f}  "
                f"net {c['net']:+11.0f}  fees {c['fees']:9.0f}  rain {k['rain']:4d}  BTC {move:+.1%}")
    k = c["counts"]
    move = c["price"] / c["price0"] - 1 if c["price0"] else 0.0
    return (f"{day}  pop {c['population']:5d}  bands {c['by_band']}  gen {c['max_generation']:3d}  "
            f"born {k['born']:6d}  died {k['died']:6d}  seeded {k['seeded']:6d}  "
            f"net {c['net']:+11.0f}  fees {c['fees']:10.0f}  BTC {move:+.1%}")


def run_planet(planet, life, rows, run_dir, census_every_s=1800, checkpoint_every_s=600,
               log=print, max_seconds=None):
    """Step through `rows` until exhausted, `max_seconds` new seconds, or Ctrl-C.

    Rows the world has already lived through are skipped, so a resumed run can be
    fed the same stream from its start.
    """
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, "planet.pkl")
    last_ckpt, n = time.time(), 0
    next_census = None
    try:                                                     # a polite kill saves the world too
        old_term = signal.signal(signal.SIGTERM, _interrupt)
    except ValueError:                                       # not the main thread
        old_term = None
    try:
        with open(os.path.join(run_dir, "census.jsonl"), "a") as out:
            for row in rows:
                if (row[0] + 1000) / 1000.0 <= planet.t:
                    continue                                 # lived through already
                life.step(row, planet.step(row))
                n += 1
                if next_census is None:
                    next_census = (int(planet.t) // census_every_s + 1) * census_every_s
                if planet.t >= next_census:
                    c = life.census()
                    out.write(json.dumps(c, default=_plain) + "\n")
                    out.flush()
                    log(status_line(c))
                    next_census += census_every_s
                if time.time() - last_ckpt >= checkpoint_every_s:
                    save(ckpt, planet, life)
                    last_ckpt = time.time()
                if max_seconds is not None and n >= max_seconds:
                    break
    except KeyboardInterrupt:
        log("\ninterrupted; saving the world")
    finally:
        save(ckpt, planet, life)
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
    return n


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def read_census(run_dir):
    with open(os.path.join(run_dir, "census.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))
