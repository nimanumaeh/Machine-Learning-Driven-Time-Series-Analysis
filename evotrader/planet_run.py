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
    if "births" in c:                                       # a soup world of organisms (physics 3)
        move = c["price"] / c["price0"] - 1 if c["price0"] else 0.0
        return (f"{day}  alive {c['alive']:5d}/{c['sites']}  born {c['births']:7d}  died {c['deaths']:7d}  "
                f"gen {c['generation_max']:4d}  lines {c['lines']:5d}  energy {c['energy']:10.0f}  "
                f"net {c['net']:+11.0f}  fees {c['fees']:9.0f}  living {c['metabolism']:9.0f}  "
                f"long {c['long']:4.0%} short {c['short']:4.0%}  hold {c['timescales']}  BTC {move:+.1%}")
    if "matter" in c:                                       # a soup world
        m, k = c["matter"], c["counts"]
        top = m["top"][0] if m["top"] else ("", 0, 0.0)
        move = c["price"] / c["price0"] - 1 if c["price0"] else 0.0
        return (f"{day}  alive {c['alive']:5d}/{c['sites']}  complexity {m['entropy']:5.2f}  "
                f"distinct {m['distinct']:5d}  top x{top[1]:<4d} copies {top[2]:4.0%}  "
                f"long {c['long']:4.0%} short {c['short']:4.0%}  taken {c['taken']:9.0f}  "
                f"heat {c.get('heat', 0.0):9.0f}  "
                f"net {c['net']:+11.0f}  fees {c['fees']:9.0f}  BTC {move:+.1%}")
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
                if (row[0] + 1000 * planet.step_s) / 1000.0 <= planet.t:
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


class _Enough(Exception):
    pass


def run_soup(soup, blocks, run_dir, census_every_s=1800, checkpoint_every_s=600, log=print,
             max_rows=None, max_wall_s=None, on_save=None, chunk=1 << 16):
    """Live a soup through `blocks` (arrays of rows), a chunk of ticks at a time.

    A census is taken at every multiple of census_every_s of market time, the world
    is saved every checkpoint_every_s of wall time (and at the end), and on_save(),
    if given, runs after each save (to commit a cloud volume, say). Rows the world
    has already lived through are skipped, so a resumed run can be fed the same
    stream from its start. Stops when the blocks run out, after max_rows new rows,
    after max_wall_s seconds of wall time, or on Ctrl-C or SIGTERM. Returns
    (rows lived, whether the stream ran out).
    """
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, "planet.pkl")
    planet = soup.planet
    step_ms = 1000 * planet.step_s
    start = last_save = time.time()
    n, finished = 0, False
    next_census = None

    def keep():
        save(ckpt, planet, soup)
        if on_save is not None:
            on_save()

    try:
        old_term = signal.signal(signal.SIGTERM, _interrupt)
    except ValueError:                                       # not the main thread
        old_term = None
    try:
        with open(os.path.join(run_dir, "census.jsonl"), "a") as out:
            for block in blocks:
                rows = np.asarray(block, np.float64)
                if rows.ndim == 1:
                    rows = rows[None]
                rows = rows[(rows[:, 0] + step_ms) / 1000.0 > soup.t]          # lived through already
                while len(rows):
                    t = (rows[:, 0] + step_ms) / 1000.0
                    if next_census is None:
                        next_census = (int(t[0]) // census_every_s + 1) * census_every_s
                    k = min(len(rows), chunk)
                    if max_rows is not None:
                        k = min(k, max_rows - n)
                    due = np.nonzero(t[:k] >= next_census)[0]
                    if len(due):
                        k = int(due[0]) + 1                  # up to the tick a census falls on
                    soup.advance(rows[:k])
                    rows = rows[k:]
                    n += k
                    if soup.t >= next_census:
                        c = soup.census()
                        out.write(json.dumps(c, default=_plain) + "\n")
                        out.flush()
                        log(status_line(c))
                        while next_census <= soup.t:
                            next_census += census_every_s
                    if time.time() - last_save >= checkpoint_every_s:
                        keep()
                        last_save = time.time()
                    if (max_rows is not None and n >= max_rows) or \
                            (max_wall_s is not None and time.time() - start >= max_wall_s):
                        raise _Enough
            finished = True
    except _Enough:
        pass
    except KeyboardInterrupt:
        log("\ninterrupted; saving the world")
    finally:
        keep()
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
    return n, finished


def read_census(run_dir):
    with open(os.path.join(run_dir, "census.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))
