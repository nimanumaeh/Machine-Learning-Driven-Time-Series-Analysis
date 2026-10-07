"""Drive a world through a stream of bars, checkpointing as it goes."""

import time

from .report import status_line


def run(world, rows, store, checkpoint_every_s=300, log=print, max_bars=None):
    """Step `world` through minute `rows` until exhausted, `max_bars`, or Ctrl-C.

    A checkpoint is written every `checkpoint_every_s` wall-clock seconds and on
    exit, so a run can always be resumed. A status line is logged at every
    ecosystem snapshot.
    """
    last_ckpt = time.time()
    n = 0
    try:
        for row in rows:
            if row[0] / 1000.0 + 60 <= world.t:
                continue                       # already seen (resume / overlap)
            world.step(row)
            n += 1
            if world._last_snapshot_t == world.t:
                log(status_line(world))
            if len(world.events) > 1000:
                store.drain(world)
            if time.time() - last_ckpt >= checkpoint_every_s:
                store.checkpoint(world)
                last_ckpt = time.time()
            if max_bars is not None and n >= max_bars:
                break
    except KeyboardInterrupt:
        log("\ninterrupted; saving checkpoint")
    finally:
        store.checkpoint(world)
    return n
