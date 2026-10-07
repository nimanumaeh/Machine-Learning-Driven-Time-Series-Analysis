"""Command line: python -m evotrader <command> ...

  download   build/update the minute store from Binance's free archives
  coverage   what the store holds, month by month
  export     write part of the store to CSV for a look
  synth      synthetic store (planted structure, or --null) for testing
  evolve     let a population live through stored history
  live       carry a run on in real time, forever (history first, then live)
  report     summarise a run
"""

import argparse
import datetime as dt
import itertools
import os
import sys
import time

import numpy as np

from . import data
from .brains import NetBrain
from .config import Config
from .mind import RRBrain
from .report import full_report
from .runner import run
from .store import Store, load_world
from .synthetic import synthetic_rows
from .world import World

BRAINS = {"rr": RRBrain, "net": NetBrain}


def _ms(day):
    if not day:
        return None
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def _when(t):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))


def _config(args):
    cfg = Config()
    for name in ("capacity", "taker_fee", "half_spread", "max_leverage", "initial_capital",
                 "death_ratio", "repro_ratio", "n_control", "min_population", "seed"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    if getattr(args, "attention_mode", None):
        cfg.mind.attention_mode = args.attention_mode
    return cfg


def _add_config_args(p):
    g = p.add_argument_group("environment (defaults in evotrader/config.py)")
    g.add_argument("--brain", choices=sorted(BRAINS), default="rr",
                   help="rr = relevance-realizing agents, net = evolved fixed nets")
    g.add_argument("--attention-mode", choices=("salience", "random", "fixed"),
                   help="rr ablation: how attention is reallocated")
    g.add_argument("--taker-fee", dest="taker_fee", type=float,
                   help="fraction of notional per fill, e.g. 0.0005 for 0.05%%")
    g.add_argument("--half-spread", dest="half_spread", type=float, help="USDT")
    g.add_argument("--max-leverage", dest="max_leverage", type=int)
    g.add_argument("--initial-capital", dest="initial_capital", type=float)
    g.add_argument("--capacity", type=int)
    g.add_argument("--population", dest="min_population", type=int)
    g.add_argument("--death-ratio", dest="death_ratio", type=float)
    g.add_argument("--repro-ratio", dest="repro_ratio", type=float)
    g.add_argument("--controls", dest="n_control", type=int)
    g.add_argument("--seed", type=int)


def _new_world(args):
    cfg = _config(args)
    return World(cfg, BRAINS[args.brain](cfg))


def cmd_download(args):
    n = data.build_market(args.store, args.symbol, args.start, cache_dir=args.cache,
                          assumed_funding=args.assumed_funding)
    print(f"{n} minutes written to {args.store}/{args.symbol}")


def cmd_coverage(args):
    print("month     minutes  perp   premium  spot  open-int  positioning")
    for month, n, perp, present in data.coverage(args.store, args.symbol):
        print(f"{month}  {n:>8}  {perp:>4.0%}  {present['premium']:>7.0%}  {present['spot_close']:>4.0%}"
              f"  {present['open_interest']:>8.0%}  {present['top_position_ls']:>11.0%}")


def cmd_export(args):
    rows = np.array(list(data.iter_rows(args.store, args.symbol, _ms(args.start), _ms(args.until))))
    data.export_csv(args.out, rows)
    print(f"{len(rows)} minutes written to {args.out}")


def cmd_synth(args):
    rows = synthetic_rows(int(args.days * 1440), planted=not args.null, seed=args.seed)
    data.save_store(args.store, args.symbol, rows)
    kind = "null (no planted effect)" if args.null else "planted crowding effect"
    print(f"{len(rows)} synthetic minutes, {kind}, written to {args.store}/{args.symbol}")


def cmd_evolve(args):
    store = Store(args.run)
    if args.resume and os.path.exists(store.checkpoint_path):
        world = load_world(store.checkpoint_path)
        print(f"resuming {args.run} at {_when(world.t)} UTC")
    else:
        world = _new_world(args)
        store.set_meta("config", world.cfg.to_dict())
        store.set_meta("brain", args.brain)
    store.set_meta("mode", "history")
    run(world, data.iter_rows(args.store, args.symbol, _ms(args.start), _ms(args.until)), store)
    print()
    print(full_report(world, store))
    store.close()


def cmd_live(args):
    print("bringing the archive up to yesterday...")
    data.build_market(args.store, args.symbol, args.start, cache_dir=args.cache)
    files = data.month_files(args.store, args.symbol)
    if not files:
        sys.exit("no stored data; check network access to data.binance.vision")
    store = Store(args.run)
    if os.path.exists(store.checkpoint_path):
        world = load_world(store.checkpoint_path)
        print(f"resuming {args.run} at {_when(world.t)} UTC")
        history = data.iter_rows(args.store, args.symbol, int(world.t * 1000))
    else:
        world = _new_world(args)
        store.set_meta("config", world.cfg.to_dict())
        store.set_meta("brain", args.brain)
        with np.load(files[-1]) as z:
            end = int(z["bars"][-1, 0])
        warm_from = end - args.warm_days * 86_400_000
        print(f"warming up perception on the last {args.warm_days} days...")
        for row in data.iter_rows(args.store, args.symbol, warm_from):
            world.warm(row)
        world.t = (end + data.MINUTE) / 1000.0
        history = iter(())
    store.set_meta("mode", "live")
    rows = itertools.chain(history, _live_tail(world, args))
    print("live paper arena running (Ctrl-C to stop; rerun the same command to resume)")
    run(world, rows, store)
    print(full_report(world, store))
    store.close()


def _live_tail(world, args):
    """Live rows starting right after whatever the world has already seen."""
    yield from data.live_rows(int(world.t * 1000), args.symbol)


def cmd_report(args):
    store = Store(args.run)
    world = load_world(store.checkpoint_path)
    print(full_report(world, store, top=args.top))
    store.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evotrader", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def store_args(p, default_symbol="BTCUSDT", default_store="data/market"):
        p.add_argument("--store", default=default_store)
        p.add_argument("--symbol", default=default_symbol)

    p = sub.add_parser("download", help="build/update the minute store from Binance archives")
    store_args(p)
    p.add_argument("--from", dest="start", default="2017-08")
    p.add_argument("--assumed-funding", type=float, default=0.0001,
                   help="funding per 8h charged before the perpetual listed (2017-2019)")
    p.add_argument("--cache", default="data/raw", help="where raw archives are kept")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("coverage", help="what the store holds, month by month")
    store_args(p)
    p.set_defaults(fn=cmd_coverage)

    p = sub.add_parser("export", help="write part of the store to CSV")
    store_args(p)
    p.add_argument("--from", dest="start")
    p.add_argument("--until")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("synth", help="synthetic store for testing the machinery")
    store_args(p, "SYNTH", "data/synthetic")
    p.add_argument("--days", type=float, default=90)
    p.add_argument("--null", action="store_true", help="no planted effect")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("evolve", help="let a population live through stored history")
    store_args(p)
    p.add_argument("--run", default="runs/history")
    p.add_argument("--from", dest="start", help="first day to use, e.g. 2019-09-08")
    p.add_argument("--until", help="stop before this day (keep later data unseen)")
    p.add_argument("--resume", action="store_true")
    _add_config_args(p)
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser("live", help="carry a run on in real time, forever")
    store_args(p)
    p.add_argument("--run", default="runs/live")
    p.add_argument("--from", dest="start", default="2017-08", help="archive start")
    p.add_argument("--cache", default="data/raw")
    p.add_argument("--warm-days", type=int, default=30,
                   help="history a new run perceives before acting")
    _add_config_args(p)
    p.set_defaults(fn=cmd_live)

    p = sub.add_parser("report", help="summarise a run")
    p.add_argument("--run", required=True)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
