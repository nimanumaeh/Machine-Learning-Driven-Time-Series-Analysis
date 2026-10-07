"""Command line: python -m evotrader {synth,download,evolve,live,report} ..."""

import argparse
import os
import sys
import time

from . import data
from .config import Config
from .features import HISTORY
from .report import full_report
from .runner import run
from .store import Store, load_world
from .world import World


def _config(args):
    cfg = Config()
    for name in ("capacity", "fee_bps", "slippage_bps", "metabolism_bps_per_day",
                 "death_ratio", "repro_ratio", "n_control", "seed"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    if getattr(args, "allow_short", False):
        cfg.allow_short = True
    return cfg


def _add_config_args(p):
    g = p.add_argument_group("ecosystem (defaults in evotrader/config.py)")
    g.add_argument("--capacity", type=int)
    g.add_argument("--fee-bps", dest="fee_bps", type=float)
    g.add_argument("--slippage-bps", dest="slippage_bps", type=float)
    g.add_argument("--metabolism", dest="metabolism_bps_per_day", type=float,
                   help="cost of living, basis points of equity per day")
    g.add_argument("--death-ratio", dest="death_ratio", type=float)
    g.add_argument("--repro-ratio", dest="repro_ratio", type=float)
    g.add_argument("--controls", dest="n_control", type=int)
    g.add_argument("--seed", type=int)
    g.add_argument("--allow-short", action="store_true")


def cmd_synth(args):
    n = int(args.days * 86400 / args.seconds)
    data.write_csv(args.out, data.synthetic_bars(n, args.seconds, seed=args.seed))
    print(f"wrote {n} synthetic {args.seconds}s bars to {args.out}")


def cmd_download(args):
    out = args.out or f"data/{args.symbol}_{data.interval_name(args.seconds)}.csv"
    n = data.download(out, args.symbol, args.seconds, args.days, source=args.source)
    print(f"{n} new bars appended to {out}")


def cmd_evolve(args):
    store = Store(args.run)
    if args.resume and os.path.exists(store.checkpoint_path):
        world = load_world(store.checkpoint_path)
        print(f"resuming {args.run} at {time.strftime('%Y-%m-%d %H:%M', time.gmtime(world.t))}")
    else:
        cfg = _config(args)
        world = World(cfg, data.detect_bar_seconds(args.data))
        store.set_meta("config", cfg.to_dict())
    store.set_meta("mode", "replay")
    run(world, data.iter_csv(args.data), store)
    print()
    print(full_report(world, store))
    store.close()


def _warm_up(world, client, symbol, seconds):
    """Prime every niche's candle history with recent real data."""
    now = int(time.time() * 1000)
    fine_start = (now - 40 * 60 * 1000) // 60000 * 60000
    coarse_tf = [tf for tf in world.timeframes if tf % 60 == 0]
    if coarse_tf:
        span = (HISTORY + 2) * max(coarse_tf) * 1000
        for b in data.api_bars(client, symbol, 60, fine_start - span, fine_start):
            world.warm(b[0], 60000, *b[1:])
    last = fine_start - seconds * 1000
    for b in data.api_bars(client, symbol, seconds, fine_start, now - seconds * 1000):
        world.warm(b[0], seconds * 1000, *b[1:])
        last = b[0]
    world.t = (last + seconds * 1000) / 1000.0
    return last + seconds * 1000


def cmd_live(args):
    store = Store(args.run)
    client = data.BinanceClient()
    if os.path.exists(store.checkpoint_path):
        world = load_world(store.checkpoint_path)
        start = int(world.t * 1000)
        print(f"resuming live arena; catching up from "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(world.t))} UTC")
    else:
        if args.seed_from:
            world = load_world(args.seed_from)
            print(f"seeding from {args.seed_from}: {int((world.alive & ~world.control).sum())} agents")
        else:
            world = World(_config(args), args.seconds)
        world.reset_market(args.seconds)
        print("warming up candle histories from recent data...")
        start = _warm_up(world, client, args.symbol, args.seconds)
        if args.seed_from:
            world.reset_economy()
        store.set_meta("config", world.cfg.to_dict())
    store.set_meta("mode", "live")
    print("live paper arena running (Ctrl-C to stop; rerun the same command to resume)")
    run(world, data.live_bars(args.symbol, args.seconds, start_ms=start, client=client), store)
    print(full_report(world, store))
    store.close()


def cmd_report(args):
    store = Store(args.run)
    world = load_world(store.checkpoint_path)
    print(full_report(world, store, top=args.top))
    store.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evotrader", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="generate synthetic BTC-like bars for offline testing")
    p.add_argument("--out", default="data/synthetic_1m.csv")
    p.add_argument("--days", type=float, default=60)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("download", help="download historical Binance klines to CSV")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--seconds", type=int, default=60, help="bar size: 1 or 60 (1s / 1m)")
    p.add_argument("--days", type=float, default=90)
    p.add_argument("--source", choices=("auto", "archive", "api"), default="auto",
                   help="archive = data.binance.vision bulk zips, api = REST")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("evolve", help="evolve a population on historical bars (the nursery)")
    p.add_argument("--data", required=True)
    p.add_argument("--run", default="runs/nursery")
    p.add_argument("--resume", action="store_true")
    _add_config_args(p)
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser("live", help="run the population on live Binance data, forever")
    p.add_argument("--run", default="runs/live")
    p.add_argument("--seed-from", help="nursery run dir whose survivors move in")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--seconds", type=int, default=1, help="base bar size of the feed")
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
