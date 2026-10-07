"""Command line: python -m evotrader {download,synth,evolve,live,report} ..."""

import argparse
import datetime as dt
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


def _ms(day):
    if not day:
        return None
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def _when(t):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))


def _config(args):
    cfg = Config()
    for name in ("capacity", "taker_fee", "half_spread", "max_leverage", "initial_capital",
                 "death_ratio", "repro_ratio", "n_control", "seed"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    return cfg


def _add_config_args(p):
    g = p.add_argument_group("environment (defaults in evotrader/config.py)")
    g.add_argument("--taker-fee", dest="taker_fee", type=float,
                   help="fraction of notional per fill, e.g. 0.0005 for 0.05%%")
    g.add_argument("--half-spread", dest="half_spread", type=float, help="USDT")
    g.add_argument("--max-leverage", dest="max_leverage", type=int)
    g.add_argument("--initial-capital", dest="initial_capital", type=float)
    g.add_argument("--capacity", type=int)
    g.add_argument("--death-ratio", dest="death_ratio", type=float)
    g.add_argument("--repro-ratio", dest="repro_ratio", type=float)
    g.add_argument("--controls", dest="n_control", type=int)
    g.add_argument("--seed", type=int)


def cmd_download(args):
    n = data.build_market(args.out, start=args.start, assumed_funding=args.assumed_funding,
                          cache_dir=args.cache)
    print(f"{n} new bars appended to {args.out}")


def cmd_synth(args):
    n = int(args.days * 1440)
    data.write_csv(args.out, (b + ("synthetic",) for b in data.synthetic_bars(n, seed=args.seed)))
    print(f"wrote {n} synthetic 1-minute bars to {args.out}")


def cmd_evolve(args):
    store = Store(args.run)
    if args.resume and os.path.exists(store.checkpoint_path):
        world = load_world(store.checkpoint_path)
        print(f"resuming {args.run} at {_when(world.t)} UTC")
    else:
        cfg = _config(args)
        world = World(cfg, data.detect_bar_seconds(args.data))
        store.set_meta("config", cfg.to_dict())
    store.set_meta("mode", "history")
    run(world, data.iter_csv(args.data, _ms(args.start), _ms(args.until)), store)
    print()
    print(full_report(world, store))
    store.close()


def cmd_live(args):
    store = Store(args.run)
    client = data.BinanceClient()
    if os.path.exists(store.checkpoint_path):
        world = load_world(store.checkpoint_path)
        start = int(world.t * 1000)
        print(f"resuming {args.run}; catching up from {_when(world.t)} UTC")
    else:
        if args.seed_from:
            world = load_world(args.seed_from)
            world.reset_market(60)
            print(f"seeding from {args.seed_from}: {int((world.alive & ~world.control).sum())} agents")
        else:
            world = World(_config(args), 60)
        print("warming up candle histories from recent data...")
        bars = data.recent_bars((HISTORY + 2) * max(world.timeframes) // 60, client=client)
        for b in bars:
            world.warm(b[0], data.MINUTE, *b[1:6])
        start = bars[-1][0] + data.MINUTE
        world.t = start / 1000.0
        if args.seed_from:
            world.reset_economy()
        store.set_meta("config", world.cfg.to_dict())
    store.set_meta("mode", "live")
    print("live paper arena running (Ctrl-C to stop; rerun the same command to resume)")
    run(world, data.live_bars(start, client=client), store)
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

    p = sub.add_parser("download", help="build the 1-minute BTCUSDT perpetual data set")
    p.add_argument("--from", dest="start", default="2017-08-01")
    p.add_argument("--out", default="data/BTCUSDT_perp_1m.csv")
    p.add_argument("--assumed-funding", type=float, default=0.0001,
                   help="funding per 8h charged before the perpetual existed (2017-2019)")
    p.add_argument("--cache", default="data/raw", help="where raw archives are kept")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("synth", help="synthetic BTC-like bars for testing offline")
    p.add_argument("--out", default="data/synthetic_1m.csv")
    p.add_argument("--days", type=float, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("evolve", help="let a population live through historical bars")
    p.add_argument("--data", required=True)
    p.add_argument("--run", default="runs/history")
    p.add_argument("--from", dest="start", help="first day to use, e.g. 2017-08-17")
    p.add_argument("--until", help="stop before this day (keep later data unseen)")
    p.add_argument("--resume", action="store_true")
    _add_config_args(p)
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser("live", help="carry on in real time with live Binance data, forever")
    p.add_argument("--run", default="runs/history",
                   help="run to continue (catches up from where it stopped)")
    p.add_argument("--seed-from", help="start a new run from this run's agents, fresh wallets")
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
