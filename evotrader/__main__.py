"""Command line: python -m evotrader <command> ...

  download   build/update the minute store from Binance's free archives
  coverage   what the store holds, month by month
  export     write part of the store to CSV for a look
  synth      synthetic store (planted structure, or --null) for testing
  evolve     let a population live through stored history
  live       carry a run on in real time, forever (history first, then live)
  report     summarise a run
  seconds    build/update the 1-second store from every perpetual trade
  planet     drop very many organisms onto a planet made of 1-second data
  soup       a planet of living matter: nothing designed, BTC as the sun
  view       turn a planet run's census into one self-contained HTML page
"""

import argparse
import datetime as dt
import itertools
import json
import os
import sys
import time

import numpy as np

from . import data, data_seconds, planet_run
from .brains import NetBrain
from .config import Config
from .life import Life
from .planet import Planet
from .soup import PHYSICS, Soup
from .mind import RRBrain
from .selfmade import SelfMadeBrain
from .report import full_report
from .runner import run
from .store import Store, load_world
from .synthetic import SyntheticMarket, synthetic_rows
from .world import World

BRAINS = {"self": SelfMadeBrain, "rr": RRBrain, "net": NetBrain}


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
    g.add_argument("--brain", choices=sorted(BRAINS), default="self",
                   help="self = agents that build their own perception and values; "
                        "rr = relevance realization over our engineered aspects; "
                        "net = evolved fixed nets")
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


def cmd_seconds(args):
    n = data_seconds.build_seconds(args.store, args.minute_store, args.symbol, args.start,
                                   args.until, cache_dir=args.cache)
    print(f"{n} seconds written to {args.store}/{args.symbol}")


def cmd_planet(args):
    ckpt = os.path.join(args.run, "planet.pkl")
    meta_path = os.path.join(args.run, "meta.json")
    if os.path.exists(ckpt):
        if not args.resume:
            sys.exit(f"{args.run} already holds a world: add --resume, or choose another --run")
        planet, life = planet_run.load(ckpt)
        with open(meta_path) as f:
            meta = json.load(f)
        if args.synthetic and meta["source"] == "synthetic":
            meta["days"] = args.synthetic                  # carry a synthetic world on further
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=1)
        print(f"resuming {args.run} at {_when(planet.t)} UTC: {int(life.alive.sum())} organisms")
    else:
        cfg = _config(args)
        planet = Planet(width=args.width, seed=args.seed, base_capacity=args.place_capacity)
        life = Life(cfg, planet, capacity=args.organisms, seed=args.seed,
                    min_population=args.population, higher_order=not args.no_higher_order,
                    culture=not args.no_culture)
        meta = {"source": "synthetic" if args.synthetic else "seconds", "days": args.synthetic,
                "null": args.null, "market_seed": args.market_seed, "store": args.store,
                "symbol": args.symbol, "from": args.start, "until": args.until,
                "width": args.width, "place_capacity": args.place_capacity,
                "organisms": args.organisms, "population": args.population, "seed": args.seed,
                "higher_order": not args.no_higher_order, "culture": not args.no_culture,
                "config": cfg.to_dict(),
                "regions": {k: v.astype(int).tolist() for k, v in planet.regions.items()}}
        os.makedirs(args.run, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    if meta["source"] == "synthetic":
        market = SyntheticMarket(planted=not meta["null"], seed=meta["market_seed"], step_s=1)
        rows = market.stream(int(meta["days"] * 86_400))
    else:
        rows = data_seconds.iter_seconds(meta["store"], meta["symbol"], _ms(meta["from"]),
                                         _ms(meta["until"]))
    print(f"planet {planet.X} x {planet.Y}, up to {life.N} organisms; census every "
          f"{args.census_every} s (Ctrl-C saves; --resume carries on)")
    planet_run.run_planet(planet, life, rows, args.run, census_every_s=args.census_every)


def cmd_soup(args):
    ckpt = os.path.join(args.run, "planet.pkl")
    meta_path = os.path.join(args.run, "meta.json")
    if os.path.exists(ckpt):
        if not args.resume:
            sys.exit(f"{args.run} already holds a world: add --resume, or choose another --run")
        planet, soup = planet_run.load(ckpt)
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("physics", 1) != PHYSICS:
            sys.exit(f"{args.run} was made under soup physics {meta.get('physics', 1)}; this code has "
                     f"physics {PHYSICS}: run it with the code it was made with, or start a new world")
        if args.synthetic and meta["source"] == "synthetic":
            meta["days"] = args.synthetic
        if args.until and meta["source"] == "store":
            meta["until"] = args.until                      # a world can be lived further
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        print(f"resuming {args.run} at {_when(soup.t)} UTC")
    else:
        cfg = _config(args)
        step_s = 60 if args.minutes else 1
        matter = np.load(args.matter) if args.matter else None
        heat = args.heat if args.heat is not None else (2e-5 if args.minutes else 1e-6)
        soup = Soup(cfg, width=args.width, height=args.height, step_s=step_s, seed=args.seed,
                    occupancy=args.occupancy, think=args.think, meet=args.meet, meetings=args.meetings,
                    metabolism_days=args.metabolism, upkeep=args.upkeep, floor=args.floor,
                    birth_min=args.birth_min, divide_at=args.divide_at,
                    mutation=args.mutation, noise=args.noise, heat=heat, digestion=args.digestion,
                    quantum=args.bite, matter=matter, layout=args.layout)
        planet = soup.planet
        store = args.store if args.store != "data/seconds" or not args.minutes else "data/market"
        meta = {"kind": "soup", "physics": PHYSICS, "source": "synthetic" if args.synthetic else "store",
                "resolution_s": step_s, "width": args.width, "height": args.height,
                "occupancy": args.occupancy, "think": args.think, "meet": args.meet,
                "meetings": args.meetings, "metabolism_days": args.metabolism, "upkeep": args.upkeep,
                "floor": args.floor, "divide_at": soup.divide_at,
                "birth_min": args.birth_min, "mutation": args.mutation, "noise": args.noise, "heat": heat,
                "digestion": args.digestion, "bite": args.bite,
                "days": args.synthetic, "null": args.null, "market_seed": args.market_seed,
                "store": store, "symbol": args.symbol, "from": args.start, "until": args.until,
                "matter": args.matter, "layout": args.layout, "seed": args.seed, "config": cfg.to_dict(),
                "regions": {k: v.astype(int).tolist() for k, v in planet.regions.items()}}
        os.makedirs(args.run, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    if args.device == "gpu":
        from .gpu import device_name
        soup.to_gpu(sync=args.sync)
        print(f"on the GPU: {device_name()}")
    step_s = meta.get("resolution_s", 1)
    if meta["source"] == "synthetic":
        market = SyntheticMarket(planted=not meta["null"], seed=meta["market_seed"], step_s=step_s)
        blocks = market.blocks(int(meta["days"] * 86_400 / step_s))
    elif step_s == 60:
        blocks = data.iter_blocks(meta["store"], meta["symbol"], _ms(meta["from"]), _ms(meta["until"]))
    else:
        blocks = data_seconds.iter_second_blocks(meta["store"], meta["symbol"], _ms(meta["from"]),
                                                 _ms(meta["until"]))
    print(f"soup of {soup.S} sites x {soup.soup.shape[1]} bytes on a {planet.X} x {planet.Y} torus, "
          f"{int(soup.alive.sum())} alive, one tick every {planet.step_s}s (Ctrl-C saves; --resume carries on)")
    tic = time.time()
    n, finished = planet_run.run_soup(soup, blocks, args.run, census_every_s=args.census_every,
                                      max_wall_s=args.max_hours * 3600 if args.max_hours else None)
    took = time.time() - tic
    print(f"{n} ticks in {took:.0f}s ({1000 * took / max(n, 1):.3f} ms a tick); "
          + ("the data ran out" if finished else "stopped early: --resume carries on"))
    return finished


def cmd_view(args):
    from .viewer import is_soup, write_soup_viewer, write_viewer
    soups = [is_soup(r) for r in args.run]
    if any(soups) and not all(soups):
        sys.exit("view: give either planet runs or soup runs, not both")
    out = args.out or os.path.join(args.run[0], "soup.html" if soups[0] else "planet.html")
    n = (write_soup_viewer if soups[0] else write_viewer)(args.run, out, max_frames=args.frames)
    print(f"{n} census frames per world written to {out}")


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

    p = sub.add_parser("seconds", help="build/update the 1-second store from every trade")
    store_args(p, default_store="data/seconds")
    p.add_argument("--minute-store", default="data/market",
                   help="minute store that supplies the slow streams (download it first)")
    p.add_argument("--from", dest="start", default="2019-09-08")
    p.add_argument("--until", help="stop before this day")
    p.add_argument("--cache", default=None,
                   help="keep the raw trade archives here (about 10 MB a day); by default they are not kept")
    p.set_defaults(fn=cmd_seconds)

    p = sub.add_parser("planet", help="let very many organisms live on a planet of 1-second data")
    store_args(p, default_store="data/seconds")
    p.add_argument("--run", default="runs/planet")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--from", dest="start", help="first day of the 1-second store to live through")
    p.add_argument("--until", help="stop before this day")
    p.add_argument("--synthetic", type=float, metavar="DAYS",
                   help="live on a synthetic market for DAYS days instead of the store")
    p.add_argument("--null", action="store_true", help="synthetic market with nothing planted")
    p.add_argument("--market-seed", type=int, default=0)
    p.add_argument("--width", type=int, default=32, help="longitudes (latitudes are the 8 timescales)")
    p.add_argument("--place-capacity", type=int, default=12,
                   help="organisms a place holds at normal water (liquidity)")
    p.add_argument("--population", type=int, default=1000,
                   help="below this, newcomers arrive from space (panspermia)")
    p.add_argument("--organisms", type=int, default=8192, help="most organisms that can ever be alive")
    p.add_argument("--census-every", type=int, default=1800, help="simulated seconds between censuses")
    p.add_argument("--no-higher-order", action="store_true",
                   help="ablation: the gate never opens (habits and blind generate-and-test only)")
    p.add_argument("--no-culture", action="store_true",
                   help="ablation: no observatories or markers to build, perceive or follow")
    g = p.add_argument_group("environment (defaults in evotrader/config.py)")
    g.add_argument("--taker-fee", dest="taker_fee", type=float)
    g.add_argument("--half-spread", dest="half_spread", type=float)
    g.add_argument("--max-leverage", dest="max_leverage", type=int)
    g.add_argument("--initial-capital", dest="initial_capital", type=float)
    g.add_argument("--death-ratio", dest="death_ratio", type=float)
    g.add_argument("--repro-ratio", dest="repro_ratio", type=float)
    g.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_planet)

    p = sub.add_parser("soup", help="living matter on the market: only trading keeps it alive")
    store_args(p, default_store="data/seconds")
    p.add_argument("--run", default="runs/soup")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--from", dest="start", help="first day of the 1-second store")
    p.add_argument("--until", help="stop before this day")
    p.add_argument("--synthetic", type=float, metavar="DAYS")
    p.add_argument("--null", action="store_true")
    p.add_argument("--market-seed", type=int, default=0)
    p.add_argument("--width", type=int, default=64, help="sites around the torus")
    p.add_argument("--height", type=int, default=64, help="sites across the torus")
    p.add_argument("--occupancy", type=float, default=0.5, help="share of sites alive at the start")
    p.add_argument("--minutes", action="store_true",
                   help="live on the minute store (2017 on), a tick a minute; otherwise a tick a second")
    p.add_argument("--metabolism", type=float, default=365.0,
                   help="days in which an organism holding one stake and earning nothing loses half of it "
                        "to Kleiber's law (energy ** 0.75)")
    p.add_argument("--upkeep", type=float, default=1.0,
                   help="what a body costs to keep, whatever it holds (USDT a day)")
    p.add_argument("--floor", type=float, default=1.0, help="an organism whose equity falls to this dies (USDT)")
    p.add_argument("--birth-min", type=float, default=100.0, help="least energy a child is born with (USDT)")
    p.add_argument("--divide-at", type=float, default=0.0,
                   help="a body that grows to this much energy (USDT) divides in half (default: two stakes)")
    p.add_argument("--mutation", type=float, default=1 / 64, help="chance each byte is miscopied at birth")
    p.add_argument("--noise", type=float, default=1e-3, help="chance a byte flips, per byte per day")
    p.add_argument("--think", type=int, default=32, help="instructions an organism's matter runs each tick")
    p.add_argument("--meet", type=int, default=128, help="instructions a meeting of two runs")
    p.add_argument("--meetings", type=float, default=1 / 16,
                   help="organisms that wake to meet a neighbor per tick, as a share of all sites, at one stake")
    p.add_argument("--heat", type=float, default=None,
                   help="energy (USDT) each byte written costs (default 2e-5 on minutes, 1e-6 on seconds)")
    p.add_argument("--bite", type=float, default=1.0, help="most energy (USDT) one T instruction moves")
    p.add_argument("--digestion", type=float, default=0.8,
                   help="share of energy taken from (or given to) another organism that arrives")
    p.add_argument("--matter", help="start from this matter (.npy of tapes, tiled over the torus) instead of random")
    p.add_argument("--layout", choices=("symmetric", "bff"), default="symmetric",
                   help="which byte means which instruction: symmetric (no lean to long or short) or the "
                        "ASCII bytes of BFF (to seed with matter from experiments/soup_emergence.py)")
    p.add_argument("--census-every", type=int, default=1800)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                   help="where the world runs (gpu: CUDA through numba; see docs/gpu.md)")
    p.add_argument("--sync", choices=("grid", "launch"), default="grid",
                   help="on the GPU: one cooperative launch per chunk (grid) or three launches a tick")
    p.add_argument("--max-hours", type=float, help="stop (and save) after this much wall time")
    g = p.add_argument_group("environment (defaults in evotrader/config.py)")
    g.add_argument("--taker-fee", dest="taker_fee", type=float)
    g.add_argument("--half-spread", dest="half_spread", type=float)
    g.add_argument("--max-leverage", dest="max_leverage", type=int)
    g.add_argument("--initial-capital", dest="initial_capital", type=float)
    g.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_soup)

    p = sub.add_parser("view", help="one HTML page to watch planet runs")
    p.add_argument("--run", nargs="+", default=["runs/planet"],
                   help="one run, or several to compare side by side")
    p.add_argument("--out")
    p.add_argument("--frames", type=int, default=400, help="most census frames to include")
    p.set_defaults(fn=cmd_view)

    args = ap.parse_args(argv)
    if args.cmd in ("planet", "soup") and not args.synthetic and not args.resume and not args.start:
        ap.error(f"{args.cmd}: give --synthetic DAYS, or --from DAY for the 1-second store")
    return args.fn(args)


if __name__ == "__main__":
    main()
    sys.exit(0)
