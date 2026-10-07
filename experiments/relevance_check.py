"""Does the relevance-realization loop find what is actually relevant?

Runs relevance-realizing agents on a synthetic market (evotrader/synthetic.py)
where a hidden crowding variable moves the price ("planted") or does not
("null"), with attention reallocated by salience or at random. Logs, day by
day, how much of the population's salience and attention sits on the streams
that carry the crowding signal, plus money and grip. Usage:

    python experiments/relevance_check.py --world planted --attention salience --out run.json
"""

import argparse
import json
import time

import numpy as np

from evotrader.aspects import NAMES, STREAM_OF
from evotrader.config import Config
from evotrader.mind import RRBrain
from evotrader.selfmade import RECEPTORS, SelfMadeBrain
from evotrader.synthetic import synthetic_rows
from evotrader.world import World

CROWD = ("premium", "funding", "basis", "top_position", "top_account")
RISK = np.array([n.startswith(("ret.vol", "range.")) for n in NAMES])
_CROWD_REC = [RECEPTORS.index(n) for n in ("premium", "funding", "top_account_ls", "top_position_ls")]
_PRICE_REC = [RECEPTORS.index(n) for n in ("close", "mark", "high", "low")]
_SPOT = RECEPTORS.index("spot")


def crowd_program(progs):
    """Programs that read a crowding receptor, or the perp-spot basis."""
    a, b = progs[..., 0], progs[..., 1]
    basis = ((a == _SPOT) & np.isin(b, _PRICE_REC)) | ((b == _SPOT) & np.isin(a, _PRICE_REC))
    return np.isin(a, _CROWD_REC) | np.isin(b, _CROWD_REC) | basis


def crowd_shares(w, wild):
    b = w.brain
    if isinstance(b, SelfMadeBrain):
        flag = crowd_program(b.prog[wild])
    else:
        flag = np.isin(STREAM_OF[b.att[wild]], CROWD)
    sal = b.salience[wild]
    return float((sal * flag).sum() / max(sal.sum(), 1e-12)), float(flag.mean())


def summary(w, day):
    s = w.snapshot()
    m = s["mind"] or {}
    wild = np.nonzero(w.alive & ~w.control)[0]
    b = w.brain
    crowd_sal, crowd_att = crowd_shares(w, wild)
    exposure = np.abs(w.exposure(wild))
    if isinstance(b, SelfMadeBrain):
        risk_sal = np.full(len(wild), np.nan)
    else:
        risk_sal = (b.salience[wild] * RISK[b.att[wild]]).sum(1)
    hi, lo = exposure >= 1.0, exposure < 0.1
    return {
        "day": day, "population": s["population"], "generation": s["max_generation"],
        "return": s["net_pnl"] / max(s["injected"], 1), "control_return":
            s["control_net_pnl"] / max(s["control_injected"], 1),
        "fees": s["totals"]["fees"], "liquidations": s["totals"]["liquidations"],
        "births": s["counts"]["births"], "deaths": s["counts"]["starved"],
        "grip": m.get("grip"), "grip_drift": m.get("grip_drift"), "grip_vol": m.get("grip_vol"),
        "crowd_salience": crowd_sal, "crowd_attention": crowd_att,
        "risk_salience_exposed": float(np.nanmean(risk_sal[hi])) if hi.any() and not np.isnan(risk_sal).all() else None,
        "risk_salience_flat": float(np.nanmean(risk_sal[lo])) if lo.any() and not np.isnan(risk_sal).all() else None,
        "top": (m.get("top_programs") or m.get("top_aspects") or [])[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=("planted", "null"), default="planted")
    ap.add_argument("--brain", choices=("rr", "self"), default="rr")
    ap.add_argument("--attention", choices=("salience", "random", "fixed"), default="salience")
    ap.add_argument("--no-anticipation", action="store_true",
                    help="pick new aspects uniformly even in salience mode")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = synthetic_rows(args.days * 1440, planted=args.world == "planted", seed=args.seed)
    cfg = Config(capacity=60, min_population=40, n_control=8, min_per_niche=2,
                 timeframes=(60, 300, 900, 3600), seed=args.seed)
    cfg.mind.attention_mode = args.attention
    cfg.mind.anticipate = not args.no_anticipation
    w = World(cfg, (SelfMadeBrain if args.brain == "self" else RRBrain)(cfg))
    timeline, t0 = [], time.time()
    for k, row in enumerate(rows):
        w.step(row)
        if (k + 1) % 1440 == 0:
            timeline.append(summary(w, (k + 1) // 1440))
            d = timeline[-1]
            print(f"day {d['day']:>3}  pop {d['population']:>3}  gen {d['generation']:>2}"
                  f"  return {d['return']:+.3f} (random {d['control_return']:+.3f})"
                  f"  crowd salience {d['crowd_salience']:.2f} attention {d['crowd_attention']:.2f}"
                  f"  grip {d['grip'] if d['grip'] is None else round(d['grip'], 3)}", flush=True)
    if args.brain == "self":
        rng = np.random.default_rng(0)
        sample = np.array([w.brain._random_program(rng) for _ in range(20000)])
        base = float(crowd_program(sample).mean())
    else:
        base = float(np.isin(STREAM_OF, CROWD).mean())
    result = {"args": vars(args), "crowd_base_rate": base, "seconds": time.time() - t0,
              "timeline": timeline,
              "top_aspects": w.snapshot()["mind"].get("top_aspects") or w.snapshot()["mind"].get("top_programs")}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
    print(f"crowd base rate {base:.0%}; done in {result['seconds']:.0f}s")


if __name__ == "__main__":
    main()
