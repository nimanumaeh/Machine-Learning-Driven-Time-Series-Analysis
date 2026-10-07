"""Human-readable views of an ecosystem."""

import time

import numpy as np


def _ts(t):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))


def _pct_day(dlg):
    return "   n/a" if dlg is None else f"{(np.exp(dlg) - 1) * 100:+6.3f}%"


def _tf(s):
    return f"{s}s" if s < 60 else f"{s // 60}m" if s < 3600 else f"{s // 3600}h"


def status_line(world):
    s = world.snapshot()
    return (f"{_ts(s['t'])}  BTC {s['price']:>9.2f}  pop {s['population']:>4}"
            f"  gen {s['max_generation']:>3}  net {s['net_pnl']:>+11.2f}"
            f"  /day: evolved {_pct_day(s['mature_daily_log_growth'])}"
            f" random {_pct_day(s['control_daily_log_growth'])}"
            f" hold {_pct_day(s['bnh_daily_log_growth'])}")


def full_report(world, store=None, top=10):
    s = world.snapshot()
    out = []
    p = out.append
    p(f"Ecosystem at {_ts(s['t'])} UTC   BTC {s['price']:.2f}")
    p(f"  living agents {s['population']}  (max generation {s['max_generation']})")
    c = s["counts"]
    p(f"  immigrants {c['immigrants']}  births {c['births']}  "
      f"starved {c['starved']}  outcompeted {c['outcompeted']}")
    p(f"  play money: injected {s['injected']:.0f}  withdrawn by deaths {s['withdrawn']:.0f}"
      f"  held {s['equity']:.0f}  ->  net PnL {s['net_pnl']:+.2f}")
    p("")
    p("Median growth per day (agents alive >= 1 day, after fees and metabolism):")
    p(f"  evolved population  {_pct_day(s['mature_daily_log_growth'])}")
    p(f"  random controls     {_pct_day(s['control_daily_log_growth'])}   <- luck baseline")
    p(f"  buy and hold        {_pct_day(s['bnh_daily_log_growth'])}")
    p("")
    p("Niches:")
    p("  timeframe  agents     equity  median equity/ref")
    for tf, n in s["niches"].items():
        med = "     -" if n["median_ratio"] is None else f"{n['median_ratio']:.3f}"
        p(f"  {_tf(int(tf)):>9}  {n['n']:>6}  {n['equity']:>9.0f}  {med:>17}")
    p("")
    rows = world.leaderboard(top)
    p(f"Top {len(rows)} agents by growth/day (alive >= 1 day):")
    p("     id  lineage  gen    tf    age(d)  lifetime x  per day  exposure  trades  kids")
    for r in rows:
        p(f"  {r['id']:>5}  {r['root']:>7}  {r['generation']:>3}  {_tf(r['timeframe']):>4}"
          f"  {r['age_days']:>8.1f}  {r['lifetime_growth']:>10.3f}  {_pct_day(r['daily_log_growth'])}"
          f"  {r['exposure']:>8.2f}  {r['trades']:>6}  {r['children']:>4}")
    wild = world.alive & ~world.control
    if wild.any():
        roots, counts = np.unique(world.root_id[wild], return_counts=True)
        order = np.argsort(-counts)[:5]
        p("")
        p("Largest living lineages:")
        for k in order:
            line = f"  lineage {roots[k]:>6}: {counts[k]} alive"
            if store is not None:
                total = store.db.execute("SELECT COUNT(*) FROM agents WHERE root=?",
                                         (int(roots[k]),)).fetchone()[0]
                line += f", {total} ever born"
            p(line)
    return "\n".join(out)
