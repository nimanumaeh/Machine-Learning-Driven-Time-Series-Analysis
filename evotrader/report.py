"""Human-readable views of an ecosystem."""

import time

import numpy as np


def _ts(t):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))


def _pct_day(dlg):
    return "   n/a" if dlg is None else f"{(np.exp(dlg) - 1) * 100:+6.3f}%"


def _ret(pnl, base):
    return f"{pnl / base * 100:+.1f}%" if base else "n/a"


def _tf(s):
    return f"{s}s" if s < 60 else f"{s // 60}m" if s < 3600 else f"{s // 3600}h"


def status_line(world):
    s = world.snapshot()
    return (f"{_ts(s['t'])}  BTC {s['price']:>9.1f}  pop {s['population']:>4}"
            f" (L{s['long']:>3}/S{s['short']:>3})  gen {s['max_generation']:>3}"
            f"  net {s['net_pnl']:>+11.0f} ({_ret(s['net_pnl'], s['injected'])})"
            f"  random {_ret(s['control_net_pnl'], s['control_injected'])}"
            f"  hold {s['bnh_return'] * 100:+.1f}%  liq {s['totals']['liquidations']}")


def full_report(world, store=None, top=10):
    s = world.snapshot()
    out = []
    p = out.append
    p(f"Ecosystem at {_ts(s['t'])} UTC   BTC {s['price']:.1f}  (mark {s['mark']:.1f})")
    p(f"  living agents {s['population']}: {s['long']} long, {s['short']} short,"
      f" {s['population'] - s['long'] - s['short']} flat   (max generation {s['max_generation']})")
    c, tot = s["counts"], s["totals"]
    p(f"  newcomers {c['immigrants']}  born {c['births']}  died {c['starved']}"
      f"  outcompeted {c['outcompeted']}  liquidations {tot['liquidations']}")
    p("")
    p("Play money (USDT):")
    p(f"  handed out {s['injected']:>12.0f}   left by the dead {s['withdrawn']:>10.0f}"
      f"   held now {s['equity']:>10.0f}")
    p(f"  net PnL    {s['net_pnl']:>+12.0f}   ({_ret(s['net_pnl'], s['injected'])} of all capital handed out)")
    p(f"  of which fees paid {tot['fees']:.0f}, funding paid {tot['funding']:+.0f}")
    p(f"  random agents, same rules, no selection: {_ret(s['control_net_pnl'], s['control_injected'])}")
    p(f"  holding 1x BTC since the start:           {s['bnh_return'] * 100:+.1f}%")
    mind = s.get("mind") or {}
    if mind.get("salience_by_receptor"):
        p("")
        g = mind.get("grip")
        p(f"What the agents' own features attend to (grip {'n/a' if g is None else f'{g:+.3f}'}):")
        p("  receptors: " + ", ".join(f"{k} {v:.0%}" for k, v in list(mind["salience_by_receptor"].items())[:8]))
        for name, share in mind["top_programs"][:6]:
            p(f"  {share:>4.0%}  {name}")
    if mind.get("salience_by_stream"):
        p("")
        r3 = lambda v: "n/a" if v is None else f"{v:+.3f}"
        p(f"What the population finds salient (grip on direction {r3(mind.get('grip_drift'))},"
          f" on volatility {r3(mind.get('grip_vol'))}, learning progress {r3(mind['learning_progress'])}):")
        p("  streams: " + ", ".join(f"{k} {v:.0%}" for k, v in list(mind["salience_by_stream"].items())[:8]))
        p("  aspects: " + ", ".join(f"{n} {v:.0%}" for n, v in mind["top_aspects"][:6]))
    p(f"  median growth per day, agents alive >= 1 day: {_pct_day(s['mature_daily_log_growth'])}")
    p("")
    p("Leverage in the living population (median "
      f"{s['median_leverage'] or 0:.0f}x): "
      + "  ".join(f"{k} {v}" for k, v in s["leverage_bands"].items()))
    p("")
    p("Niches:")
    p("  timeframe  agents     equity  median equity/ref")
    for tf, n in s["niches"].items():
        med = "     -" if n["median_ratio"] is None else f"{n['median_ratio']:.3f}"
        p(f"  {_tf(int(tf)):>9}  {n['n']:>6}  {n['equity']:>9.0f}  {med:>17}")
    p("")
    rows = world.leaderboard(top)
    p(f"Top {len(rows)} agents by growth per day (alive >= 1 day):")
    p("     id  lineage  gen   tf   lev  margin  age(d)  lifetime x  per day"
      "   pos BTC  trades  liq  kids")
    for r in rows:
        p(f"  {r['id']:>5}  {r['root']:>7}  {r['generation']:>3}  {_tf(r['timeframe']):>3}"
          f"  {r['leverage']:>3.0f}x  {r['margin_frac']:>6.2f}  {r['age_days']:>6.1f}"
          f"  {r['lifetime_growth']:>10.3f}  {_pct_day(r['daily_log_growth'])}"
          f"  {r['position_btc']:>+8.3f}  {r['trades']:>6}  {r['liquidations']:>3}  {r['children']:>4}")
        attends = r["mind"].get("attends")
        if attends:
            p("         attends to " + ", ".join(f"{n} ({v:.0%})" for n, v in attends))
    wild = world.alive & ~world.control
    if wild.any():
        roots, counts = np.unique(world.root_id[wild], return_counts=True)
        p("")
        p("Largest living lineages:")
        for k in np.argsort(-counts)[:5]:
            line = f"  lineage {roots[k]:>6}: {counts[k]} alive"
            if store is not None:
                total = store.db.execute("SELECT COUNT(*) FROM agents WHERE root=?",
                                         (int(roots[k]),)).fetchone()[0]
                line += f", {total} ever born"
            p(line)
    return "\n".join(out)
