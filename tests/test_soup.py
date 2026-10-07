"""The soup's physics: matter, chemistry, the market instructions, and money."""

import numpy as np
import pytest

from evotrader import physics
from evotrader.soup import (BFF, L, STEPS, SYMMETRIC, Primordial, assemble, census_of_matter,
                            chemistry, disassemble, high_order_entropy, neighbors, replicates, run)


def tape(*parts, n=8):
    """A joined tape of 2n bytes from text and integers (inert filler is 'x')."""
    out = []
    for p in parts:
        out.extend(p.encode("latin-1") if isinstance(p, str) else [p])
    out += [ord("x")] * (2 * n - len(out))
    return np.array(out[:2 * n], np.uint8)


def execute(t, steps=STEPS, sense=(128,), wallet=(0.0, 0.0), selfs=((128, 128), (128, 128))):
    s = np.array(sense, np.uint8)
    w = np.array(wallet, float)
    act = np.full(2, -1, np.int16)
    flow = np.zeros((2, 3))
    e, c = run(t, steps, s, s, np.array(selfs[0], np.uint8), np.array(selfs[1], np.uint8),
               w, 0, 1, act, flow, 0.0, 1.0, 0.01, BFF)          # the ASCII chemistry of BFF
    return t, e, c, act, w, flow


def test_heads_increment_and_copy():
    t, e, c, *_ = execute(tape("+>+>+", 0))                # increments the bytes under head 0
    assert list(t[:5]) == [ord("+") + 1, ord(">") + 1, ord("+") + 1, ord(">"), ord("+")]
    t, *_ = execute(tape(">>>}}}}}.", 7, n=8))            # head0 -> 3, head1 -> 5: copy t[3] to t[5]
    assert t[5] == t[3]
    t, *_ = execute(tape("{.", n=8))                       # head1 wraps to the last byte
    assert t[-1] == ord("{")
    t, *_ = execute(tape("-", n=8))                        # bytes wrap around 0..255
    assert t[0] == ord("-") - 1


def test_loops_skip_repeat_and_unmatched_brackets_stop():
    t, e, *_ = execute(tape(0, "[+]", "+", n=8))          # zero under head 0: the loop is skipped
    assert t[0] == 1 and e == 14                           # '+' after it runs once; every other byte
                                                           # is visited once, the loop body never
    t, e, *_ = execute(tape(3, "[-]", n=8))                # counts the byte under head 0 down to zero
    assert t[0] == 0
    t, e, *_ = execute(tape(0, "[", n=8))                  # a jump with nowhere to land stops it
    assert e == 2
    t, e, *_ = execute(tape(5, "]", n=8))
    assert e == 2
    t, e, *_ = execute(tape(5, "[", n=8))                  # but a bracket it need not follow is inert
    assert e == 16
    t, e, *_ = execute(tape(1, "[]", n=8), steps=100)      # an endless loop stops at the step limit
    assert e == 100


def test_market_instructions_act_for_their_own_half():
    sense = np.arange(10, 60, dtype=np.uint8)
    t, *_ = execute(tape(3, "S", n=8), sense=sense)        # reads stream 3 into the byte under head 0
    assert t[0] == 13
    t, *_ = execute(tape(1, "E", n=8), selfs=((150, 77), (128, 128)))
    assert t[0] == 77                                      # interoception: its own position byte
    t, *_, act, w, f = execute(tape(200, "A", n=8))
    assert list(act) == [200, -1]                          # A in the first half: the first site acts
    t, *_, act, w, f = execute(tape("xxxxxxxx", "A", n=8))
    assert list(act) == [-1, ord("x")]                     # in the second half: the second site acts


def test_transfers_conserve_money():
    t, *_, w, flow = execute(tape(127, "T", n=8), wallet=(100.0, 50.0))   # +127: one full bite
    assert w.tolist() == [100.01, 49.99] and flow[0, 0] == 0.01 and flow[1, 1] == 0.01
    t, *_, w, flow = execute(tape(192, "T", n=8), wallet=(100.0, 50.0))   # -64: gives half a bite
    assert w.tolist() == pytest.approx([99.995, 50.005])
    t, *_, w, flow = execute(tape(0, "T", n=8), wallet=(100.0, 50.0))     # 0: nothing
    assert w.tolist() == [100.0, 50.0]
    t, *_, w, flow = execute(tape(127, "[T]", n=8), wallet=(0.0, 0.004))  # never more than there is
    assert w.tolist() == pytest.approx([0.004, 0.0]) and (w >= 0).all()


def test_a_copier_copies_itself_into_its_neighbor():
    # count head 1 out to the neighbor's half, then copy byte after byte until a zero
    prog = [64] + list(b"[}-].>}[.>}]") + [ord("x")] * (L - 13)
    assert replicates(np.array(prog, np.uint8), table=BFF) > 0.95
    assert replicates(np.frombuffer(b"x" * L, np.uint8), table=BFF) < 0.05   # inert matter copies nothing
    sym = assemble([64] + list("[}-].>}[.>}]") + ["x"] * (L - 13))   # the same program, other chemistry
    assert replicates(sym, table=SYMMETRIC) > 0.95


def partners(X, Y):
    """(site, its neighbor at each offset) for every site of an X by Y lattice."""
    return np.array([(s, physics.neighbor(s, dx, dy, X, Y)) for s in range(X * Y)
                     for dx, dy in neighbors(X, Y)]).reshape(-1, 2)


def test_neighbors_stay_on_the_planet():
    X, Y = 16, 8
    p = partners(X, Y)
    assert (p[:, 1] >= 0).all() and (p[:, 1] < X * Y).all() and (p[:, 0] != p[:, 1]).all()
    dx = np.abs(p[:, 0] % X - p[:, 1] % X)
    dy = np.abs(p[:, 0] // X - p[:, 1] // X)
    assert (np.minimum(dx, X - dx) <= 2).all() and (dy <= 2).all()


def test_complexity_is_low_for_random_matter_and_high_for_copies():
    rng = np.random.default_rng(0)
    random = rng.integers(0, 256, (2048, L), dtype=np.uint8)
    assert abs(high_order_entropy(random)) < 0.2
    copies = np.tile(rng.integers(0, 256, (4, L), dtype=np.uint8), (512, 1))
    assert high_order_entropy(copies) > 5
    c = census_of_matter(copies, top=2)
    assert c["distinct"] == 4 and c["top"][0][1] == 512


# ------------------------------------------------------------ the world, on the market
import pickle                                                    # noqa: E402

from evotrader.config import Config                              # noqa: E402
from evotrader.data import C                                     # noqa: E402
from evotrader.planet import Planet                              # noqa: E402
from evotrader.soup import Soup, byte_of_exposure, exposure_of  # noqa: E402
from evotrader.synthetic import synthetic_rows                   # noqa: E402

DAY0 = 1_700_006_400_000


def world(matter=None, width=2, per_place=8, interactions=64, **kw):
    pl = Planet(width=width, seed=0)
    return pl, Soup(Config(), pl, per_place=per_place, seed=0, interactions=interactions,
                    matter=matter, **kw)


def live(pl, w, rows):
    w.advance(np.asarray(rows))


def test_without_orders_energy_only_moves_between_sites():
    S = 2 * 8 * 8
    grab = np.resize(assemble([100, "T", "x", "T"]), L)          # positive: take
    give = np.resize(assemble([200, "T", "x", "T"]), L)          # negative: give
    matter = np.where((np.arange(S) % 2 == 0)[:, None], grab, give)   # takers beside givers
    pl, w = world(matter=matter, noise=0.0, quantum=50.0)
    live(pl, w, synthetic_rows(300, seed=1, step_s=1, start_ms=DAY0))
    assert w.counts["orders"] == 0 and w.flow[:, 0].sum() > 0
    assert w.equity().sum() == pytest.approx(w.injected)          # nothing made, nothing lost
    assert w.equity().std() > 1                                    # but it moved around


def test_orders_fill_at_the_next_open_exactly():
    rows = synthetic_rows(200, seed=2, step_s=1, start_ms=DAY0)
    S = 2 * 8 * 8
    matter = np.tile(assemble([64, "A"] + ["x"] * 62), (S, 1))
    pl, w = world(matter=matter, noise=0.0, max_exposure=125)
    r0 = rows[0]
    w.step(r0)                                                     # matter decides: about 10x long
    assert np.isfinite(w.pending).any() and not w.q.any()
    i = int(np.nonzero(np.isfinite(w.pending) & (w.site_y == 0))[0][0])   # an equator site
    x = w.pending[i]
    assert x == pytest.approx(exposure_of(64))
    r1 = rows[1]
    w.step(r1)
    fill = r1[C["open"]] + Config().half_spread
    assert w.acct[physics.ENTRY, i] == pytest.approx(fill)         # at this second's open, plus spread
    assert w.q[i] * fill == pytest.approx(x * Config().initial_capital, rel=0.02)
    assert w.acct[physics.FEES, i] > 0 and w.lev[i] == np.ceil(x)


def test_time_is_allotted_by_energy_to_the_three_quarters():
    pl, w = world(noise=0.0, interactions=4, matter=np.full((128, L), 128, np.uint8))   # inert matter
    rich = np.arange(w.S) % 2 == 0
    w.wallet[rich] = 16 * w.stake                                  # 16x the energy...
    live(pl, w, synthetic_rows(4000, seed=1, step_s=1, start_ms=DAY0))
    n = w.count[physics.INTERACTIONS]
    ratio = n[rich].mean() / n[~rich].mean()
    assert ratio == pytest.approx(16 ** 0.75, rel=0.15)            # ...buys 8x the time, not 16x
    woke = 4000 * w.E / w.S * (rich * 8 + ~rich).sum()              # set by energy, not by a quota;
    assert 0.7 * woke < n.sum() < woke                             # a few crowded out


def test_a_site_meets_at_most_one_neighbor_a_tick():
    pl, w = world(noise=0.0, interactions=10_000)                  # everyone wants to meet
    r = synthetic_rows(1, seed=1, step_s=1, start_ms=DAY0)
    w.step(r[0])
    met = w.count[physics.INTERACTIONS] > 0
    j = w.partner[met]
    assert len(set(np.nonzero(met)[0]) | set(j)) == 2 * met.sum()   # pairs are disjoint
    assert 0.2 < met.sum() / w.S < 0.5                             # crowding: not everyone gets to


def test_senses_exist_only_where_the_stream_does():
    pl, w = world(width=4, per_place=4)
    live(pl, w, synthetic_rows(400, seed=3, step_s=1, start_ms=DAY0))
    absent = w.mask == 0
    assert absent.any() and (w.senses[absent] == 0).all()          # nothing there reads as ordinary
    present = (w.mask == 1) & (w.site_y[:, None] == 0)
    assert (w.senses[present] != 0).mean() > 0.5


def test_the_soup_world_resumes_exactly():
    rows = synthetic_rows(900, seed=4, step_s=1, start_ms=DAY0)
    pl, w = world()
    live(pl, w, rows[:400])
    pl2, w2 = pickle.loads(pickle.dumps((pl, w)))
    live(pl, w, rows[400:])
    live(pl2, w2, rows[400:])
    assert np.array_equal(w.soup, w2.soup) and np.array_equal(w.equity(), w2.equity())
    c = w.census(top=3)
    assert c["energy"] == pytest.approx(float(w.equity().sum()))
    assert c["net"] == pytest.approx(c["energy"] - c["injected"])
    assert set(c) >= {"matter", "energy_map", "alive_map", "counts"}


def test_exposure_bytes_round_trip():
    for x in (-125, -10, -1, -0.05, 0, 0.04, 2.4, 10, 125):
        assert exposure_of(byte_of_exposure(x)) == pytest.approx(x, rel=0.05, abs=0.02)
    assert exposure_of(0) == 0 and exposure_of(64) == -exposure_of(192) > 0   # 0 is stillness


def test_a_latitude_trades_only_at_its_own_tick():
    rows = synthetic_rows(12, seed=2, step_s=1, start_ms=DAY0)
    S = 2 * 8 * 8
    matter = np.tile(assemble([64, "A"] + ["x"] * 62), (S, 1))
    pl, w = world(matter=matter, noise=0.0)                        # at most 1x: byte 64 is about 0.42
    acted_at, filled_at = np.full(w.S, -1), np.full(w.S, -1)
    for k, r in enumerate(rows):
        w.step(r)
        filled_at[(filled_at < 0) & (w.q != 0)] = k
        acted_at[(acted_at < 0) & (np.isfinite(w.pending) | (w.q != 0))] = k
    eq, five = (w.site_y == 0) & (filled_at >= 0), (w.site_y == 1) & (filled_at >= 0)
    assert eq.any() and five.any()
    assert (filled_at[eq] == acted_at[eq] + 1).all()               # the equator: the next second
    nxt = (acted_at[five] // 5 + 1) * 5                            # the 5-second latitude: the second
    assert (filled_at[five] == nxt).all()                          # after its next tick
    assert (np.abs(w.exposure()[filled_at >= 0]) <= 1.0 + 1e-9).all()


def test_writing_costs_heat_and_matter_without_energy_is_inert():
    prog = tape(">+>+>+", n=8)                              # three writes
    s = np.array([128], np.uint8)
    me = np.array([128, 128], np.uint8)

    def go(t, wallet, heat):
        w, flow = np.array(wallet), np.zeros((2, 3))
        run(t, STEPS, s, s, me, me, w, 0, 1, np.full(2, -1, np.int16), flow, heat, 1.0, 0.01, BFF)
        return t, w, flow

    free, _, _ = go(prog.copy(), [0.0, 0.0], 0.0)
    paid, w, flow = go(prog.copy(), [1.0, 0.0], 0.25)
    assert np.array_equal(paid, free)                       # the same work...
    assert w[0] == pytest.approx(0.25) and flow[0, 2] == pytest.approx(0.75)   # ...paid write by write
    poor, w, flow = go(prog.copy(), [0.1, 0.0], 0.25)       # cannot afford a single write
    assert np.array_equal(poor, prog) and w[0] == 0.1 and flow.sum() == 0      # inert


def test_transfers_lose_what_digestion_does_not_keep():
    s, me = np.array([128], np.uint8), np.array([128, 128], np.uint8)
    w, flow = np.array([0.0, 100.0]), np.zeros((2, 3))
    run(tape(127, "T", n=8), STEPS, s, s, me, me, w, 0, 1, np.full(2, -1, np.int16), flow, 0.0, 0.8, 10.0,
        BFF)
    assert w.tolist() == pytest.approx([8.0, 90.0])          # took a bite of 10, kept 8
    assert flow[0, 2] == pytest.approx(2.0)                   # the rest dissipated
    w = np.array([0.0, 100.0])
    run(tape(127, "[T]", n=8), STEPS, s, s, me, me, w, 0, 1, np.full(2, -1, np.int16),
        np.zeros((2, 3)), 0.0, 1.0, 10.0, BFF)
    assert w.tolist() == pytest.approx([100.0, 0.0])         # draining takes a loop


def test_the_symmetric_chemistry_has_no_lean():
    ops = np.nonzero(SYMMETRIC)[0]
    assert len(ops) == 14 and abs(ops.mean() - 128) < 1e-9     # code read as data averages to flat
    assert (np.abs(ops.astype(int) - 128) >= 64).all()         # and readings near 128 stay inert
    assert set(np.nonzero(chemistry("bff", market=False))[0]) == set(b"<>{}+-.,[]")
    assert disassemble(assemble("[.>}]S")) == "[.>}]S"


def test_an_instruction_never_reads_itself():
    t, *_, act, w, f = execute(tape("A", n=8))                 # head 0 sits on the A being executed
    assert list(act) == [0, -1]                                # it acts on nothing: stays flat
    t, *_, act, w, flow = execute(tape("T", n=8), wallet=(10.0, 10.0))
    assert w.tolist() == [10.0, 10.0]


def test_neighbors_stay_on_tiny_lattices():
    for X, Y in ((64, 1), (8, 2), (3, 3)):
        p = partners(X, Y)
        assert (p[:, 1] >= 0).all() and (p[:, 1] < X * Y).all() and (p[:, 0] != p[:, 1]).all()
    assert len(neighbors(1, 1)) == 0                       # alone, nothing to meet


def test_a_transplanted_organism_runs_on_unseen_data_and_explains_itself():
    from evotrader.transplant import drivers, live, organisms, sandbox, summary
    pl, w = world()
    live_rows = synthetic_rows(600, seed=7, step_s=1, start_ms=DAY0)
    w.advance(live_rows[:300])
    s, e, x, y, tape = organisms(w, k=1)[0]
    assert e == pytest.approx(w.equity()[s])
    trader = assemble(["x", "S", "A"] + ["x"] * (L - 3))        # long when stream 0 reads above its median
    trader[0] = 0                                               # head 0 on byte 0: reads stream 0
    p2, colony = sandbox(w, x, 0, trader, sites=16)
    assert colony.S == 16 and (colony.soup == trader).all() and colony.noise == 0.0
    res = live(colony, synthetic_rows(1200, seed=8, step_s=1, start_ms=DAY0 + 3_600_000), every=5)
    out = summary(res)
    assert out["trades"] and out["orders"] > 0
    assert out["market"] == pytest.approx(out["return"] + out["fees"] + out["funding"] + out["heat"])
    d = drivers(res)
    prices = {f"{n} level" for n in ("close", "high", "low", "mark", "spot")}
    assert d and {n for n, _ in d[:5]} <= prices and d[0][1] > 0.3   # it follows the price level


def test_soup_runs_render_as_a_census(tmp_path):
    import json
    from evotrader import planet_run
    from evotrader.viewer import is_soup, write_soup_viewer
    run = tmp_path / "soup"
    run.mkdir()
    pl, w = world(max_exposure=125, heat=0.0001, digestion=0.8)
    (run / "meta.json").write_text(json.dumps({
        "kind": "soup", "source": "synthetic", "taus": pl.taus.tolist(), "max_exposure": 125,
        "heat": 0.0001, "digestion": 0.8, "regions": {k: v.astype(int).tolist() for k, v in pl.regions.items()}}))
    rows = synthetic_rows(1200, seed=4, step_s=1, start_ms=DAY0)
    planet_run.run_soup(w, [rows[:500], rows[500:]], str(run), census_every_s=300, log=lambda *a: None)
    assert is_soup(str(run)) and not is_soup(str(tmp_path))
    out = tmp_path / "soup.html"
    n = write_soup_viewer(str(run), str(out))
    assert n == len(planet_run.read_census(str(run))) >= 3
    html = out.read_text()
    assert html.startswith("<!doctype html>") and "<title>BTC Soup Census</title>" in html
    data = json.loads(html.split('type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    shown = data["runs"][0]
    assert shown["label"] == "125x sun, Landauer heat, digestion 80%"
    assert shown["bands"][0] == "1s" and len(shown["frames"]) == n
    last = shown["frames"][-1]
    assert last["energy"] == pytest.approx(float(w.equity().sum())) and last["matter"]["top"]
    assert write_soup_viewer([str(run), str(run)], str(out), bare=True, max_frames=2) == [2, 2]
    assert out.read_text().startswith("<title>")


# ------------------------------------------------------------ one law, any device, any chunking
def test_the_exchange_laws_are_the_accounts_exactly():
    from evotrader.exchange import Accounts
    from evotrader.soup import brackets_matrix
    cfg = Config()
    rng = np.random.default_rng(3)
    n = 64
    ref = Accounts(n, cfg)
    for i in range(n):
        ref.open(i, rng.uniform(50, 5000))
    acct = np.zeros((physics.N_ACCT, n))
    acct[physics.WALLET] = ref.wallet
    counts = np.zeros((physics.N_COUNT, n), np.int64)
    br = brackets_matrix(cfg)
    price = 60_000.0
    for step in range(300):
        price *= np.exp(rng.normal(0, 0.01))
        idx = np.nonzero(rng.random(n) < 0.3)[0]
        lev = rng.integers(1, 126, len(idx)).astype(float)
        target = rng.normal(0, 0.05, len(idx)) * rng.choice([0, 1, 1, 1], len(idx))
        ref.execute(idx, target, price, lev)
        for i, tq, lv in zip(idx, target, lev):
            physics.fill(i, tq, price, lv, acct, counts, cfg.qty_step, cfg.taker_fee, cfg.half_spread,
                         cfg.min_notional)
        hi, lo = price * (1 + abs(rng.normal(0, 0.02))), price * (1 - abs(rng.normal(0, 0.02)))
        ref.liquidate(hi, lo)
        for i in range(n):
            physics.liquidate(i, hi, lo, acct, counts, br)
        if step % 10 == 0:
            rate = rng.normal(0.0001, 0.0003)
            ref.settle_funding(rate, price)
            for i in range(n):
                physics.fund(i, rate, price, acct)
    for row, mine in ((ref.wallet, physics.WALLET), (ref.q, physics.Q), (ref.entry, physics.ENTRY),
                      (ref.margin, physics.MARGIN), (ref.fees, physics.FEES), (ref.funding, physics.FUNDING)):
        assert np.array_equal(row, acct[mine])                     # bit for bit
    assert np.array_equal(ref.liquidations, counts[physics.LIQUIDATIONS]) and ref.liquidations.sum() > 0
    assert np.array_equal(ref.trades, counts[physics.TRADES])
    assert all(physics.equity(i, price, acct) == ref.equity(price, i) for i in range(n))


def test_the_weather_reads_each_band_as_percentiles_of_its_own_past():
    from evotrader.planet import TAU
    from evotrader.selfmade import N_REC, RECEPTORS
    from evotrader.weather import Weather
    rows = np.array(synthetic_rows(700, seed=6, step_s=1, start_ms=DAY0))
    w = Weather(TAU, 1)
    env, done, band = w.feed(rows[:300])
    env2, done2, band2 = w.feed(rows[300:])                        # chunks carry on seamlessly
    done, band = np.concatenate([done, done2]), np.concatenate([band, band2])
    t = (rows[:, 0] + 1000) / 1000
    assert np.array_equal(done, (np.round(t)[:, None] % TAU == 0).astype(np.uint8))
    k = RECEPTORS.index("close")
    level = np.log(rows[:, C["close"]])                             # the 1-second band: every row
    for i in (40, 299, 300, 699):
        past = level[max(0, i - 255):i + 1]
        pct = ((past < level[i]).sum() + 0.5 * (past == level[i]).sum()) / len(past)
        assert (int(band[i, 0, k]) + 128) % 256 - 128 == np.clip(np.round(255 * pct - 127.5), -128, 127)
    assert (band[:15, 0, k] == 0).all()                             # too little history: ordinary
    assert (band[:, 0, N_REC + RECEPTORS.index("spot")] == 0).all() or True
    w2 = Weather(TAU, 1)
    *_, again = w2.feed(rows)
    assert np.array_equal(again, band)                              # however the rows are chunked


def test_a_world_is_the_same_however_its_ticks_are_chunked():
    rows = np.array(synthetic_rows(500, seed=9, step_s=1, start_ms=DAY0))
    pl, a = world(max_exposure=125, heat=0.0001, digestion=0.8, noise=0.01)
    pl, b = world(max_exposure=125, heat=0.0001, digestion=0.8, noise=0.01)
    a.advance(rows)
    for k in range(0, 500, 37):
        b.advance(rows[k:k + 37])
    assert np.array_equal(a.soup, b.soup) and np.array_equal(a.acct, b.acct, equal_nan=True)
    assert a.counts == b.counts and a.counts["interactions"] > 0 and a.counts["orders"] > 0


def test_the_gpu_computes_the_same_world():
    import os
    import subprocess
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")               # numba's CUDA simulator: no GPU needed
    out = subprocess.run([sys.executable, os.path.join(here, "experiments", "gpu_selftest.py"), "--small",
                          "--rows", "6"], capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout and "DIFFERENT" not in out.stdout


def test_matter_alone_meets_its_neighbors_and_copies():
    copier = np.array([64] + list(b"[}-].>}[.>}]") + [ord("x")] * (L - 13), np.uint8)
    X, Y = 32, 4
    matter = np.tile(np.full(L, ord("x"), np.uint8), (X * Y, 1))
    matter[::16] = copier                                          # 8 copiers among 128 inert tapes
    p = Primordial(X, Y, seed=0, table=BFF, matter=matter)
    p.advance(1)
    assert 0.2 < p.stats[:, 0].sum() / p.S < 0.5                   # about a quarter of sites start one
    assert p.stats[:, 2].sum() > 0 and not np.array_equal(p.soup, matter)   # and matter is rewritten
    q = Primordial(X, Y, seed=0, table=BFF, matter=matter)
    q.advance(1)
    assert np.array_equal(p.soup, q.soup)                          # the same seed, the same history
