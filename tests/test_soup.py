"""The soup's physics: matter, chemistry, the market instructions, and money."""

import numpy as np
import pytest

from evotrader.soup import (L, STEPS, census_of_matter, high_order_entropy, interact, neighbors,
                            pair_up, replicates, run)


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
               w, 0, 1, act, flow)
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
    t, *_, w, flow = execute(tape(255, "T", n=8), wallet=(100.0, 50.0))   # take everything free
    assert w.tolist() == [150.0, 0.0] and flow[0, 0] == 50.0 and flow[1, 1] == 50.0
    t, *_, w, flow = execute(tape(64, "T", n=8), wallet=(100.0, 50.0))    # give half of its own
    assert w.tolist() == [50.0, 100.0]
    rng = np.random.default_rng(1)
    soup = rng.integers(0, 256, (64, L), dtype=np.uint8)
    soup[:, :8] = np.frombuffer(b"T>T<T>T<", np.uint8)                  # lots of grabbing and giving
    wallet = rng.uniform(0, 1000, 64)
    total = wallet.sum()
    flow = np.zeros((64, 3))
    pairs = pair_up(rng, 8, 8, rng.permutation(64), neighbors(8, 8))
    interact(soup, pairs, STEPS, np.full((64, 4), 128, np.uint8), np.full((64, 2), 128, np.uint8),
             wallet, np.full(64, -1, np.int16), flow, np.zeros((64, 2), np.int64))
    assert wallet.sum() == pytest.approx(total) and (wallet >= -1e-9).all()
    assert flow[:, 0].sum() == pytest.approx(flow[:, 1].sum()) and flow.sum() > 0


def test_a_copier_copies_itself_into_its_neighbor():
    # count head 1 out to the neighbor's half, then copy byte after byte until a zero
    prog = [64] + list(b"[}-].>}[.>}]") + [ord("x")] * (L - 13)
    assert replicates(np.array(prog, np.uint8)) > 0.95
    assert replicates(np.frombuffer(b"x" * L, np.uint8)) < 0.05   # inert matter copies nothing


def test_neighbors_stay_on_the_planet():
    rng = np.random.default_rng(0)
    X, Y = 16, 8
    sites = np.arange(X * Y)
    for _ in range(20):
        p = pair_up(rng, X, Y, sites, neighbors(X, Y))
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


def world(matter=None, width=2, per_place=8, **kw):
    pl = Planet(width=width, seed=0)
    return pl, Soup(Config(), pl, per_place=per_place, seed=0, interactions=64, matter=matter, **kw)


def live(pl, w, rows):
    for r in rows:
        w.step(r, pl.step(r))


def test_without_orders_energy_only_moves_between_sites():
    S = 2 * 8 * 8
    grab_give = np.array([200, ord("T"), ord(">"), ord("T"), ord(">"), 40, ord("<"), ord("T")], np.uint8)
    matter = np.tile(np.resize(grab_give, L), (S, 1))              # grabbing and giving, nothing else
    pl, w = world(matter=matter, noise=0.0)
    live(pl, w, synthetic_rows(300, seed=1, step_s=1, start_ms=DAY0))
    assert w.counts["orders"] == 0 and w.flow[:, 0].sum() > 0
    assert w.equity().sum() == pytest.approx(w.injected)          # nothing made, nothing lost
    assert w.equity().std() > 1                                    # but it moved around


def test_orders_fill_at_the_next_open_exactly():
    rows = synthetic_rows(200, seed=2, step_s=1, start_ms=DAY0)
    S = 2 * 8 * 8
    matter = np.tile(np.resize(np.frombuffer(bytes([192]) + b"A" + b"x" * 62, np.uint8), L), (S, 1))
    pl, w = world(matter=matter, noise=0.0, max_exposure=125)
    r0 = rows[0]
    w.step(r0, pl.step(r0))                                        # matter decides: about 10x long
    assert np.isfinite(w.pending).any() and not w.acct.q.any()
    i = int(np.nonzero(np.isfinite(w.pending) & (w.site_y == 0))[0][0])   # an equator site
    x = w.pending[i]
    assert x == pytest.approx(exposure_of(192))
    r1 = rows[1]
    w.step(r1, pl.step(r1))
    fill = r1[C["open"]] + Config().half_spread
    assert w.acct.entry[i] == pytest.approx(fill)                  # at this second's open, plus spread
    assert w.acct.q[i] * fill == pytest.approx(x * Config().initial_capital, rel=0.02)
    assert w.acct.fees[i] > 0 and w.lev[i] == np.ceil(x)


def test_time_is_allotted_by_energy_to_the_three_quarters():
    pl, w = world(noise=0.0)
    rich = np.arange(w.S) % 2 == 0
    w.acct.wallet[rich] = 16 * w.stake                             # 16x the energy...
    w.mark = 60000.0
    counts = np.zeros(w.S)
    w.soup[:] = ord("x")
    for _ in range(200):
        eq = np.maximum(w.equity(), 0.0) ** w.kleiber
        counts += np.bincount(w.rng.choice(w.S, size=w.E, p=eq / eq.sum()), minlength=w.S)
    ratio = counts[rich].mean() / counts[~rich].mean()
    assert ratio == pytest.approx(16 ** 0.75, rel=0.1)             # ...buys 8x the time, not 16x


def test_senses_exist_only_where_the_stream_does():
    pl, w = world(width=4, per_place=4)
    live(pl, w, synthetic_rows(400, seed=3, step_s=1, start_ms=DAY0))
    absent = ~w.mask
    assert absent.any() and (w.senses[absent] == 128).all()
    present = w.mask & (w.site_y[:, None] == 0)
    assert (w.senses[present] != 128).mean() > 0.5


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


def test_a_latitude_trades_only_at_its_own_tick():
    rows = synthetic_rows(12, seed=2, step_s=1, start_ms=DAY0)
    S = 2 * 8 * 8
    matter = np.tile(np.resize(np.frombuffer(bytes([192]) + b"A" + b"x" * 62, np.uint8), L), (S, 1))
    pl, w = world(matter=matter, noise=0.0)                        # at most 1x: byte 192 is about 0.42
    acted_at, filled_at = np.full(w.S, -1), np.full(w.S, -1)
    for k, r in enumerate(rows):
        w.step(r, pl.step(r))
        filled_at[(filled_at < 0) & (w.acct.q != 0)] = k
        acted_at[(acted_at < 0) & (np.isfinite(w.pending) | (w.acct.q != 0))] = k
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
        run(t, STEPS, s, s, me, me, w, 0, 1, np.full(2, -1, np.int16), flow, heat)
        return t, w, flow

    free, _, _ = go(prog.copy(), [0.0, 0.0], 0.0)
    paid, w, flow = go(prog.copy(), [1.0, 0.0], 0.25)
    assert np.array_equal(paid, free)                       # the same work...
    assert w[0] == pytest.approx(0.25) and flow[0, 2] == pytest.approx(0.75)   # ...paid write by write
    poor, w, flow = go(prog.copy(), [0.1, 0.0], 0.25)       # cannot afford a single write
    assert np.array_equal(poor, prog) and w[0] == 0.1 and flow.sum() == 0      # inert
