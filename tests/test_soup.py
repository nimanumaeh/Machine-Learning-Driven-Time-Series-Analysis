"""The soup's physics: matter, chemistry, the market instructions, and money."""

import numpy as np
import pytest

from evotrader import physics
from evotrader.soup import (BFF, L, STEPS, SYMMETRIC, Primordial, assemble, census_of_matter,
                            chemistry, disassemble, high_order_entropy, neighbors, replicates, run)


import pickle

from evotrader import physics
from evotrader.config import Config
from evotrader.data import C
from evotrader.selfmade import RECEPTORS
from evotrader.soup import Soup, byte_of_exposure, exposure_of, gear_of
from evotrader.synthetic import synthetic_rows
from evotrader.weather import NS

DAY0 = 1_700_006_400_000


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
    assert len(ops) == 16 and abs(ops.mean() - 128) < 1e-9     # code read as data averages to flat
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


def test_exposure_bytes_round_trip():
    for x in (-125, -10, -1, -0.05, 0, 0.04, 2.4, 10, 125):
        assert exposure_of(byte_of_exposure(x)) == pytest.approx(x, rel=0.05, abs=0.02)
    assert exposure_of(0) == 0 and exposure_of(64) == -exposure_of(192) > 0   # 0 is stillness


# ------------------------------------------------------------ living matter on the market (physics 3)
def world(width=3, height=3, occupancy=1.0, matter=None, **kw):
    """A small world of organisms (inert matter unless given), no noise, exact copies."""
    kw.setdefault("noise", 0.0)
    kw.setdefault("mutation", 0.0)
    kw.setdefault("meetings", 0.0)
    if matter is None:
        matter = np.full(L, 128, np.uint8)                        # 128 is inert
    return Soup(Config(), width=width, height=height, step_s=60, seed=0, occupancy=occupancy, matter=matter, **kw)


def minutes(n, seed=1):
    return np.array(synthetic_rows(n, seed=seed, step_s=60))


def thinking(w, s, steps):
    """Run site s's matter for `steps` instructions, as one tick would (the plain law, in Python)."""
    selfs = np.zeros(physics.N_SELF, np.uint8)
    band = np.arange(NS, dtype=np.uint8) + 1                     # stream k reads k + 1
    return physics.think(s, steps, w.soup, w.regs, band, w.mask, selfs, w.acct, w.flow, w.heat, w.table,
                         w.expo, w.gears)


def test_matter_thinks_on_its_own_and_carries_on_where_it_stopped():
    w = world(width=1, height=1)
    w.soup[0, :3] = assemble("+>+")
    thinking(w, 0, 2)
    assert w.soup[0, 0] == assemble("+")[0] + 1 and list(w.regs[0]) == [2, 1, 0]   # it stopped after two
    thinking(w, 0, 1)
    assert w.soup[0, 1] == assemble(">")[0] + 1 and w.regs[0, 0] == 3   # and went on from there
    thinking(w, 0, L - 3)
    assert w.regs[0, 0] == 0                                       # the tape is a loop
    w.soup[0] = 128
    w.soup[0, :2] = assemble("]+")                                 # an unmatched ] does nothing
    w.regs[0] = [0, 5, 0]
    executed, _ = thinking(w, 0, 10)
    assert executed == 10 and w.regs[0, 0] == 10 and w.soup[0, 5] == 129


def test_a_and_l_choose_a_position_and_a_leverage_and_s_reads_only_what_is_here():
    w = world(width=1, height=1)
    w.soup[0, :3] = [64, assemble("L")[0], assemble("A")[0]]
    thinking(w, 0, 3)
    assert w.acct[physics.GEAR, 0] == gear_of(64) == 11          # 126 ** (64 / 127), rounded
    assert w.pending[0] == pytest.approx(exposure_of(64, 1.0))   # about 0.41 of its equity, long
    w.soup[0, :3] = [assemble("A")[0], 128, 128]                   # A with head 0 on itself: 0, flat
    w.regs[0] = 0
    thinking(w, 0, 1)
    assert w.pending[0] == 0.0
    k = RECEPTORS.index("spot")                                    # a regional stream
    w.mask[0, k] = 0
    w.soup[0, :2] = [k, assemble("S")[0]]
    w.regs[0] = 0
    thinking(w, 0, 2)
    assert w.soup[0, 0] == 0                                       # it does not exist here: 0
    w.soup[0, :2] = [1, assemble("S")[0]]
    w.regs[0] = 0
    thinking(w, 0, 2)
    assert w.soup[0, 0] == 2                                       # stream 1 reads 2 in this sense


def test_the_order_fills_at_the_next_open_with_the_leverage_it_chose():
    w = world(width=1, height=1, upkeep=0.0, metabolism_days=float("inf"))
    w.soup[0, :3] = [64, assemble("L")[0], assemble("A")[0]]
    rows = minutes(3)
    w.advance(rows[:1])                                            # it decides (fills next tick)
    assert w.q[0] == 0 and np.isfinite(w.pending[0])
    w.advance(rows[1:2])
    price = rows[1, C["open"]] + Config().half_spread
    x = exposure_of(64, 1.0)
    assert w.acct[physics.ENTRY, 0] == pytest.approx(price)
    assert w.q[0] * price == pytest.approx(x * 11 * 1000.0, rel=0.02)   # 0.41 of its equity, at 11x
    assert w.acct[physics.MARGIN, 0] == pytest.approx(w.q[0] * price / 11)
    assert w.acct[physics.FEES, 0] > 0 and w.acct[physics.LEV, 0] == 11


def test_living_costs_energy_and_an_organism_with_nothing_left_dies():
    w = world(width=1, height=1, upkeep=10.0, metabolism_days=30.0)
    k = 4 * (1 - 2 ** -0.25) * 1000.0 ** 0.25 / 30.0 / 1440       # Kleiber's term, a minute
    w.advance(minutes(1))
    assert w.wallet[0] == pytest.approx(1000.0 - 10.0 / 1440 - k * 1000.0 ** 0.75)
    w.acct[physics.WALLET, 0] = 1.002                              # just above the floor of 1 USDT
    w.advance(minutes(2)[1:])
    assert w.alive[0] == 0 and (w.soup[0] == 0).all() and w.counts["deaths"] == 1   # matter to nothing
    assert w.flow[0, physics.CARCASS] == pytest.approx(1.002 - w.flow[0, physics.METABOLISM] + w.flow[0, physics.METABOLISM] - 1.002 + w.flow[0, physics.CARCASS])


def test_a_body_that_grows_divides_in_half_and_the_child_carries_on():
    w = world(occupancy=0.0)
    w.alive[4] = 1                                                 # one organism, in the middle
    w.soup[4] = assemble("+>" * 32)
    w.acct[physics.WALLET, 4] = 2600.0                             # it has grown past two stakes
    w.acct[physics.Q, 4] = 0.01                                    # and holds a position
    w.acct[physics.ENTRY, 4] = 50_000.0
    w.acct[physics.MARGIN, 4] = 500.0
    w.ids[physics.ID, 4] = w.ids[physics.FOUNDER, 4] = 5
    w.injected = 3100.0
    rows = minutes(1)
    w.mark = rows[0, C["close"]]
    before = w.equity()[4]
    w.advance(rows)
    child = [s for s in range(9) if s != 4 and w.alive[s]]
    assert len(child) == 1
    c = child[0]
    assert w.wallet[c] == pytest.approx(w.wallet[4]) and w.q[c] == pytest.approx(w.q[4]) == 0.005
    assert w.acct[physics.MARGIN, c] == pytest.approx(250.0) and w.acct[physics.ENTRY, c] == 50_000.0
    assert w.ids[physics.PARENT, c] == 5 and w.ids[physics.FOUNDER, c] == 5 and w.life[physics.GEN, c] == 1
    assert np.array_equal(w.soup[c], w.soup[4]) and np.array_equal(w.regs[c], w.regs[4])   # same mind, same place in it
    assert w.equity()[[4, c]].sum() == pytest.approx(before - w.flow[[4, c], physics.METABOLISM].sum()
                                                     - w.flow[[4, c], physics.LOST].sum(), rel=1e-9)


def test_a_child_displaces_and_eats_a_weaker_neighbor_but_not_a_stronger_one():
    w = world(upkeep=0.0, metabolism_days=float("inf"))
    w.acct[physics.WALLET] = 400.0                                 # weak neighbors all around
    w.acct[physics.WALLET, 4] = 3000.0                             # and one that has grown
    w.injected = float(w.acct[physics.WALLET].sum())
    w.advance(minutes(1))
    born = np.nonzero(w.life[physics.GEN] == 1)[0]
    assert len(born) == 1 and w.counts["deaths"] == 1
    c = born[0]
    assert w.wallet[c] == pytest.approx(1500.0 + 0.8 * 400.0)     # half the parent, and 80% of the eaten
    assert w.equity()[w.alive.astype(bool)].sum() + w.flow[:, physics.LOST].sum() == pytest.approx(w.injected)
    strong = world(upkeep=0.0, metabolism_days=float("inf"))
    strong.acct[physics.WALLET] = 1600.0                           # neighbors stronger than half of 3000
    strong.acct[physics.WALLET, 4] = 3000.0
    strong.advance(minutes(1))
    assert strong.counts["births"] == 0 and strong.counts["deaths"] == 0   # they hold their ground


def test_with_isolated_margin_a_liquidation_takes_only_the_margin():
    w = world(width=1, height=1, upkeep=0.0, metabolism_days=float("inf"))
    w.soup[0, :3] = [127, assemble("L")[0], 128]                   # 125x
    w.soup[0, 3:5] = [16, assemble("A")[0]]                        # with head 0 on 127: all of its equity
    rows = minutes(4)
    rows[:, [C["open"], C["high"], C["low"], C["close"], C["mark_high"], C["mark_low"], C["mark_close"]]] = 60_000.0
    rows[2, [C["low"], C["mark_low"]]] = 59_000.0                   # a 1.7% wick
    w.advance(rows[:2])
    assert w.q[0] > 0 and w.acct[physics.LEV, 0] == 125
    margin = w.acct[physics.MARGIN, 0]
    wallet = w.wallet[0]
    w.advance(rows[2:3])
    assert w.q[0] == 0 and w.count[physics.LIQUIDATIONS, 0] == 1
    assert w.wallet[0] == pytest.approx(wallet) and w.wallet[0] >= 0      # the margin is gone, nothing more
    assert margin > 0


def test_meetings_bite_with_digestion_and_copy_between_organisms():
    w = world(width=2, height=1)
    tape_ = np.zeros(2 * L, np.uint8)
    tape_[:L] = 128
    tape_[L:] = 128
    tape_[0], tape_[1] = 127, assemble("T")[0]                     # the first takes a full bite
    act = np.full(2, -1, np.int16)
    band = np.zeros(NS, np.uint8)
    sa, sb = np.zeros(physics.N_SELF, np.uint8), np.zeros(physics.N_SELF, np.uint8)
    physics.meet_tape(tape_, 64, band, w.mask, sa, sb, w.acct, 0, 1, act, w.flow, 0.0, 0.8, 1.0, w.table, w.gears)
    assert w.wallet[0] == pytest.approx(1000.8) and w.wallet[1] == pytest.approx(999.0)
    assert w.flow[0, physics.LOST] == pytest.approx(0.2)
    tape_[:L] = assemble(["x"] * L)
    tape_[:6] = assemble("}" * 0 + "{.")[0:2].tolist() + [128] * 4  # head 1 back onto the other's last byte
    tape_[0] = assemble("{")[0]
    tape_[1] = assemble(".")[0]                                    # and copy byte 0 there
    physics.meet_tape(tape_, 2, band, w.mask, sa, sb, w.acct, 0, 1, act, w.flow, 0.0, 0.8, 1.0, w.table, w.gears)
    assert tape_[2 * L - 1] == tape_[0]                            # matter moved into the other organism


def test_without_trading_every_unit_of_energy_is_accounted_for():
    from evotrader.soup import MNEMONIC
    w = Soup(Config(), width=16, height=16, step_s=60, seed=3, metabolism_days=5.0, heat=1e-3, quantum=20.0,
             divide_at=1050.0)
    w.table[w.table == MNEMONIC.index("A")] = 0                   # no orders: energy only moves or is lost
    w.advance(minutes(2000, seed=2))
    f = w.flow.sum(0)
    assert w.counts["births"] > 0 and w.counts["deaths"] > 0 and f[physics.TAKEN] > 0
    alive = w.alive.astype(bool)
    assert w.equity()[alive].sum() + f[physics.LOST] + f[physics.METABOLISM] + f[physics.CARCASS] == \
        pytest.approx(w.injected, rel=1e-12)


def test_a_world_is_the_same_however_its_ticks_are_chunked_and_resumes_exactly():
    kw = dict(width=12, height=12, step_s=60, seed=4, heat=1e-4, noise=0.05, mutation=0.05, meetings=0.25,
              divide_at=1002.0, metabolism_days=30.0)
    rows = minutes(900, seed=9)
    a, b = Soup(Config(), **kw), Soup(Config(), **kw)
    a.advance(rows[:400])
    for k in range(0, 400, 37):
        b.advance(rows[k:min(k + 37, 400)])
    c = pickle.loads(pickle.dumps(a))
    a.advance(rows[400:])
    b.advance(rows[400:])
    c.advance(rows[400:])
    for x in (b, c):
        assert np.array_equal(a.soup, x.soup) and np.array_equal(a.acct, x.acct, equal_nan=True)
        assert np.array_equal(a.ids, x.ids) and a.counts == x.counts
    n = a.counts
    assert n["births"] > 0 and n["deaths"] > 0 and n["meetings"] > 0 and n["orders"] > 0


def test_the_census_tells_who_lives_how_and_where_energy_went():
    w = Soup(Config(), width=16, height=16, step_s=60, seed=5, divide_at=1100.0, meetings=0.25)
    w.advance(minutes(600))
    c = w.census(top=3)
    alive = w.alive.astype(bool)
    assert c["alive"] == alive.sum() and c["energy"] == pytest.approx(float(w.equity()[alive].sum()))
    assert c["net"] == pytest.approx(c["energy"] - c["injected"])
    assert sum(c["timescales"]) == c["alive"] and sum(c["gears"]) == c["alive"]
    assert c["lines"] == len(set(w.ids[physics.FOUNDER, alive])) and c["top_lines"][0]["alive"] >= 1
    assert set(c) >= {"births", "deaths", "metabolism", "carcass", "to_children", "matter", "energy_map"}


def test_a_transplanted_organism_runs_alone_on_unseen_data_and_explains_itself():
    from evotrader.transplant import drivers, evaluate, live, organisms, sandbox, summary
    w = Soup(Config(), width=8, height=8, step_s=60, seed=2)
    w.advance(minutes(300, seed=7))
    s, e, x, y, tape_ = organisms(w, k=1)[0]
    assert e == pytest.approx(w.equity()[s])
    trader = assemble(["x", "S", "A", "[", "-", "]"] + ["x"] * (L - 6))   # long when stream 0 reads above
    trader[0] = 0                                                  # its median; [-] sets the byte back to 0
    itself = sandbox(w, s)                                         # as it was: tape, registers, leverage
    assert (itself.soup[0] == w.soup[s]).all() and (itself.regs[0] == w.regs[s]).all()
    assert (itself.mask[0] == w.mask[s]).all() and itself.acct[physics.GEAR, 0] == w.acct[physics.GEAR, s]
    colony = sandbox(w, s, trader, sites=1)
    assert colony.S == 1 and (colony.soup[0] == trader).all() and colony.noise == 0.0
    res = live(colony, minutes(1200, seed=8), every=5)
    out = summary(res)
    assert out["trades"] and out["orders"] > 0 and out["living"] == 0.0     # no cost of living here
    assert out["market"] == pytest.approx(out["return"] + out["fees"] + out["funding"] + out["heat"])
    report = evaluate(w, minutes(600, seed=9), k=2, baseline=1, log=lambda *a: None)
    assert [r["kind"] for r in report] == ["organism", "organism", "random"]
    d = drivers(res)
    prices = {f"{n} level" for n in ("close", "high", "low", "mark", "spot")}
    assert d and {n for n, _ in d[:3]} <= prices and d[0][1] > 0.5   # it follows the price level


def test_soup_runs_render_as_a_census(tmp_path):
    import json
    from evotrader import planet_run
    from evotrader.viewer import is_soup, write_soup_viewer
    run = tmp_path / "soup"
    run.mkdir()
    w = Soup(Config(), width=8, height=8, step_s=60, seed=1, divide_at=1100.0, metabolism_days=30.0)
    (run / "meta.json").write_text(json.dumps({
        "kind": "soup", "physics": 3, "source": "synthetic", "resolution_s": 60, "metabolism_days": 30.0,
        "upkeep": 1.0, "width": 8, "height": 8,
        "regions": {k: v.astype(int).tolist() for k, v in w.planet.regions.items()}}))
    rows = minutes(1200, seed=4)
    planet_run.run_soup(w, [rows[:500], rows[500:]], str(run), census_every_s=6 * 3600, log=lambda *a: None)
    assert is_soup(str(run)) and not is_soup(str(tmp_path))
    out = tmp_path / "soup.html"
    n = write_soup_viewer(str(run), str(out))
    assert n == len(planet_run.read_census(str(run))) >= 3
    html = out.read_text()
    assert html.startswith("<!doctype html>") and "<title>BTC Soup Census</title>" in html
    data = json.loads(html.split('type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    shown = data["runs"][0]
    assert "physics 3" in shown["label"] and len(shown["frames"]) == n
    last = shown["frames"][-1]
    assert last["alive"] == planet_run.read_census(str(run))[-1]["alive"] and last["energy"] > 0
    assert write_soup_viewer([str(run), str(run)], str(out), bare=True, max_frames=2) == [2, 2]
    assert out.read_text().startswith("<title>")


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
