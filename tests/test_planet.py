"""The 1-second data layer, the planet, and the organisms that live on it."""

import io
import pickle
import zipfile

import numpy as np
import pytest

from evotrader import data
from evotrader import data_seconds as ds
from evotrader import life as life_mod
from evotrader.config import Config
from evotrader.data import COLUMNS, C, MINUTE
from evotrader.exchange import Accounts
from evotrader.life import ACTIONS, R, YOUTH, Life, _harvest
from evotrader.planet import REGIONS, TAU, Planet, evaluate
from evotrader.selfmade import RECEPTORS, ZERO
from evotrader.synthetic import synthetic_rows

DAY0 = 1_700_006_400_000                                   # 2023-11-15 00:00 UTC


def seconds(n, seed=0):
    return synthetic_rows(n, seed=seed, step_s=1, start_ms=DAY0)


def new_life(width=6, n=48, seed=0, capacity=256, **planet_kw):
    pl = Planet(width=width, seed=seed, **planet_kw)
    return pl, Life(Config(), pl, capacity=capacity, seed=seed, min_population=n)


def live(pl, life, rows):
    for r in rows:
        life.step(r, pl.step(r))


def zipped(name, text):
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as zf:
        zf.writestr(name, text)
    return blob.getvalue()


# ------------------------------------------------------------ 1-second data
def test_every_second_is_rebuilt_from_the_trades():
    t = DAY0 + np.array([1500, 1700, 1900, 4200, -10, 86_400_000], np.int64)
    price = np.array([100.0, 102.0, 99.0, 101.0, 50.0, 50.0])
    qty = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 9.0])
    buyer_maker = np.array([False, True, False, False, False, False])
    rows = ds.seconds_from_trades(t, price, qty, buyer_maker, DAY0, prev_close=98.0)
    assert rows.shape == (86_400, len(COLUMNS))
    assert rows[1, 0] - rows[0, 0] == 1000
    assert rows[0, C["open"]] == rows[0, C["close"]] == 98.0 and rows[0, C["volume"]] == 0
    r = rows[1]
    assert (r[C["open"]], r[C["high"]], r[C["low"]], r[C["close"]]) == (100.0, 102.0, 99.0, 99.0)
    assert (r[C["volume"]], r[C["trades"]]) == (6.0, 3)
    assert r[C["quote_volume"]] == pytest.approx(100 + 204 + 297)
    assert r[C["taker_buy_volume"]] == 4.0                 # the buyer took in the 1st and 3rd
    assert (r[C["mark_high"]], r[C["mark_low"]]) == (102.0, 99.0)
    assert rows[3, C["open"]] == rows[3, C["close"]] == 99.0 and rows[3, C["volume"]] == 0
    assert rows[4, C["close"]] == rows[-1, C["close"]] == 101.0
    assert rows[:, C["volume"]].sum() == 10.0              # other days' trades left out


def test_slow_streams_come_from_the_last_closed_minute():
    rows = ds.seconds_from_trades(np.array([DAY0 + 500], np.int64), np.array([100.0]),
                                  np.array([1.0]), np.array([False]), DAY0)
    minutes = np.full((4, len(COLUMNS)), np.nan)
    minutes[:, 0] = DAY0 + (np.arange(4) - 1) * MINUTE      # the minute before the day, then 0..2
    minutes[:, C["premium"]] = (10.0, 20.0, 30.0, 40.0)
    minutes[:, C["funding_rate"]] = (0.0, 0.0, 0.0001, 0.0)  # settles as minute 1 ends
    out = ds.join_slow(rows, minutes)
    assert out[0, C["premium"]] == out[59, C["premium"]] == 10.0
    assert out[60, C["premium"]] == out[119, C["premium"]] == 20.0
    assert out[120, C["premium"]] == 30.0
    assert out[119, C["funding_rate"]] == 0.0001           # the second that ends on the settlement
    assert np.count_nonzero(out[:, C["funding_rate"]]) == 1


def test_aggtrades_archive_with_header_and_microseconds():
    csv = ("agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
           f"2,101.5,0.2,5,5,{(DAY0 + 2000) * 1000},true\n"
           f"1,100.0,0.1,3,4,{(DAY0 + 1000) * 1000},false\n")
    t, price, qty, maker = ds.parse_aggtrades(zipped("BTCUSDT-aggTrades-2023-11-15.csv", csv))
    assert list(t) == [DAY0 + 1000, DAY0 + 2000]
    assert list(price) == [100.0, 101.5] and list(qty) == [0.1, 0.2]
    assert list(maker) == [False, True]


def test_seconds_store_roundtrip_across_days(tmp_path):
    rows = np.zeros((20, len(COLUMNS)))
    rows[:, 0] = DAY0 - 5000 + np.arange(20) * 1000
    rows[:, C["close"]] = np.arange(20)
    ds.save_seconds(str(tmp_path), "BTCUSDT", rows)
    assert len(list((tmp_path / "BTCUSDT").glob("*.npz"))) == 2
    assert np.array_equal(np.array(list(ds.iter_seconds(str(tmp_path), "BTCUSDT"))), rows)
    part = np.array(list(ds.iter_seconds(str(tmp_path), "BTCUSDT", DAY0, DAY0 + 3000)))
    assert list(part[:, C["close"]]) == [5, 6, 7]


def test_build_seconds_from_archives(tmp_path, monkeypatch):
    csv = f"1,100.0,0.5,1,1,{DAY0 + 30_000},false\n2,101.0,0.25,2,2,{DAY0 + 90_500},true\n"
    blob = zipped("BTCUSDT-aggTrades-2023-11-15.csv", csv)
    monkeypatch.setattr(data, "_get", lambda url, cache_dir=None: blob if "2023-11-15" in url else None)
    minutes = synthetic_rows(5, start_ms=DAY0 - 2 * MINUTE)
    data.save_store(str(tmp_path / "market"), "BTCUSDT", minutes)
    n = ds.build_seconds(str(tmp_path / "sec"), str(tmp_path / "market"), "BTCUSDT",
                         start="2023-11-15", end="2023-11-17", cache_dir=None,
                         log=lambda *a, **k: None)
    assert n == 86_400
    rows = np.array(list(ds.iter_seconds(str(tmp_path / "sec"), "BTCUSDT")))
    assert rows[0, C["close"]] == minutes[1, C["close"]]   # the last price before the day
    assert rows[30, C["close"]] == rows[89, C["close"]] == 100.0
    assert rows[90, C["close"]] == 101.0
    assert rows[0, C["premium"]] == minutes[1, C["premium"]]
    assert rows[60, C["premium"]] == minutes[2, C["premium"]]


# ------------------------------------------------------------------ planet
def test_bands_tick_on_their_own_clocks_and_aggregate_bars():
    rows = seconds(3600)
    pl = Planet(width=4, seed=0)
    for r in rows:
        pl.step(r)
    assert list(pl.ticks) == [3600 // T for T in TAU]
    y = list(TAU).index(30)
    first = rows[:30]
    close, low, high, fund = pl.pbuf[y, 0]
    assert close == first[-1, C["close"]] and fund == 0.0
    assert (low, high) == (first[:, C["low"]].min(), first[:, C["high"]].max())
    rec = pl.rbuf[y, 0]
    assert rec[RECEPTORS.index("close")] == pytest.approx(np.log(close))
    assert rec[RECEPTORS.index("volume")] == pytest.approx(np.log1p(first[:, C["volume"]].sum()))
    assert rec[RECEPTORS.index("taker_buy")] == pytest.approx(
        np.log1p(first[:, C["taker_buy_volume"]].sum()))
    assert np.all((pl.water >= 0.5) & (pl.water <= 2.0))


def test_bands_never_see_the_future():
    a, b = seconds(400, seed=1), seconds(400, seed=1)
    b[200:, 1:9] *= 1.1
    pa, pb = Planet(width=4, seed=0), Planet(width=4, seed=0)
    for ra, rb in zip(a, b):
        pa.step(ra)
        pb.step(rb)
    for y, T in enumerate(TAU):
        done = 200 // T                                    # bars closed before the futures split
        assert np.array_equal(pa.rbuf[y, :done], pb.rbuf[y, :done], equal_nan=True)
        assert np.array_equal(pa.pbuf[y, :done], pb.pbuf[y, :done], equal_nan=True)
        if pa.ticks[y] > done:
            assert not np.array_equal(pa.pbuf[y, done], pb.pbuf[y, done])


def test_senses_exist_only_on_their_continents():
    pl = Planet(width=32, seed=3)
    for name, receptors in REGIONS.items():
        here = pl.regions[name]
        assert 0.3 < here.mean() < 0.8
        for r in receptors:
            assert np.array_equal(pl.avail[:, :, RECEPTORS.index(r)], here)
    everywhere = [RECEPTORS.index(n) for n in ("close", "high", "volume", "trades", "day_sin")]
    assert pl.avail[:, :, everywhere].all()


def test_observatories_compute_their_program_and_wear_away():
    pl = Planet(width=4, seed=0, obs_decay=0.01)
    y, prog = 1, np.array([RECEPTORS.index("close"), ZERO, 1, 0, 0])   # z64(change1(close))
    assert pl.build(2, y, prog, builder=7) and pl.build(2, y, prog, builder=8)
    assert not pl.build(2, y, prog, builder=9)             # nothing worn enough to build over
    state = np.zeros((1, 4))
    for r in seconds(600):
        if y in pl.step(r):
            v, state = evaluate(prog[None], pl.rbuf[y], pl.ticks[y], state)
            if pl.ticks[y] < 100:
                assert np.array_equal(pl.obs_value[2, y, :1], v, equal_nan=True)
            else:
                assert np.isnan(pl.obs_value[2, y]).all() and (pl.obs_integrity[2, y] == 0).all()
    assert np.isfinite(v).all()


def test_capacity_follows_each_bands_water():
    pl = Planet(width=5, seed=0, base_capacity=10)
    pl.water[:] = np.linspace(0.5, 2.0, pl.Y)
    cap = pl.capacity()
    assert cap.shape == (5, pl.Y) and (cap == cap[0]).all()
    assert list(cap[0]) == list(np.round(10 * pl.water).astype(int))


# --------------------------------------------------------------- organisms
def test_harvests_are_exactly_what_the_account_would_have_done():
    c = Config()
    c.taker_fee = c.half_spread = c.min_notional = 0.0
    c.qty_step = 1e-12
    p0, lev, xmax = 60000.0, 10.0, 8.0
    bars = [(60300.0, 59900.0, 60400.0, 0.0), (60100.0, 60000.0, 60500.0, 0.0001),
            (60650.0, 60050.0, 60700.0, 0.0)]
    close, low, high, rate = (np.array(v) for v in zip(*bars))
    one = lambda v: np.array([v])
    y = _harvest(ACTIONS[None] * xmax, one(lev), one(p0), one(close[-1]), one(low.min()),
                 one(high.max()), one((rate * close).sum()))[0]
    for k, u in enumerate(ACTIONS):
        acct = Accounts(1, c)
        acct.open(0, 1000.0)
        acct.execute(np.array([0]), np.array([u * xmax * 1000.0 / p0]), p0, np.array([lev]))
        for cl, lo, hi, f in bars:
            acct.liquidate(hi, lo)
            if f:
                acct.settle_funding(f, cl)
        assert y[k] == pytest.approx(np.log(acct.equity(close[-1])[0] / 1000.0), abs=1e-9)
    y = _harvest(ACTIONS[None] * xmax, one(lev), one(p0), one(55000.0), one(54000.0),
                 one(60000.0), one(0.0))[0]                # through a 10x long's liquidation
    assert y[-1] == pytest.approx(np.log(1 - xmax / lev))  # its whole margin is gone
    assert y[0] > 0                                        # the full short gained


def test_crops_grow_only_on_the_path_after_planting(monkeypatch):
    rows = seconds(1500)
    k = np.arange(len(rows), dtype=float)
    for col in ("open", "low", "mark_low"):
        rows[:, C[col]] = 60000.0 + k                      # each second opens where the last closed
    for col in ("close", "high", "mark_high", "mark_close"):
        rows[:, C[col]] = 60001.0 + k
    rows[:, C["funding_rate"]] = 0.0
    seen, real = [], life_mod._harvest

    def spy(x, lev, p0, last, low, high, fund):
        seen.append((p0.copy(), last.copy(), low.copy(), high.copy()))
        return real(x, lev, p0, last, low, high, fund)

    monkeypatch.setattr(life_mod, "_harvest", spy)
    pl, life = new_life()
    live(pl, life, rows)
    p0, last, low, high = (np.concatenate(v) for v in zip(*seen))
    assert len(p0) > 1000
    assert np.array_equal(low, p0)          # the path starts right after planting, not before
    assert np.array_equal(high, last)
    grown = set(np.unique(last - p0))       # seconds it grew: whole bars of its own band
    assert grown <= {h * T for h in (1, 2, 4, 8) for T in TAU} and len(grown) > 4


def test_migration_goes_only_where_there_is_room():
    pl, life = new_life(n=1)
    live(pl, life, seconds(5))
    i = int(np.nonzero(life.alive)[0][0])
    cap = pl.capacity()
    dens = cap.copy()                                      # everywhere full...
    dens[3, 3] = 0                                         # ...but one neighbor
    life.explore[i] = 1.0                                  # exploring: anywhere with room
    for _ in range(5):
        life.x[i], life.y[i] = 2, 3
        life._migrate(i, dens.copy(), cap)
        assert (life.x[i], life.y[i]) == (3, 3)
    life.explore[i] = 0.0                                  # exploiting: only toward better harvests
    pl.growth[:] = 0.0
    pl.growth[1, 3] = 1.0                                  # better, but full
    life.x[i], life.y[i] = 2, 3
    life._migrate(i, dens.copy(), cap)
    assert (life.x[i], life.y[i]) == (2, 3)
    pl.growth[3, 3] = 0.5
    life._migrate(i, dens.copy(), cap)
    assert (life.x[i], life.y[i]) == (3, 3)


def test_division_spreads_to_free_neighbors_or_displaces_the_weakest_here():
    pl, life = new_life(n=3)
    live(pl, life, seconds(5))
    a, b, c = np.nonzero(life.alive)[0][:3]
    life.x[[a, b]], life.y[[a, b]] = 2, 3
    cap = pl.capacity()
    ratio = np.ones(life.N)
    ratio[a], ratio[b] = 2.0, 0.9
    wealth_a, wealth_b = float(life.equity(a)), float(life.equity(b))
    life._divide(a, ratio, cap.copy(), cap)                # crowded everywhere: b makes room
    assert life.counts["displaced"] == 1 and life.alive[b]
    assert life.withdrawn == pytest.approx(wealth_b)       # what b held leaves with it
    assert life.root[b] == life.root[a] and life.gen[b] == life.gen[a] + 1
    assert (life.x[b], life.y[b]) == (2, 3)
    assert life.equity(a) == pytest.approx(life.equity(b))
    assert life.equity(a) + life.equity(b) == pytest.approx(wealth_a)
    ratio[b] = 3.0                                         # nobody weaker here: no room, no child
    born = life.counts["born"]
    life._divide(a, ratio, cap.copy(), cap)
    assert life.counts["born"] == born
    dens = cap.copy()
    dens[3, 3] = 0                                         # a free neighbor
    before = float(life.equity(a))
    life._divide(a, ratio, dens, cap)
    child = int(np.nonzero(life.alive & (life.x == 3) & (life.y == 3))[0][0])
    assert life.root[child] == life.root[a]
    assert life.equity(a) + life.equity(child) == pytest.approx(before)


def test_panspermia_fills_only_places_with_room():
    pl, life = new_life(width=3, n=100, base_capacity=1)
    live(pl, life, seconds(1))
    assert life.alive.sum() == pl.X * pl.Y                 # one per place, no more
    assert (life._density() <= pl.capacity()).all()


def test_the_gate_learns_from_whether_its_episodes_paid():
    pl, life = new_life(n=2)
    live(pl, life, seconds(5))
    i, j = np.nonzero(life.alive)[0][:2]
    idx = np.array([i, j])
    life.gate_theta[idx], life.gate_w[idx] = 1.0, 0.0      # (and keep it shut now)
    life.episode_due[idx], life.episode_grip[idx], life.drift[idx] = 50, 0.0, 0.0
    life.grip[i], life.grip[j] = 0.5, -0.5                 # i's last episode paid, j's did not
    opened = life._gate(idx, 50)
    assert not opened.any()
    assert life.gate_theta[i] < 1.0 < life.gate_theta[j]   # i will engage more readily


def test_salient_organs_are_kept_and_the_rest_wear_away():
    pl, life = new_life(n=1)
    live(pl, life, seconds(5))
    i = int(np.nonzero(life.alive)[0][0])
    life.age[i], life.recent_n[i] = YOUTH, R
    life.integrity[i] = 1.0
    before = life.prog[i].copy()

    def fixed(idx):                                        # two organs carry all the salience
        life.salience[idx] = (0.5, 0.5, 0, 0, 0, 0)
        return np.ones(len(idx))

    life.perspective = fixed
    for t in range(300):
        life.decisions[i] = 8 * (t + 1)
        life._maintain(np.array([i]), int(life.y[i]))
    kept = (life.prog[i] == before).all(1)
    assert kept[:2].all() and not kept[2:].any()
    assert life.integrity[i, :2].min() > 0.9


def test_observatories_are_kept_up_by_those_who_find_them_salient():
    pl, life = new_life(n=1)
    live(pl, life, seconds(5))
    i = int(np.nonzero(life.alive)[0][0])
    x, y = life.x[i], life.y[i]
    pl.obs_integrity[x, y] = (0.5, 0.0)
    life.prog[i, :, :2] = (RECEPTORS.index("close"), ZERO)
    life.prog[i, 0, 0], life.prog[i, 1, 0] = ZERO + 1, ZERO + 2   # read observatories 0 and 1
    life._repair_observatories(np.array([i]), y, np.array([[3.0, 3.0, 0, 0, 0, 0]]))
    assert pl.obs_integrity[x, y, 0] == pytest.approx(0.65)
    assert pl.obs_integrity[x, y, 1] == 0.0                # a fallen one is not raised by use


def test_each_organism_anticipates_what_matters_to_it():
    pl, life = new_life(n=2)
    live(pl, life, seconds(5))
    i, j = np.nonzero(life.alive)[0][:2]
    good = np.array([RECEPTORS.index("close"), ZERO, 1, 2, 1])
    bad = np.array([RECEPTORS.index("volume"), ZERO, 0, 0, 0])
    body = life.body_state(np.array([i, j]))
    for _ in range(300):
        life._learn_anticipation(np.array([i, i, j, j]), body[[0, 0, 1, 1]],
                                 np.array([good, bad, good, bad]),
                                 np.log([3.0, 0.05, 0.05, 3.0]))   # opposite lessons
    zi = life._anticipate(i, body[0], np.array([good, bad]))
    zj = life._anticipate(j, body[1], np.array([good, bad]))
    assert zi[0] > zi[1] and zj[1] > zj[0]


def test_life_on_the_planet_runs_and_resumes_exactly():
    rows = seconds(2400, seed=3)
    pl, life = new_life(n=40)
    live(pl, life, rows[:1200])
    pl2, life2 = pickle.loads(pickle.dumps((pl, life)))
    live(pl, life, rows[1200:])
    live(pl2, life2, rows[1200:])
    assert np.array_equal(life.equity(), life2.equity())
    c = life.census()
    assert c == life2.census()
    assert c["population"] >= 40 and np.isfinite(life.equity()[life.alive]).all()
    assert c["counts"]["episodes"] > 0 and c["counts"]["organs_made"] > 0
    assert sum(c["by_band"]) == c["population"] == np.sum(c["density"])
    assert c["net"] == pytest.approx(float(life.equity()[life.alive].sum()) + life.withdrawn
                                     - life.injected)
