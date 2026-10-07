import datetime as dt
import io
import pickle
import zipfile

import numpy as np
import pytest

from evotrader import data
from evotrader.config import Config
from evotrader.exchange import Accounts, Brackets
from evotrader.features import HISTORY, N_FEATURES, TimeframeBook
from evotrader.world import World

MIN = 60_000


def cfg(**kw):
    c = Config(capacity=20, n_control=4, min_population=10, min_per_niche=1,
               timeframes=(60, 300), seed=1)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def account(capital=1000.0, **kw):
    a = Accounts(1, cfg(**kw))
    a.open(0, capital)
    return a


def trade(a, target_q, price, lev):
    return a.execute(np.array([0]), np.array([float(target_q)]), price, np.array([float(lev)]))


def flat_bars(n, price=60000.0):
    return [(k * MIN, price, price, price, price, 1.0, price, price, price, 0.0) for k in range(n)]


# --------------------------------------------------------------- accounting
def test_open_long_pays_fee_and_half_spread_only():
    a = account()
    trade(a, 0.1, 60000.0, 10)
    assert a.q[0] == pytest.approx(0.1)
    assert a.entry[0] == pytest.approx(60000.05)
    assert a.margin[0] == pytest.approx(0.1 * 60000.05 / 10)
    fee = 0.1 * 60000.05 * 0.0005
    assert a.fees[0] == pytest.approx(fee)
    assert a.equity(60000.0)[0] == pytest.approx(1000 - fee - 0.1 * 0.05)


def test_round_trip_profit_is_move_minus_costs():
    a = account()
    trade(a, 0.1, 60000.0, 10)
    trade(a, 0.0, 61000.0, 10)
    fees = 0.1 * 60000.05 * 0.0005 + 0.1 * 60999.95 * 0.0005
    assert a.q[0] == 0 and a.margin[0] == 0
    assert a.wallet[0] == pytest.approx(1000 + 0.1 * 1000 - fees - 0.1 * 0.1)


def test_short_profits_when_price_falls():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, -0.1, 60000.0, 20)
    assert a.equity(57000.0)[0] == pytest.approx(1000 + 300)
    trade(a, 0.0, 57000.0, 20)
    assert a.wallet[0] == pytest.approx(1300)


def test_flip_closes_then_opens_the_other_way():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, 0.1, 60000.0, 10)
    trade(a, -0.05, 62000.0, 10)
    assert a.q[0] == pytest.approx(-0.05)
    assert a.entry[0] == pytest.approx(62000.0)
    assert a.margin[0] == pytest.approx(0.05 * 62000 / 10)
    assert a.equity(62000.0)[0] == pytest.approx(1200.0)


def test_partial_close_releases_margin_proportionally():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, 0.1, 60000.0, 10)
    trade(a, 0.04, 60000.0, 10)
    assert a.q[0] == pytest.approx(0.04)
    assert a.margin[0] == pytest.approx(240.0)
    assert a.wallet[0] == pytest.approx(760.0)


def test_orders_respect_lot_size_min_notional_and_wallet():
    a = account()
    trade(a, 0.0014, 60000.0, 10)                  # 0.001 BTC = 60 USDT < 100 minimum
    assert a.q[0] == 0
    trade(a, 1.0, 60000.0, 1)                      # can only margin what the wallet holds
    assert a.q[0] == pytest.approx(0.016)
    assert a.wallet[0] >= 0


def test_liquidation_prices_match_binance_isolated_formula():
    a = account(taker_fee=0.0, half_spread=0.0)
    trade(a, 0.1, 60000.0, 50)                     # margin 120, maintenance rate 0.4%
    long_liq = (120 - 0.1 * 60000) / (0.1 * 0.004 - 0.1)
    assert a.liquidation_price(np.array([0]))[0] == pytest.approx(long_liq)
    assert len(a.liquidate(60000.0, long_liq + 0.01)) == 0
    assert list(a.liquidate(60000.0, long_liq - 0.01)) == [0]
    assert a.q[0] == 0 and a.liquidations[0] == 1
    assert a.wallet[0] == pytest.approx(880.0)     # isolated: only the margin is lost

    b = account(taker_fee=0.0, half_spread=0.0)
    trade(b, -0.1, 60000.0, 50)
    short_liq = (120 + 0.1 * 60000) / (0.1 * 0.004 + 0.1)
    assert b.liquidation_price(np.array([0]))[0] == pytest.approx(short_liq)
    assert list(b.liquidate(short_liq + 0.01, 60000.0)) == [0]


def test_funding_longs_pay_shorts_receive():
    a = Accounts(2, cfg(taker_fee=0.0, half_spread=0.0))
    a.open(0, 1000.0)
    a.open(1, 1000.0)
    a.execute(np.array([0, 1]), np.array([0.1, -0.1]), 60000.0, np.array([10.0, 10.0]))
    a.settle_funding(0.0001, 60000.0)
    assert a.equity(60000.0) == pytest.approx([999.4, 1000.6])


def test_brackets_cap_size_by_leverage_and_set_maintenance():
    br = Brackets(Config().brackets)
    assert br.max_notional(np.array([125, 100, 50, 20])) == pytest.approx([5e4, 2.5e5, 3e6, 1.5e7])
    mmr, cum = br.tier(np.array([40_000.0, 60_000.0]))
    assert mmr == pytest.approx([0.004, 0.005]) and cum == pytest.approx([0.0, 50.0])


def test_split_halves_a_leveraged_account_exactly():
    a = Accounts(2, cfg())
    a.open(0, 1000.0)
    trade(a, 0.1, 60000.0, 25)
    before = a.equity(61000.0)[0]
    liq = a.liquidation_price(np.array([0]))[0]
    a.split(0, 1)
    assert a.equity(61000.0) == pytest.approx([before / 2, before / 2])
    assert a.liquidation_price(np.array([0, 1])) == pytest.approx([liq, liq])


# ------------------------------------------------------------------- world
def test_flat_market_loses_exactly_the_fees():
    w = World(cfg(half_spread=0.0), 60)
    for b in flat_bars(3 * 1440):
        w.step(*b)
    s = w.snapshot()
    assert s["totals"]["fees"] > 0
    assert s["net_pnl"] == pytest.approx(-s["totals"]["fees"], abs=1e-6)


def test_decisions_fill_at_next_bar_open():
    w = World(cfg(taker_fee=0.0, half_spread=0.0), 60)
    bars = flat_bars(HISTORY + 1)
    for b in bars:
        w.step(*b)
    i = int(np.nonzero(w.alive & ~w.control & (w.tf == 60))[0][0])
    w.pending[i] = 1.0
    w.lev[i], w.frac[i] = 2.0, 0.5
    t = bars[-1][0] + MIN
    w.step(t, 61000, 61000, 61000, 61000, 1, 61000, 61000, 61000, 0.0)
    assert w.acct.entry[i] == pytest.approx(61000.0)


def test_checkpoint_resumes_deterministically():
    bars = list(data.synthetic_bars(4000, seed=5))
    a = World(cfg(), 60)
    for b in bars[:2000]:
        a.step(*b)
    b_world = pickle.loads(pickle.dumps(a))
    for b in bars[2000:]:
        a.step(*b)
        b_world.step(*b)
    assert np.array_equal(a.equity(), b_world.equity())
    assert a.next_id == b_world.next_id


# -------------------------------------------------------------------- books
def test_book_aggregates_ohlcv():
    book = TimeframeBook(300)
    closed = [book.add(k * MIN, MIN, o, o + 2, o - 1, o + 1, 1.0)
              for k, o in enumerate([10, 11, 12, 13, 14])]
    assert closed == [False, False, False, False, True]
    assert book.closes[-1] == 15 and book.highs[-1] == 16
    assert book.lows[-1] == 9 and book.vols[-1] == 5.0


def test_features_finite_and_bounded():
    book = TimeframeBook(60)
    rng = np.random.default_rng(0)
    p = 100.0
    for k in range(HISTORY + 1):
        p *= np.exp(rng.normal(0, 0.01))
        book.add(k * MIN, MIN, p, p * 1.01, p * 0.99, p, rng.random() + 0.1)
    f = book.features()
    assert f.shape == (N_FEATURES,) and np.all(np.isfinite(f)) and np.all(np.abs(f) <= 2)


# --------------------------------------------------------------------- data
def _zip(lines, header=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        body = ([header] if header else []) + [",".join(map(str, r)) for r in lines]
        zf.writestr("x.csv", "\n".join(body))
    return buf.getvalue()


def _ms(*args):
    return int(dt.datetime(*args, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _klines(start, n, price):
    return [[start + k * MIN, price, price + 5, price - 5, price + k, 1, start + k * MIN + MIN - 1]
            for k in range(n)]


def test_parse_klines_handles_microsecond_stamps():
    rows = [[1735689600000000, 1, 2, 0.5, 1.5, 9, 1735689659999999]]
    assert data.parse_klines(_zip(rows)) == [(1735689600000, 1.0, 2.0, 0.5, 1.5, 9.0)]


def test_build_market_joins_spot_perp_mark_and_funding(tmp_path, monkeypatch):
    aug = _ms(2019, 8, 31, 7, 58)                  # spot era, crosses 08:00 settlement
    sep_spot = _ms(2019, 9, 1, 0, 0)
    listing = _ms(2019, 9, 8, 7, 58)               # perpetual starts mid-month
    files = {
        "spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2019-08.zip": _zip(_klines(aug, 3, 9000)),
        "spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2019-09.zip":
            _zip(_klines(sep_spot, 2, 10000) + _klines(listing, 3, 10000)),
        "futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2019-09.zip":
            _zip(_klines(listing, 3, 10100), header="open_time,open,high,low,close,volume,close_time"),
        "futures/um/monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2019-09.zip":
            _zip([[listing + MIN, 0, 10200, 10000, 10150, 0, 0]]),
        "futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2019-09.zip":
            _zip([[_ms(2019, 9, 8, 8) + 3, 8, 0.00025]],
                 header="calc_time,funding_interval_hours,last_funding_rate"),
    }
    monkeypatch.setattr(data, "_get", lambda url, cache_dir=None: files.get(url.split("/data/", 1)[1]))
    out = str(tmp_path / "btc.csv")
    n = data.build_market(out, start="2019-08-01", log=lambda *a, **k: None)
    rows = list(data.iter_csv(out))
    assert n == len(rows) == 3 + 2 + 3
    times = [r[0] for r in rows]
    assert times == sorted(times)
    by_t = {r[0]: r for r in rows}
    assert by_t[aug + MIN][9] == pytest.approx(0.0001)          # assumed, bar ending 08:00
    assert by_t[aug][9] == 0.0
    assert by_t[sep_spot][4] == 10000                           # spot fills days before listing
    assert by_t[listing][4] == 10100                            # perpetual from listing on
    assert by_t[listing + MIN][6:9] == (10200, 10000, 10150)    # mark price joined
    assert by_t[listing + MIN][9] == pytest.approx(0.00025)     # real settlement at 08:00
    assert by_t[listing][6:9] == (10105, 10095, 10100)          # no mark row: trade price
    with open(out) as fh:
        sources = [line.rsplit(",", 1)[1].strip() for line in fh.readlines()[1:]]
    assert sources == ["spot"] * 5 + ["perp"] * 3
    assert data.build_market(out, start="2019-08-01", log=lambda *a, **k: None) == 0


class FakeClient:
    def __init__(self, bars, funding_after_calls):
        self.bars, self.calls, self.after = bars, 0, funding_after_calls

    def klines(self, start_ms, symbol="BTCUSDT", mark=False, limit=1500):
        return [b for b in self.bars if b[0] >= start_ms][:limit]

    def funding(self, start_ms, end_ms, symbol="BTCUSDT"):
        self.calls += 1
        return {_ms(2024, 1, 1, 8): 0.0003} if self.calls > self.after else {}


def test_live_feed_waits_for_the_funding_rate(monkeypatch):
    start = _ms(2024, 1, 1, 7, 58)
    bars = [(t, 1, 2, 0, 1, 1, t + MIN - 1) for t in (start, start + MIN, start + 2 * MIN)]
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    client = FakeClient(bars, funding_after_calls=2)
    feed = data.live_bars(start, client=client, log=lambda m: None)
    got = [next(feed) for _ in range(3)]
    assert [g[0] for g in got] == [b[0] for b in bars]
    assert [g[9] for g in got] == [0.0, 0.0003, 0.0]
    assert client.calls == 3                       # held back until the rate appeared
